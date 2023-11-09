#include "mpi.h"
#include "omp.h"
#include "omptarget.h"
#include "TargetDART.h"
#include "TD_common.h"
#include "TD_scheduling.h"
#include "TD_comm_thread.h"
#include "TD_communication.h"
#include <atomic>
#include <cstdio>
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <tuple>
#include <unistd.h>
#include <vector>


bool doRepartition = false;
std::vector<pthread_t> *spawned_threads;

std::vector<td_device_affinity>* affinity_assignment;
pthread_barrier_t   barrier; 

/**
* Defines the routine performed by the dedicated scheduling thread.
*/
[[nodiscard]] void *__td_schedule_thread_loop(void *ptr) {
    int iter = 0;
    DB_TD("Starting scheduler thread");
    while (!td_test_finalization(td_get_total_load(), td_start_finalize->load()) && td_comm_size > 1) {
        if (iter == 200000 || doRepartition) {
            iter = 0;
            //td_global_reschedule(TD_ANY);
            doRepartition = false;
            DB_TD("ping");
        }
        iter++;        
        td_iterative_schedule(TD_ANY);

        td_test_and_receive_results();
    }

    DB_TD("Scheduling thread finished");    
    pthread_barrier_wait (&barrier);
    pthread_exit(NULL);
}


void td_trigger_global_repartitioning(td_device_affinity affinity) {
    doRepartition = true;
}


// executes the task on the targeted Device 
int __td_invoke_task(int DeviceId, td_task_t* task) {
  return __tgt_target_kernel(task->Loc, DeviceId, task->num_teams, task->thread_limit, (void *) apply_image_base_address(task->host_base_ptr, true), task->KernelArgs);
}

/**
* Defines the routine performed by the executor threads.
* The parameter ptr defines the target device this thread is assigned to. 
* The device id defines which affinities are relevant and which queues are accessible.
*/
[[nodiscard]] void *__td_exec_thread_loop(void *ptr) {
    int deviceID = ((long) ptr);

    DB_TD("Starting executor thread for device %d", deviceID);
    td_device_affinity affinity = affinity_assignment->at(deviceID);
    while (!td_finalize_executor->load()) {
        td_task_t *task;
        if (td_get_next_task(affinity, deviceID, &task) == TARGETDART_SUCCESS) {
            DB_TD("start execution of task (%d%d)", task->local_proc, task->uid);
            //execute the task on your own device
            int return_code = __td_invoke_task(deviceID, task);
            task->return_code = return_code;
            if (return_code == TARGETDART_FAILURE) {         
                //handle_error_en(-1, "Task execution failed.");
                //exit(-1);
            }
            //finalize after the task finished
            if (task->local_proc != td_comm_rank) {
                DB_TD("finished remote execution of task (%d%d)", task->local_proc, task->uid);
                td_send_task_result(task);
            } else {
                DB_TD("finished local execution of task (%d%d)", task->local_proc, task->uid);
                td_signal(task);
            }
        } 
    }    

    DB_TD("executor thread for device %d finished", deviceID);
    pthread_exit(NULL);
}

//wrap the thread initialization and pinning
void __td_init_and_pin_thread(void *(*func)(void *), int core, pthread_t *thread, int id) {
    cpu_set_t cpuset;// = CPU_ALLOC(N);

    int s;

    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);

    //initialize atribute
    pthread_attr_t pthread_attr;
    s = pthread_attr_init(&pthread_attr);
    if (s != 0) 
        handle_error_en(s, "pthread_attr_init");
    
    //assign attribute to cpuset
    s = pthread_attr_setaffinity_np(&pthread_attr, sizeof(cpu_set_t), &cpuset);
    if (s != 0) 
        handle_error_en(s, "pthread_attr_setaffinity_np");
            
    s = pthread_create(thread, &pthread_attr,  func, (void*) id);
    pthread_attr_destroy(&pthread_attr);
    if (s != 0) 
        handle_error_en(s, "Failed to initialize thread");

}

/*
* initializes the scheduling and executor threads.
* The threads are pinned to the cores defined in the parameters.
* the number of exec placements must be equal to the omp_get_num_devices() + 1.
*/
tdrc td_init_threads(int scheduler_placement, int *exec_placements) {

    spawned_threads = new std::vector<pthread_t>(omp_get_num_devices() + 2);

    __td_init_and_pin_thread(__td_schedule_thread_loop, scheduler_placement, &spawned_threads->at(0), 0);

    affinity_assignment = new std::vector<td_device_affinity>(omp_get_num_devices() + 1, TD_GPU);
    affinity_assignment->at(omp_get_num_devices()) = TD_CPU;

    //initialize all executor threads
    for (int i = 0; i <= omp_get_num_devices(); i++) {
        __td_init_and_pin_thread(__td_exec_thread_loop, exec_placements[i], &spawned_threads->at(i + 1), i);

    }

    //delete affinity_assignment;
    DB_TD("spawned management threads");
    return TARGETDART_SUCCESS;
}

/**
* synchronizes all processes to ensure all tasks are finished.
*/
tdrc td_finalize_threads() {
    
    pthread_barrier_init (&barrier, NULL, 2);
    DB_TD("begin finalization of targetDARTLib, wait for remaining work");
    td_start_finalize->store(true);  
    pthread_barrier_wait (&barrier);
    td_finalize_executor->store(true);

    DB_TD("Synchronized threads start joining %d managment threads", spawned_threads->size());
    for (int i = 0; i < spawned_threads->size(); i++) {         
        DB_TD("joining thread: %d", i);   
        //TODO: This may Deadlock, look into this.
        //pthread_join( spawned_threads->at(i), NULL);
        DB_TD("joined thread: %d", i);
    }

    delete spawned_threads;
    DB_TD("finalized managment threads");

    return TARGETDART_SUCCESS;
}