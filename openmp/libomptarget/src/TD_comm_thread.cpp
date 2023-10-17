#include "mpi.h"
#include "omp.h"
#include "omptarget.h"
#include "TargetDART.h"
#include "TD_common.h"
#include "TD_scheduling.h"
#include "TD_comm_thread.h"
#include "TD_communication.h"
#include <cstdio>
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <tuple>
#include <vector>


bool doRepartition = false;
std::vector<pthread_t> *spawned_threads;

std::vector<td_device_affinity>* affinity_assignment;

/**
* Defines the routine performed by the dedicated scheduling thread.
*/
[[nodiscard]] void *__td_schedule_thread_loop(void *ptr) {
    int iter = 0;
    while (!td_test_finalization(td_finalize->load()) && td_comm_size > 1) {
        if (iter == ITER_TILL_REPARTITION || doRepartition) {
            iter = 0;
            //td_global_reschedule(TD_ANY);
            doRepartition = false;
        }
        iter++;
        
        td_iterative_schedule(TD_ANY);
    }

    printf("stop scheduler\n");
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

    printf("Device is %d\n", deviceID);
    td_device_affinity affinity = affinity_assignment->at(deviceID);
    while (!td_finalize->load()) {
        td_task_t *task;
        if (td_get_next_task(affinity, deviceID, &task) == TARGETDART_SUCCESS) {
            std::cout << "start task " << task << std::endl;
            //execute the task on your own device
            int return_code = __td_invoke_task(deviceID, task);
            printf("Task local %d\n", task->local_proc);
            printf("return code %d\n", return_code);
            task->return_code = return_code;
            if (return_code == TARGETDART_FAILURE) {                      
                printf("Running task on CPU\n");          
                //handle_error_en(-1, "Task execution failed.");
                //exit(-1);
            }
            //finalize after the task finished
            if (task->local_proc != td_comm_rank) {
                td_send_task_result(task);
            } else {
                std::cout << "finished task " << task << std::endl;
                td_signal(task->uid);
            }
        } 
    }
    printf("stop executor\n");
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
        std::cout << "Failed to initialize thread" << std::endl;

}

/*
* initializes the scheduling and executor threads.
* The threads are pinned to the cores defined in the parameters.
* the number of exec placements must be equal to the omp_get_num_devices() + 1.
*/
tdrc td_init_threads(int scheduler_placement, int *exec_placements) {

    spawned_threads = new std::vector<pthread_t>(omp_get_num_devices() + 2);

    pthread_t scheduler;

    __td_init_and_pin_thread(__td_schedule_thread_loop, scheduler_placement, &spawned_threads->at(0), 0);

    //store thread for later joining
    //spawned_threads->at(0) = &scheduler;

    affinity_assignment = new std::vector<td_device_affinity>(omp_get_num_devices() + 1, TD_GPU);
    affinity_assignment->at(omp_get_num_devices()) = TD_CPU;

    //initialize all executor threads
    for (int i = 0; i <= omp_get_num_devices(); i++) {
        pthread_t executor;
        __td_init_and_pin_thread(__td_exec_thread_loop, exec_placements[i], &spawned_threads->at(i + 1), i);

        //store thread for later joining
        //spawned_threads->at(i + 1) = &executor;
    }

    //delete affinity_assignment;
    printf("spawned threads\n");
    return TARGETDART_SUCCESS;
}

/**
* synchronizes all processes to ensure all tasks are finished.
*/
tdrc td_finalize_threads() {
    printf("begin finalize\n");
    #pragma omp taskwait

    MPI_Barrier(targetdart_comm);
    td_finalize->store(true);

    printf("num threads %d\n", spawned_threads->size());

    for (int i = 0; i < spawned_threads->size(); i++) {         
        printf("joining thread: %d\n", i);   
        pthread_join( spawned_threads->at(i), NULL);
        printf("joined thread: %d\n", i);
    }

    //delete spawned_threads;
    printf("finalize threads\n");

    return TARGETDART_SUCCESS;
}