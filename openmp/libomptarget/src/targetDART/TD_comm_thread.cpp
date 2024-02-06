#include "mpi.h"
#include "omp.h"
#include "omptarget.h"
#include "targetDART/TargetDART.h"
#include "targetDART/TD_common.h"
#include "targetDART/TD_scheduling.h"
#include "targetDART/TD_comm_thread.h"
#include "targetDART/TD_communication.h"
#include <atomic>
#include <cstdio>
#include <functional>
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <pthread.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <vector>


std::atomic<bool> *td_start_finalize;
std::atomic<bool> *td_finalize_executor;
std::atomic<bool> *td_scheduler_done;


bool doRepartition = false;
std::vector<std::thread> spawned_threads;

std::vector<td_device_affinity> affinity_assignment;

/**
* Defines the routine performed by the dedicated scheduling thread.
*/
[[nodiscard]] void __td_schedule_thread_loop(int deviceID) {
    int iter = 0;
    DB_TD("Starting scheduler thread");
    while (!td_test_finalization(td_get_total_load(), td_start_finalize->load()) && td_comm_size > 1) {
        if (iter == 800000 || doRepartition) {
            iter = 0;
            //td_global_reschedule(TD_ANY);
            doRepartition = false;
            DB_TD("ping");
        }
        iter++;        
        td_iterative_schedule(TD_GPU);

        td_test_and_receive_results();
    }


    td_scheduler_done->store(true);
    DB_TD("Scheduling thread finished");    
}

//Declares the scheduling as a callable parameter
std::function<void(int)> scheduler_func(__td_schedule_thread_loop);


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
[[nodiscard]] void __td_exec_thread_loop(int deviceID) {

    DB_TD("Starting executor thread for device %d", deviceID);
    td_device_affinity affinity = affinity_assignment.at(deviceID);
    while (!td_finalize_executor->load() || !td_scheduler_done->load()) {
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
}


//Declares the execution as a callable parameter
std::function<void(int)> exec_func(__td_exec_thread_loop);

// pins 
void __td_pin_and_workload(std::thread* thread, int core, std::function<void(int)> *work, int deviceID) {
    if (core != -1) {
    
        cpu_set_t cpuset;// = CPU_ALLOC(N);

        int s;

        CPU_ZERO(&cpuset);
        CPU_SET(core, &cpuset);

        // pin current thread to core
        // WARNING: Only works on Unix systems
        s = pthread_setaffinity_np(thread->native_handle(), sizeof(cpu_set_t), &cpuset);
        if (s != 0) 
            handle_error_en(s, "Failed to initialize thread");
    }

    //Do work
    work[0](deviceID);
}


/*
* initializes the scheduling and executor threads.
* The threads are pinned to the cores defined in the parameters.
* the number of exec placements must be equal to the omp_get_num_devices() + 1.
*/
tdrc td_init_threads(std::vector<int> *assignments) {

    
    td_start_finalize = new std::atomic<bool>(false);  
    td_finalize_executor = new std::atomic<bool>(false);
    td_scheduler_done = new std::atomic<bool>(false);

    DB_TD("Creating %d Threads", omp_get_num_devices() + 2);

    spawned_threads = std::vector<std::thread>(omp_get_num_devices() + 2);

    spawned_threads[0] = std::thread(__td_pin_and_workload, &spawned_threads[0], (*assignments)[0], &scheduler_func, -1);

    affinity_assignment = std::vector<td_device_affinity>(omp_get_num_devices() + 1, TD_GPU);
    affinity_assignment.at(omp_get_num_devices()) = TD_CPU;

    //initialize all executor threads
    for (int i = 0; i <= omp_get_num_devices(); i++) {
        spawned_threads[i+1] = std::thread(__td_pin_and_workload, &spawned_threads[i+1], (*assignments)[i+1], &exec_func, i);
    }

    //delete affinity_assignment;
    delete assignments;
    DB_TD("spawned management threads");
    return TARGETDART_SUCCESS;
}

/**
* synchronizes all processes to ensure all tasks are finished.
*/
tdrc td_finalize_threads() {
    
    DB_TD("begin finalization of targetDARTLib, wait for remaining work");
    td_start_finalize->store(true);  
    td_finalize_executor->store(true);

    DB_TD("Synchronized threads start joining %d managment threads", spawned_threads.size());
    for (int i = 0; i < spawned_threads.size(); i++) {         
        DB_TD("joining thread: %d", i);   
        spawned_threads.at(i).join();
        DB_TD("joined thread: %d", i);
    }

    DB_TD("finalized managment threads");

    return TARGETDART_SUCCESS;
}