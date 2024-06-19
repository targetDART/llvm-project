#include "../include/threading.h"
#include <functional>

/**
* Defines the routine performed by the dedicated scheduling thread.
*/
[[nodiscard]] void __td_schedule_thread_loop(int deviceID) {
    int iter = 0;
    DP("Starting scheduler thread");
    while (!td_test_finalization(td_get_total_load(), td_start_finalize->load()) && td_comm_size > 1) {
        if (iter == 800000 || doRepartition) {
            iter = 0;
            //td_global_reschedule(TD_ANY);
            doRepartition = false;
            DP("ping");
        }
        iter++;        
        td_iterative_schedule(ANY);

        td_test_and_receive_results();
    }


    td_scheduler_done->store(true);
    DP("Scheduling thread finished");    
}

//Declares the scheduling as a callable parameter
std::function<void(int)> scheduler_func(__td_schedule_thread_loop);

void td_trigger_global_repartitioning(td_device_affinity affinity) {
    doRepartition = true;
}

/**
* Defines the routine performed by the executor threads.
* The parameter ptr defines the target device this thread is assigned to. 
* The device id defines which affinities are relevant and which queues are accessible.
*/
[[nodiscard]] void TD_Thread_Manager::__td_exec_thread_loop(int deviceID) {

    DP("Starting executor thread for device %d", deviceID);
    device_affinity affinity = deviceID == physical_device_count ? CPU : GPU;
    int iter = 0;
    while (!schedule_man->is_empty() || !is_finalizing) {
        td_task_t *task;
        iter++;
        if (iter == 8000000) {
            iter = 0;
            DP("ping from executor of device %d", deviceID);
        }
        if (schedule_man->get_task( deviceID, &task) == TARGETDART_SUCCESS) {
            DP("start execution of task (%ld%ld)", task->uid.rank, task->uid.id);
            //execute the task on your own device
            int return_code = __td_invoke_task(deviceID, task);
            task->return_code = return_code;
            /* if (return_code == TARGETDART_FAILURE) {         
                //handle_error_en(-1, "Task execution failed.");
                //exit(-1);
            } */
            //finalize after the task finished
            if (task->uid.rank != comm_man->rank) {
                comm_man->send_task_result(task);
                schedule_man->notify_task_completion(task->uid, true);
                DP("finished remote execution of task (%d%d)", task->uid.rank, task->uid.id);
            } else {
                schedule_man->notify_task_completion(task->uid, false);
                DP("finished local execution of task (%d%d)", task->uid.rank, task->uid.id);
            }
        } 
    }    

    DP("executor thread for device %d finished", deviceID);
}


//Declares the execution as a callable parameter
std::function<void(int)> exec_func(__td_exec_thread_loop);