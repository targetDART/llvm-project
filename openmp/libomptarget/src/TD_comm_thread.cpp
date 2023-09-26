#include "omptarget.h"
#include "TargetDART.h"
#include "TD_common.h"
#include "TD_scheduling.h"
#include "TD_comm_thread.h"
#include "TD_communication.h"
#include <cstdlib>


bool doRepartition = false;

//TODO: add Communication thread implementation

void td_schedule_thread_loop(bool *continue_loop) {
    int iter = 0;
    while (*continue_loop) {
        if (iter == ITER_TILL_REPARTITION || doRepartition) {
            iter = 0;
            td_global_reschedule(TD_ANY);
        }
        iter++;
        
        td_iterative_schedule(TD_ANY);
    }
}


void td_trigger_global_repartitioning(td_device_affinity affinity) {
    doRepartition = true;
}



void td_exec_thread_loop(td_device_affinity affinity, int deviceID) {
    while (!td_finalize) {
        td_task_t *task;

        if (td_get_next_task(affinity, deviceID, task) == TARGETDART_SUCCESS) {
            //execute the task on your own device
            int return_code = td_invoke_task(deviceID, task);
            task->return_code = return_code;
            if (return_code == TARGETDART_FAILURE) {                
                handle_error_en(-1, "Task execution failed.");
                //exit(-1);
            }
            //finalize after the task finished
            if (task->local_proc != td_comm_rank) {
                td_send_task_result(task);
            } else {
                td_signal(task->uid);
            }
        } 
    }
}

// executes the task on the targeted Device 
int td_invoke_task(int DeviceId, td_task_t* task) {
  return __tgt_target_kernel(task->Loc, DeviceId, task->num_teams, task->thread_limit, (void *) apply_image_base_address(task->host_base_ptr, true), task->KernelArgs);
}
