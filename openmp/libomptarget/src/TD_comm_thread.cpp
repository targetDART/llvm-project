#include "omptarget.h"
#include "TargetDART.h"
#include "TD_common.h"
#include "TD_scheduling.h"
#include "TD_comm_thread.h"


bool doRepartition = false;

//TODO: add Communication thread implementation

void td_schedule_thread_loop(bool *continue_loop) {
    int iter = 0;
    while (*continue_loop) {
        if (iter == ITER_TILL_REPARTITION || doRepartition) {
            iter = 0;
            td_global_reschedule(TD_ANY_AF);
        }
        iter++;
        
        td_iterative_schedule(TD_ANY_AF);
    }
}


void td_trigger_global_repartitioning(td_device_affinity affinity) {
    doRepartition = true;
}

void td_exec_thread_loop(td_device_affinity affinities...) {
    //TODO implement task polling and execution.
}
