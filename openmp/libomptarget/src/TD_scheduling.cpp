#include "TargetDART.h"
#include "omptarget.h"
#include "device.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include "private.h"
#include "mpi.h"
#include <link.h>
#include <queue>
#include <dlfcn.h>
#include <unistd.h>
#include "TD_scheduling.h"
#include "TD_cost_estimation.h"
#include "TD_queue.h"


std::vector<td_progression_t> td_global_progression;
std::vector<TD_Device_Queue> td_device_list;

tdrc TD_Device_Queue::add_remote_task(td_task_t* task) {
    return remote_queue.offerTask(task);
}

tdrc TD_Device_Queue::add_local_task(td_task_t* task){
    return base_queue.offerTask(task);
}

tdrc TD_Device_Queue::poll_task(td_task_t* task){
    task = remote_queue.pollTask(nullptr);
    if (task == nullptr) {
        task = base_queue.pollTask(nullptr);
    }

    if (task == nullptr) {
        return TARGETDART_FAILURE;
    }

    return TARGETDART_SUCCESS;
}

TD_Device_Queue::TD_Device_Queue(){
    return;
}
TD_Device_Queue::~TD_Device_Queue(){
    return;
}