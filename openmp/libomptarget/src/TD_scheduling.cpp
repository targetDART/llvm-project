#include "TargetDART.h"
#include "omp.h"
#include "omptarget.h"
#include "device.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include "private.h"
#include "mpi.h"
#include "float.h"
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

//Execute remote first
//TODO: test optimal order of queue progression
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

TD_Device_Queue::TD_Device_Queue(bool isGPU){
    
    return;
}
TD_Device_Queue::~TD_Device_Queue(){
    return;
}

double TD_Device_Queue::get_load() {
    return cost.load();
}

/**
* Greedy assignment of tasks to the Device queues of the system
*/
tdrc __td_greedy_assignment(td_task_t* task, bool local) {
    int min_id = 0;
    double min_load = DBL_MAX;
    int num_devices = omp_get_num_devices() + 1;

    if (task->affinity == TD_CPU) {
        if (local) {            
            return td_device_list.at(omp_get_num_devices()).add_local_task(task);
        }
        return td_device_list.at(omp_get_num_devices()).add_remote_task(task);
    } else if (task->affinity == TD_GPU) {
        num_devices -= 1;
    }

    for (int i = 0; i < num_devices; i++) {
        if (td_device_list.at(i).get_load() < min_load) {
            min_id = i;
        }
    }
    if (local) {            
        return td_device_list.at(min_id).add_local_task(task);
    }
    return td_device_list.at(min_id).add_remote_task(task);
}

// adds a task to the local queue with the lowest load
tdrc td_add_to_load_local(td_task_t* task) {
    return __td_greedy_assignment(task, true);
}

// adds a task to the remote queue with the lowest load
tdrc td_add_to_load_remote(td_task_t * task) {
    return __td_greedy_assignment(task, false);
}