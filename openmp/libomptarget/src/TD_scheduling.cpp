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
#include <vector>
#include "TD_scheduling.h"
#include "TD_cost_estimation.h"
#include "TD_queue.h"
#include "TD_communication.h"


std::vector<td_progression_t> td_global_progression;
std::vector<TD_Device_Queue> td_device_list;

tdrc TD_Device_Queue::add_remote_task(td_task_t* task) {
    cost.fetch_add(td_get_task_cost(task->host_base_ptr, device_type), MEM_ORDER);
    return remote_queue.offerTask(task);
}

tdrc TD_Device_Queue::add_local_task(td_task_t* task){
    cost.fetch_add(td_get_task_cost(task->host_base_ptr, device_type), MEM_ORDER);
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
    
    cost.fetch_sub(td_get_task_cost(task->host_base_ptr, device_type), MEM_ORDER);

    return TARGETDART_SUCCESS;
}

TD_Device_Queue::TD_Device_Queue(td_device_type type){
    device_type = type;
    return;
}

TD_Device_Queue::~TD_Device_Queue(){
    return;
}

COST_DATA_TYPE TD_Device_Queue::get_load() {
    return cost.load();
}

td_device_type TD_Device_Queue::get_device_type(){
    return device_type;
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


COST_DATA_TYPE td_get_local_load(td_device_affinity affinity) {
    COST_DATA_TYPE total_load = 0;

    for (int i = 0; i < td_device_list.size(); i++) {        
        if (affinity == TD_ANY_AF || affinity == TD_CPU_AF) {
            if (td_device_list.at(i).get_device_type() == TD_CPU) {
                total_load += td_device_list.at(i).get_load();
            }
        } 
        if (affinity == TD_ANY_AF || affinity == TD_GPU_AF) {
            if (td_device_list.at(i).get_device_type() == TD_GPU) {
                total_load += td_device_list.at(i).get_load();
            }
        } 
        if (affinity == TD_ANY_AF || affinity == TD_FPGA_AF) {
            if (td_device_list.at(i).get_device_type() == TD_FPGA) {
                total_load += td_device_list.at(i).get_load();
            }
        } 
        if (affinity == TD_ANY_AF || affinity == TD_VECTOR_AF) {
            if (td_device_list.at(i).get_device_type() == TD_VECTOR) {
                total_load += td_device_list.at(i).get_load();
            }
        }
    }
    return total_load;
}


void td_global_reschedule(td_device_affinity affinity) {
    td_global_sched_params_t params = __td_global_cost_communicator(affinity, td_get_local_load(affinity));
    double target_load = ((double) params.total_cost) / td_comm_size;

    double pre_transfer = 0;
    double post_transfer = 0;

    //compute pre_transfer
    if (td_comm_rank != 0) {
        double predecessor_load = ((double) params.prefix_sum) / td_comm_rank;
        pre_transfer = (target_load - predecessor_load) * td_comm_rank;
    }

    //compute post_transfer
    if (td_comm_rank != td_comm_size - 1) {
        double successor_cost = ((double) params.total_cost) - params.local_cost - params.prefix_sum;
        int num_successors = td_comm_size - 1 - td_comm_rank; //inverted rank
        double successor_load = successor_cost/num_successors;
        post_transfer = (target_load - successor_load) * num_successors;
    }

    //calculate num tasks per direktion
    if (pre_transfer < 0) {
        pre_transfer = 0;
    }
    if (post_transfer < 0) {
        post_transfer = 0;
    }

    //compute furthest data transfer
    int pre_distance = pre_transfer/target_load + 1;
    int post_distance = post_transfer/target_load + 1;

    //general case transfers predecessor
    for (int i = 1; i < pre_distance; i++) {
        std::vector<td_task_t*> transferred_tasks;
        COST_DATA_TYPE totalcost = 0;
        while (totalcost < BALANCE_FACTOR * target_load) {
            td_task_t *next_task = {}; //TODO: get task from correct queue
            if (td_get_task_cost(next_task->host_base_ptr, get_device_from_affinity(affinity)) >= target_load/BALANCE_FACTOR) {
                break;
            } else {
                transferred_tasks.push_back(next_task);
            }
        }
        for (int t = 0; t < transferred_tasks.size(); t++) {
            td_send_task(td_comm_rank - i, transferred_tasks.at(t));
        }
    }
    //general case transfers successor
    for (int i = 1; i < post_distance; i++) {
        std::vector<td_task_t*> transferred_tasks;
        COST_DATA_TYPE totalcost = 0;
        while (totalcost < BALANCE_FACTOR * target_load) {
            td_task_t *next_task = {}; //TODO: get task from correct queue
            if (td_get_task_cost(next_task->host_base_ptr, get_device_from_affinity(affinity)) >= target_load/BALANCE_FACTOR) {
                break;
            } else {
                transferred_tasks.push_back(next_task);
            }
        }
        for (int t = 0; t < transferred_tasks.size(); t++) {
            td_send_task(td_comm_rank + i, transferred_tasks.at(t));
        }
    }
    //TODO: turn the inner case into a single funtion
    //Remainder case 
    {
        long pre_remainder_load = ((long) pre_transfer) % ((long) target_load);
        std::vector<td_task_t*> transferred_tasks;
        COST_DATA_TYPE totalcost = 0;
        while (totalcost < BALANCE_FACTOR * pre_remainder_load) {
            td_task_t *next_task = {}; //TODO: get task from correct queue
            if (td_get_task_cost(next_task->host_base_ptr, get_device_from_affinity(affinity)) >= pre_remainder_load/BALANCE_FACTOR) {
                break;
            } else {
                transferred_tasks.push_back(next_task);
            }
        }
        for (int t = 0; t < transferred_tasks.size(); t++) {
            td_send_task(td_comm_rank - pre_distance, transferred_tasks.at(t));
        }
    }
    {
        long post_remainder_load = ((long) post_transfer) % ((long) target_load);
        std::vector<td_task_t*> transferred_tasks;
        COST_DATA_TYPE totalcost = 0;
        while (totalcost < BALANCE_FACTOR * post_remainder_load) {
            td_task_t *next_task = {}; //TODO: get task from correct queue
            if (td_get_task_cost(next_task->host_base_ptr, get_device_from_affinity(affinity)) >= post_remainder_load/BALANCE_FACTOR) {
                break;
            } else {
                transferred_tasks.push_back(next_task);
            }
        }
        for (int t = 0; t < transferred_tasks.size(); t++) {
            td_send_task(td_comm_rank + post_distance, transferred_tasks.at(t));
        }
    }

    //TODO: Receive tasks
    //TODO: Transform task migration to non-blocking to avoid serialization.


}