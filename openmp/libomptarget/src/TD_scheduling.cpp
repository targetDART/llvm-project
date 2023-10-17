#include "TargetDART.h"
#include "omp.h"
#include "omptarget.h"
#include "device.h"
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include "private.h"
#include "mpi.h"
#include "float.h"
#include <link.h>
#include <queue>
#include <dlfcn.h>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include "TD_scheduling.h"
#include "TD_cost_estimation.h"
#include "TD_queue.h"
#include "TD_communication.h"

std::vector<td_progression_t> td_global_progression;

/**
* 0: TD_ANY affinity queue
* 1: TD_CPU affinity queue
* 2: TD_GPU affinity queue
* 3: TD_FPGA affinity queue
* 4: TD_VEC affinity queue
* 5 to num_devices + 5: device specific queues
*/
std::vector<TD_Task_Queue*>* __td_local_task_queues;
/**
* 0: TD_ANY affinity queue
* 1: TD_CPU affinity queue
* 2: TD_GPU affinity queue
* 3: TD_FPGA affinity queue
* 4: TD_VEC affinity queue
*/
std::vector<TD_Task_Queue*>* __td_remote_task_queues;
/**
* 0: TD_ANY affinity queue
* 1: TD_CPU affinity queue
* 2: TD_GPU affinity queue
* 3: TD_FPGA affinity queue
* 4: TD_VEC affinity queue
*/
std::vector<TD_Task_Queue*>* __td_replica_task_queues;

/**
* 0: local queues
* 1: remote queues
* 2: replica queues
*/
std::vector<std::vector<TD_Task_Queue*>*>* td_queue_classes;

tdrc td_init_task_queues() {
    __td_local_task_queues = new std::vector<TD_Task_Queue*>();
    __td_remote_task_queues = new std::vector<TD_Task_Queue*>();
    __td_replica_task_queues = new std::vector<TD_Task_Queue*>();
    for (int i = 0; i < TD_NUM_AFFINITIES; i++) {
        auto queue = new TD_Task_Queue();
        queue->init();
        __td_local_task_queues->push_back(queue);
        auto queue2 = new TD_Task_Queue();
        queue2->init();
        __td_remote_task_queues->push_back(queue2);
        auto queue3 = new TD_Task_Queue();
        queue3->init();
        __td_replica_task_queues->push_back(queue3);
    }
    for (int i = 0; i <= omp_get_num_devices(); i++) {
        auto queue = new TD_Task_Queue();
        queue->init();
        __td_local_task_queues->push_back(queue);
    }
    td_queue_classes = new std::vector<std::vector<TD_Task_Queue*>*>({__td_local_task_queues, __td_remote_task_queues, __td_replica_task_queues});

    return TARGETDART_SUCCESS;
}

/**
* Greedy assignment of tasks to the Device queues of the system
*/
tdrc __td_greedy_assignment(td_task_t* task, td_queue_class queue, int deviceID=0) {
    
    if (task->affinity == TD_FIXED) {
        return td_queue_classes->at(queue)->at(deviceID + TD_NUM_AFFINITIES)->offer_task(task);
    }

    return td_queue_classes->at(queue)->at(task->affinity)->offer_task(task);
}

// adds a task to the local queue with the lowest load
tdrc td_add_to_load_local(td_task_t* task, int deviceID) {
    return __td_greedy_assignment(task, TD_LOCAL, deviceID);
}

// adds a task to the remote queue with the lowest load
tdrc td_add_to_load_remote(td_task_t * task) {
    return __td_greedy_assignment(task, TD_REMOTE);
}

// adds a task to the replica queue with the lowest load
tdrc td_add_to_load_replica(td_task_t * task) {
    return __td_greedy_assignment(task, TD_REPLICA);
}


// get the cost of a given queue
inline COST_DATA_TYPE __get_queue_cost(std::vector<TD_Task_Queue*> *queue_list, td_device_affinity affinity) {
    return queue_list->at(affinity)->get_cost();
}

// get the cost of the local queue
COST_DATA_TYPE td_get_local_load(td_device_affinity affinity) {
    return __get_queue_cost(__td_local_task_queues, affinity);
}

// get the cost of the remote queue
COST_DATA_TYPE td_get_remote_load(td_device_affinity affinity) {
    return __get_queue_cost(__td_remote_task_queues, affinity);
}

// get the cost of the replica queue
COST_DATA_TYPE td_get_replica_load(td_device_affinity affinity) {
    return __get_queue_cost(__td_replica_task_queues, affinity);
}

// get the load that can be migrated to another process
COST_DATA_TYPE td_get_migratable_load() {
    COST_DATA_TYPE sum = 0;
    for (td_device_affinity affinity : TD_AFFINITIES) {
        sum += td_get_local_load(affinity);
        sum += td_get_remote_load(affinity);
    }
    return sum;
}

// get the complete load on the system
COST_DATA_TYPE td_get_total_load() {
    COST_DATA_TYPE sum = td_get_migratable_load();
    for (td_device_affinity affinity : TD_AFFINITIES) {
        sum += td_get_replica_load(affinity);
    }
    for (int i = TD_NUM_AFFINITIES; i < __td_local_task_queues->size(); i++) {
        sum += __td_local_task_queues->at(i)->get_cost();
    }
    return sum;
}


tdrc td_get_next_task(td_device_affinity affinity, int deviceID, td_task_t **task) {
    tdrc ret_code;

    /**
    1. queue to access: fixed device 
    */
    ret_code = __td_local_task_queues->at(TD_NUM_AFFINITIES + deviceID)->get_task(task);

    if (ret_code == TARGETDART_SUCCESS) {
        return TARGETDART_SUCCESS;
    }

    /**
    2. queue to access: local affinity queue
    */
    ret_code = __td_local_task_queues->at(affinity)->get_task(task);

    if (ret_code == TARGETDART_SUCCESS) {
        return TARGETDART_SUCCESS;
    }

    /**
    3. queue to access: local (TD_ANY) queue
    */
    ret_code = __td_local_task_queues->at(TD_ANY)->get_task(task);

    if (ret_code == TARGETDART_SUCCESS) {
        printf("got task: \n");
        return TARGETDART_SUCCESS;
    }

    /**
    4. queue to access: remote affinity queue
    */
    ret_code = __td_remote_task_queues->at(affinity)->get_task(task);

    if (ret_code == TARGETDART_SUCCESS) {
        return TARGETDART_SUCCESS;
    }

    /**
    5. queue to access: remote (TD_ANY) queue
    */
    ret_code = __td_remote_task_queues->at(TD_ANY)->get_task(task);

    if (ret_code == TARGETDART_SUCCESS) {
        return TARGETDART_SUCCESS;
    }

    /**
    6. queue to access: replica affinity queue
    */
    ret_code = __td_replica_task_queues->at(affinity)->get_task(task);

    if (ret_code == TARGETDART_SUCCESS) {
        return TARGETDART_SUCCESS;
    }

    /**
    7. queue to access: replica (TD_ANY) queue
    */
    ret_code = __td_replica_task_queues->at(TD_ANY)->get_task(task);

    if (ret_code == TARGETDART_SUCCESS) {
        return TARGETDART_SUCCESS;
    }

    return TARGETDART_FAILURE;
}

tdrc __td_get_next_migratable_task(td_device_affinity affinity, td_task_t **task) {
    tdrc ret_code;

    /**
    2. queue to access: local affinity queue
    */
    ret_code = __td_local_task_queues->at(affinity)->get_task(task);

    if (ret_code == TARGETDART_SUCCESS) {
        return TARGETDART_SUCCESS;
    }

    /**
    4. queue to access: remote affinity queue
    */
    ret_code = __td_remote_task_queues->at(affinity)->get_task(task);

    if (ret_code == TARGETDART_SUCCESS) {
        return TARGETDART_SUCCESS;
    }


    return TARGETDART_FAILURE;
}

/**
* Implements the rescheduling of tasks for the local MPI process and its current victim, defined by offset.
* target_load: defines the load the victim should have in total after migration.
* affinity: defines which kinds of tasks should be considered for a rescheduling.
*/
void __td_do_partial_global_reschedule(double target_load, td_device_affinity affinity, int offset) {
    std::vector<td_task_t*> transferred_tasks;
    COST_DATA_TYPE totalcost = 0;
    while (totalcost < BALANCE_FACTOR * target_load) {
        td_task_t *next_task;
        tdrc return_code = __td_get_next_migratable_task(affinity, &next_task);
        if (return_code == TARGETDART_FAILURE) {
            break;
        }
        if (td_get_task_cost(next_task->host_base_ptr, affinity) >= target_load/BALANCE_FACTOR) {
            break;
        } else {
            transferred_tasks.push_back(next_task);
        }
    }
    
    //TODO: think about MPI_pack as well
    for (int t = 0; t < transferred_tasks.size(); t++) {
        td_send_task(td_comm_rank + offset, transferred_tasks.at(t));
    }
}

void td_global_reschedule(td_device_affinity affinity) {
    td_global_sched_params_t params = td_global_cost_communicator(td_get_local_load(affinity));
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
        __td_do_partial_global_reschedule(target_load, affinity, -i);
    }
    //general case transfers successor
    for (int i = 1; i < post_distance; i++) {
        __td_do_partial_global_reschedule(target_load, affinity, i);
    }
    
    long pre_remainder_load = ((long) pre_transfer) % ((long) target_load);    
    __td_do_partial_global_reschedule(pre_remainder_load, affinity, -pre_distance);
    long post_remainder_load = ((long) post_transfer) % ((long) target_load);    
    __td_do_partial_global_reschedule(post_remainder_load, affinity, post_distance);

    //TODO: Receive tasks
    //TODO: Transform task migration to non-blocking to avoid serialization.


}

/**
* swaps two elements from a vector and updates gives the new index of proc_idx, if it overlaps with idx1 or idx2.
*/
template<typename T1>
int __td_coswap(std::vector<T1> *load_vector, int idx1, int idx2, int proc_idx) {
    if (idx1 == idx2) {
        return proc_idx;
    }
    std::swap(load_vector[0][idx1], load_vector[0][idx2]);

    //calculate current index to avoid searching for it later
    int local_idx = proc_idx;
    if (idx1 == local_idx) {
        local_idx = idx2;
    } else if (idx2 == local_idx) {
        local_idx = idx1;
    } 
    return local_idx;
}

/**
* Sorts a vector and provides the new index of the local value.
* The load of the local MPI rank must correspond to the entry idx = rank.
*/
int __td_cosort(std::vector<td_sort_cost_tuple_t> *load_vector) {
    int proc_idx = td_comm_rank;
    //TODO: use more efficient sort implementation merge + insertion 
    for (int i = load_vector->size() - 1; i > 0; i--) {
        COST_DATA_TYPE max = 0;
        int max_idx = -1;
        for (int j = 0; j < i; j++) {
            if (load_vector->at(j).cost > max) {
                max = load_vector->at(j).cost;
                max_idx = j;
            }
            proc_idx = __td_coswap(load_vector, max_idx, i, proc_idx);
        }
    }
    return proc_idx;
}

/**
* Returns 0 iff local_cost = remote_cost
* Returns the desire load to transfer from local to remote, iff local_cost > remote_cost
* Returns the desire load to transfer from remote to local as a negative value, iff local_cost < remote_cost
*/
COST_DATA_TYPE __td_compute_transfer_load(COST_DATA_TYPE local_cost, COST_DATA_TYPE remote_cost) {
    COST_DATA_TYPE result;
    if (local_cost == remote_cost) {
        result = 0;
    } else if (local_cost > remote_cost) {
        result = TD_SIMPLE_REACTIVITY_LOAD;
    } else {
        result = -TD_SIMPLE_REACTIVITY_LOAD;
    }
    return result;
}


void td_iterative_schedule(td_device_affinity affinity) {    
    std::vector<COST_DATA_TYPE> cost_vector = td_global_cost_vector_propagation(td_get_local_load(affinity));
    std::vector<td_sort_cost_tuple_t> combined_vector(cost_vector.size());

    for (int i = 0; i < combined_vector.size(); i++) {
        combined_vector[i].cost = cost_vector[i];
        combined_vector[i].id = i;
    }
    std::sort(combined_vector.begin(), combined_vector.end(), [](td_sort_cost_tuple_t a, td_sort_cost_tuple_t b) 
                                                                                {                                                                                
                                                                                    return a.cost < b.cost;
                                                                                });

    int local_idx = NULL;
    for (int i = 0; i < combined_vector.size(); i++) {
        if (combined_vector.at(i).id == td_comm_rank) {
            local_idx = i;
            break;
        }
    }
/*
    // implement Chameleon based victim selection
    int partner_idx = NULL;
    if (combined_vector.size() % 2 == 0) {
        int half = combined_vector.size() / 2;
        if (local_idx < half) {
            partner_idx = combined_vector.size() + local_idx - half;
        } else {
            partner_idx = local_idx - half;
        }
    } else {
        int half = combined_vector.size() / 2;
        if (local_idx < half) {
            partner_idx = combined_vector.size() + local_idx - half;
        } else {
            partner_idx = local_idx - half - 1;
        }
    }
    partner_idx = combined_vector.size() - local_idx - 1;
    int partner_proc = combined_vector.at(partner_idx).id;

    COST_DATA_TYPE transfer_load = __td_compute_transfer_load(combined_vector.at(local_idx).cost, combined_vector.at(partner_idx).cost);
    
    
    if (transfer_load == 0) {
        return;
    } else if (transfer_load > 0) {
        for (int i = 0; i < TD_SIMPLE_REACTIVITY_LOAD; i++) {
            td_task_t *task; 
            tdrc ret_code = __td_get_next_migratable_task(affinity, &task);
            td_send_task(partner_proc, task);
        }
    } else {
        for (int i = 0; i < TD_SIMPLE_REACTIVITY_LOAD; i++) {
            td_task_t *task = (td_task_t*) std::malloc(sizeof(td_task_t));
            td_receive_task(partner_proc, task);
            td_add_to_load_remote(task);
        }
    } */
}