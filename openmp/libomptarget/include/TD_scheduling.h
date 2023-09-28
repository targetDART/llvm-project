
#ifndef _OMPTARGET_TD_SCHEDULING_H
#define _OMPTARGET_TD_SCHEDULING_H

//TODO: define scheduling interface for TargetDART

#include <cstdint>
#include <vector>
#include "omptarget.h"
#include "device.h"
#include "TD_common.h"
#include "TD_queue.h"
#include "TD_cost_estimation.h"

#define BALANCE_FACTOR 0.8
#define TD_SIMPLE_REACTIVITY_LOAD 1

extern std::vector<td_progression_t> td_global_progression;

typedef struct td_sort_cost_tuple_t{
    COST_DATA_TYPE        cost;
    int                   id;
} td_sort_cost_tuple_t;

extern std::vector<TD_Task_Queue> td_local_task_queues;
extern std::vector<TD_Task_Queue> td_remote_task_queues;
extern std::vector<TD_Task_Queue> td_replica_task_queues;

tdrc td_add_to_load_local(td_task_t * task, int deviceID=0);
tdrc td_add_to_load_remote(td_task_t * task);
tdrc td_add_to_load_replica(td_task_t * task);


// returns the total load on the current device for a given affinity
COST_DATA_TYPE td_get_local_load(td_device_affinity affinity);

/**
* This implements the prefix sum based scheduling proposed here: https://ieeexplore.ieee.org/abstract/document/6270840/
*/
void td_global_reschedule(td_device_affinity affinity);

/**
* this function implments the reactive rescheduling defined in the chameleon project.
*/
void td_iterative_schedule(td_device_affinity affinity);

/**
* This function provides the next task for  given device + affinity.
*/
tdrc td_get_next_task(td_device_affinity affinity, int deviceID, td_task_t *task);

#endif