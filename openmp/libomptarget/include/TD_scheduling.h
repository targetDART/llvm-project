
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

class TD_Device_Queue {

    private:

    alignas(64) std::atomic<COST_DATA_TYPE> cost{0};

    TD_Task_Queue base_queue;

    TD_Task_Queue remote_queue;

    td_device_type device_type;

    public:

    tdrc add_remote_task(td_task_t* task);

    tdrc add_local_task(td_task_t* task);

    tdrc poll_task(td_task_t* task);

    COST_DATA_TYPE get_load();

    td_device_type get_device_type();

    TD_Device_Queue(td_device_type type=TD_GPU);
    ~TD_Device_Queue();
};

typedef struct td_sort_cost_tuple_t{
    COST_DATA_TYPE        cost;
    int                   id;
} td_sort_cost_tuple_t;

extern std::vector<TD_Device_Queue> td_device_list;

tdrc td_add_to_load_local(td_task_t * task);
tdrc td_add_to_load_remote(td_task_t * task);


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

#endif