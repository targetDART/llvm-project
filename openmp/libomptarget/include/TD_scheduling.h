
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


typedef struct td_progression_t{
    uint64_t    advancement;
    uint64_t    compute_load;
    uint64_t    memory_load;
    double      time_load;
};

extern std::vector<td_progression_t> td_global_progression;

void td_advance(uint64_t value);

class TD_Device_Queue {

    private:

    alignas(64) std::atomic<double> cost{0};

    TD_Task_Queue base_queue;

    TD_Task_Queue remote_queue;

    public:

    tdrc add_remote_task(td_task_t* task);

    tdrc add_local_task(td_task_t* task);

    tdrc poll_task(td_task_t* task);

    TD_Device_Queue();
    ~TD_Device_Queue();
};


extern std::vector<TD_Device_Queue> td_device_list;


#endif