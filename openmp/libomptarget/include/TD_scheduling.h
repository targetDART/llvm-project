
#ifndef _OMPTARGET_TD_SCHEDULING_H
#define _OMPTARGET_TD_SCHEDULING_H

//TODO: define scheduling interface for TargetDART

#include <cstdint>
#include <vector>
#include "omptarget.h"
#include "device.h"
#include "TD_common.h"
#include "TD_queue.h"

class TD_Device_Queue {

    private:

    alignas(64) std::atomic<uint64_t> cost{0};

    TD_Task_Queue base_queue;

    TD_Task_Queue remote_queue;

    public:

    tdrc add_remote_task(td_task_t* task);

    tdrc add_local_task(td_task_t* task);

    tdrc poll_task(td_task_t* task);

    TD_Device_Queue();
};

class TD_Node {
    private:
    std::vector<TD_Device_Queue*> td_devices;

    public:
    double get_total_cost();

    double get_total_tasks();
};



tdrc td_init_devices(int64_t num);

tdrc td_finalize_devices();

#endif