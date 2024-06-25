#ifndef _TARGETDART_THREADING_H
#define _TARGETDART_THREADING_H

#include "task.h"
#include "communication.h"
#include "scheduling.h"
#include <__atomic/atomic.h>
#include <functional>
#include <thread>
#include <vector>

class TD_Thread_Manager {
private:
    std::thread scheduler_th;
    std::vector<std::thread> executor_th;

    TD_Scheduling_Manager *schedule_man;
    TD_Communicator *comm_man;

    bool is_finalizing = false;

    int32_t physical_device_count;

    std::atomic<bool> scheduler_done{false};

    /**
    * Defines the routine performed by the dedicated scheduling thread.
    */
    std::function<void(int)> schedule_thread_loop;

    /**
    * Defines the routine performed by the executor threads.
    * The parameter ptr defines the target device this thread is assigned to. 
    * The device id defines which affinities are relevant and which queues are accessible.
    */
    std::function<void(int)> exec_thread_loop;

    /**
    * Reads the environment variable TD_MANAGEMENT
    */
    tdrc get_thread_placement_from_env(std::vector<int> *placements);

    // Initializes threads and starts their execution  
    tdrc init_threads(std::vector<int> *assignments);

public: 
    TD_Thread_Manager(int32_t device_count, TD_Communicator *comm, TD_Scheduling_Manager *sched);
    ~TD_Thread_Manager();
};


#endif //_TARGETDART_THREADING_H