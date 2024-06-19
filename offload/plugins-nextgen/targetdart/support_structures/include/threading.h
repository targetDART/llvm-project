#include "task.h"
#include "communication.h"
#include "scheduling.h"
#include <functional>
#include <thread>

class TD_Thread_Manager {
private:
    std::thread scheduler_th;
    std::vector<std::thread> executor_th;

    TD_Scheduling_Manager *schedule_man;
    TD_Communicator *comm_man;

    bool is_finalizing = false;

    int32_t physical_device_count;

    /**
    * Defines the routine performed by the dedicated scheduling thread.
    */
    [[nodiscard]] void __td_schedule_thread_loop(int deviceID);

    /**
    * Defines the routine performed by the executor threads.
    * The parameter ptr defines the target device this thread is assigned to. 
    * The device id defines which affinities are relevant and which queues are accessible.
    */
    [[nodiscard]] void __td_exec_thread_loop(int deviceID);

public: 
    TD_Thread_Manager(int32_t physical_device_count);
    ~TD_Thread_Manager();
};