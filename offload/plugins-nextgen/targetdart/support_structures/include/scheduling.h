#ifndef _TARGETDART_SCHEDULING_H
#define _TARGETDART_SCHEDULING_H

#include "queue.h"
#include "task.h"

#include <__atomic/aliases.h>
#include <cstdint>
#include <mutex>
#include <unordered_set>
#include <vector>
#include "PluginManager.h"

class Set_Wrapper {

private:
    std::unordered_set<size_t> *internal_set;
    std::mutex set_mutex;

public: 
    Set_Wrapper();
    ~Set_Wrapper();

    // adds a task identifier to the set
    void add_task(td_uid_t uid);
    // tests if the tas identifier is already part of the set
    // iff yes, it will also be removed and return 0
    bool task_exists(td_uid_t uid);
};

/*
Wrapper class managing all tasks on a single process.
*/
class TD_Scheduling_Manager {
private:

    // Number of physical devices managed by other plugins
    int32_t physical_device_count;

    // A set of task queues defining all relevant affinities
    std::vector<TD_Task_Queue> *affinity_queues;

    // Keeps the number of non completed tasks spawned by this process
    std::atomic<int64_t> active_tasks;

    // The uids of replicated tasks to ensure they won`t be executed twice
    // tasks in this set are defined on a remote process
    Set_Wrapper *finalized_replicated;

    // The uids of replica tasks started on their home machine t ensure they won`t be executed twice
    // tasks in this set are defined on the local process
    Set_Wrapper *started_local_replica;

    // defines the priorities of affinities
    std::vector<int32_t> priorities;

    // array that holds image base addresses
    std::vector<intptr_t> _image_base_addresses;

    // stores all tasks that are migrated or replicated to simplify receiving results.
    std::unordered_map<long long, td_task_t*> td_remote_task_map;

public:
    TD_Scheduling_Manager(int32_t external_device_count);
    ~TD_Scheduling_Manager();

    // adds a tasks to the user defined queue
    void add_task(td_task_t *task, int32_t DeviceID);
    // adds a task migrated to the local process
    void add_remote_task(td_task_t *task, int32_t DeviceID);
    // adds a task replicated from another process
    void add_replicated_task(td_task_t *task, int32_t DeviceID);

    // returns true, iff task is available ready for execution. Implements the priorities
    bool get_task(int32_t PhysicalDeviceID, td_task_t **task);

    // returns true, iff task is available ready for execution. Only covers migratable tasks for the given device affinity
    bool get_migrateable_task(device_affinity affinity, td_task_t **task);

    // notify the completion of a local task
    void notify_task_completion(td_uid_t taskID, bool remote);

    /*
    * Function set_image_base_address
    * Sets base address of particular image index.
    * This is necessary to determine the entry point for functions that represent a target construct
    */
    tdrc set_image_base_address(int idx_image, intptr_t base_address);

    /*
    * Function apply_image_base_address
    * Adds the base address to the address if iBaseAddress == true
    * Else it creates a base address
    */
    intptr_t apply_image_base_address(intptr_t base_address, bool isBaseAddress);

};


#endif //_TARGETDART_SCHEDULING_H
