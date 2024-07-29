#ifndef _TARGETDART_SCHEDULING_H
#define _TARGETDART_SCHEDULING_H

#include "queue.h"
#include "task.h"
#include "communication.h"

#include "PluginManager.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_set>
#include <vector>

#define SIMPLE_REACTIVITY_LOAD 1

typedef struct td_sort_cost_tuple_t{
    COST_DATA_TYPE        cost;
    int                   id;
} td_sort_cost_tuple_t;

class Set_Wrapper {

private:
    std::unordered_set<size_t> internal_set;
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

    // Stores the next id for any created task.
    std::atomic<int64_t> local_id_tracker;

    // The uids of replicated tasks to ensure they won`t be executed twice
    // tasks in this set are defined on a remote process
    Set_Wrapper finalized_replicated;

    // The uids of replica tasks started on their home machine t ensure they won`t be executed twice
    // tasks in this set are defined on the local process
    Set_Wrapper started_local_replica;

    // defines the priorities of affinities
    std::vector<int32_t> priorities;

    // States if the repartitioning should be triggered
    bool repartition;

    /// Find the table information in the map or look it up in the translation
    /// tables.
    TableMap *getTableMap(void *HostPtr);

    // Reference to the communication manager
    TD_Communicator *comm_man;

public:
    TD_Scheduling_Manager(int32_t external_device_count, TD_Communicator *communicator);
    ~TD_Scheduling_Manager();

    // creates a new targetDART task
    td_task_t *create_task(intptr_t hostptr, KernelArgsTy *KernelArgs, ident_t *Loc);

    // adds a tasks to the user defined queue
    void add_task(td_task_t *task, int32_t DeviceID);
    // adds a task migrated to the local process
    void add_remote_task(td_task_t *task, device_affinity DeviceType);
    // adds a task replicated from another process
    void add_replicated_task(td_task_t *task, device_affinity DeviceType);

    // returns true, iff task is available ready for execution. Implements the priorities
    tdrc get_task(int32_t PhysicalDeviceID, td_task_t **task);

    // returns true, iff task is available ready for execution. Only covers migratable tasks for the given device affinity
    tdrc get_migrateable_task(device_affinity affinity, td_task_t **task);

    // notify the completion of a local task
    void notify_task_completion(td_uid_t taskID, bool isReplica);

    // returns true, iff no tasks are remaining in any queue
    bool is_empty();    

    // test if repartitioning is required 
    bool do_repartition();

    // reset the repartitioning state
    void reset_repatition();

    // implements an iterative scheduling algorithm 
    void iterative_schedule(device_affinity affinity);

    // returns the number of user visible devices
    int32_t public_device_count();

    // synchronize the targetdart plugin with the current thread
    void synchronize();

    // Returns the number of active tasks
    int64_t get_active_tasks();

    // executes a task on a given device;
    tdrc invoke_task(td_task_t *task, int64_t Device);

};


#endif //_TARGETDART_SCHEDULING_H
