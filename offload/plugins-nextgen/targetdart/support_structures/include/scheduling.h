#ifndef _TARGETDART_SCHEDULING_H
#define _TARGETDART_SCHEDULING_H

#include "queue.h"
#include "task.h"

#include <__atomic/aliases.h>
#include <cstdint>
#include <unordered_set>
#include <vector>
#include "PluginManager.h"

// Main affinities
#define TD_CPU_OFFSET 2  // Task that should only run on the CPU
#define TD_OFFLOAD_OFFSET 4 // Task that should only run on an offload device
#define TD_ANY_OFFSET TD_CPU_OFFSET + TD_OFFLOAD_OFFSET // Task that can run on any hardware
// Sub affinities/priorities
#define TD_MIGRATABLE_OFFSET 0 // The task may be migrated to another process
#define TD_LOCAL_OFFSET -1 // The task must not be migrated to another process
#define TD_REMOTE_OFFSET 5 // The task was created on another device and exist only a single time accross the runtime
#define TD_REPLICA_OFFSET 6 // The task was created on another device and exists multiple times. Additional communication for cancelation necessary

/* QUEUE Order:
physical_devices + 1 => Available GPUs + host device
=> Addressable scheduling queues by the user (also available as devices)
+ CPU_affinity(Local, Migratable) 
+ Offload_affinity(Local, Migratable) 
+ Any_affinity(Local, Migratable)
=> Internal scheduling queues required for managing remote/replicated data
+ CPU_affinity(Replica, Remote) 
+ Offload_affinity(Replica, Remote) 
+ Any_affinity(Replica, Remote)
*/

/*
Wrapper class managing all tasks on a single process.
*/
class TD_Scheduling_Manager {
private:

    // Number of physical devices managed by other plugins
    int32_t physical_device_count;

    // A set of task queues defining all relevant affinities
    std::vector<TD_Task_Queue> &affinity_queues;

    // Keeps the number of non completed tasks spawned by this process
    std::atomic_int64_t aktive_tasks;

    std::unordered_set<td_uid_t> finalized_replicas;

public:
    TD_Scheduling_Manager(int32_t external_device_count);
    ~TD_Scheduling_Manager();

    void add_task(td_task_t *task, int32_t DeviceID);
    void add_remote_task(td_task_t *task, int32_t DeviceID);
    void add_replica_task(td_task_t *task, int32_t DeviceID);

    td_task_t *get_task(int32_t PhysicalDeviceID);


    void notify_task_completion(td_uid_t taskID);

};


#endif //_TARGETDART_SCHEDULING_H
