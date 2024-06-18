#include "../include/scheduling.h"
#include <vector>


Set_Wrapper::Set_Wrapper() {
    internal_set = new std::unordered_set<size_t>();
}
Set_Wrapper::~Set_Wrapper() {
    delete internal_set;
}


void Set_Wrapper::add_task(td_uid_t uid) {
    std::unique_lock<std::mutex> lock(set_mutex);
    internal_set->insert(uid.id);
}

bool Set_Wrapper::task_exists(td_uid_t uid) {
    std::unique_lock<std::mutex> lock(set_mutex);
    if (internal_set->find(uid.id) != internal_set->end()) {
        internal_set->erase(uid.id);
        return true;
    }
    return false;
}

TD_Scheduling_Manager::TD_Scheduling_Manager(int32_t external_device_count) {
    physical_device_count = external_device_count;

    // Create affinity queues: GPUS + CPU + targetDART Scheduling devices {(local, migratable, replica, replicated, remote) * (CPU, GPU, ANY)}
    affinity_queues = new std::vector<TD_Task_Queue>(physical_device_count + 1 + 5 * 3);

    active_tasks = {0};

    priorities= {TD_LOCAL_OFFSET, TD_REPLICATED_OFFSET, TD_REMOTE_OFFSET, TD_MIGRATABLE_OFFSET, TD_REPLICA_OFFSET};

    
    finalized_replicated = new Set_Wrapper();
    started_local_replica = new Set_Wrapper();

    // Initialize the map of remote and replicated tasks
    td_remote_task_map = std::unordered_map<long long, td_task_t*>();
}

TD_Scheduling_Manager::~TD_Scheduling_Manager(){
    delete finalized_replicated;
    delete started_local_replica;
    delete affinity_queues;
}

void TD_Scheduling_Manager::add_task(td_task_t *task, int32_t DeviceID) {
    active_tasks.fetch_add(1);
    affinity_queues->at(DeviceID).addTask(task);
}

void TD_Scheduling_Manager::add_remote_task(td_task_t *task, int32_t DeviceID) {
    active_tasks.fetch_add(1);
    affinity_queues->at(DeviceID + TD_REMOTE_OFFSET).addTask(task);
}

void TD_Scheduling_Manager::add_replicated_task(td_task_t *task, int32_t DeviceID) {
    active_tasks.fetch_add(1);
    affinity_queues->at(DeviceID + TD_REPLICA_OFFSET).addTask(task);
}

bool TD_Scheduling_Manager::get_task(int32_t PhysicalDeviceID, td_task_t **task) {
    // Prio 0: get fixed device tasks first
    *task = affinity_queues->at(PhysicalDeviceID).getTask();
    if (*task != nullptr) {
        return true;
    }

    int affinity_prio[2];
    affinity_prio[1] = TD_ANY_OFFSET;

    if (PhysicalDeviceID < physical_device_count - 1) {
        affinity_prio[0] = TD_OFFLOAD_OFFSET;
    } else {
        affinity_prio[0] = TD_CPU_OFFSET;
    }

    // Prio 1: fitting device type
    // Prio 2: any device affinity
    for (auto device_offset : affinity_prio) {
        for (auto sub_offset : priorities) {
            *task = affinity_queues->at(physical_device_count + device_offset + sub_offset).getTask();
            if (*task != nullptr) {
                if (sub_offset == TD_REPLICA_OFFSET) {            
                    started_local_replica->add_task((*task)->uid);
                } else if (sub_offset == TD_REPLICATED_OFFSET && finalized_replicated->task_exists((*task)->uid)) {
                    // TODO: clean up task
                    active_tasks.fetch_sub(1);
                    continue;
                }
                return true;
            }
        }
    }
    return false;
}

bool TD_Scheduling_Manager::get_migrateable_task(device_affinity affinity, td_task_t **task) {
    *task = affinity_queues->at(physical_device_count + affinity + TD_ANY_OFFSET).getTask();
    if (*task != nullptr) {
        return true;
    }
    return false;
}

void TD_Scheduling_Manager::notify_task_completion(td_uid_t taskID, bool isReplica) {
    active_tasks.fetch_sub(1);

    if (isReplica) {        
        finalized_replicated->add_task(taskID);
    }
}