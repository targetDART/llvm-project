#include "../include/scheduling.h"
#include "../include/communication.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>


Set_Wrapper::Set_Wrapper() {
}
Set_Wrapper::~Set_Wrapper() {
}

void Set_Wrapper::add_task(td_uid_t uid) {
    std::unique_lock<std::mutex> lock(set_mutex);
    internal_set.insert(uid.id);
}

bool Set_Wrapper::task_exists(td_uid_t uid) {
    std::unique_lock<std::mutex> lock(set_mutex);
    if (internal_set.find(uid.id) != internal_set.end()) {
        internal_set.erase(uid.id);
        return true;
    }
    return false;
}

TD_Scheduling_Manager::TD_Scheduling_Manager(int32_t external_device_count, TD_Communicator *communicator) {
    physical_device_count = external_device_count;

    comm_man = communicator;

    // Create affinity queues: GPUS + CPU + targetDART Scheduling devices {(local, migratable, replica, replicated, remote) * (CPU, GPU, ANY)}
    // two additional padding slots are required to ensure that GPU/ANY replicated, remote queues are accessed correctly.
    affinity_queues = new std::vector<TD_Task_Queue>(physical_device_count + 1 + 5 * 3 + 2);

    active_tasks.store(0);
    local_id_tracker.store(0);

    // Defines the order in which the corresponding queues are accessed during task execution
    priorities = {TD_LOCAL_OFFSET, TD_REPLICATED_OFFSET, TD_REMOTE_OFFSET, TD_MIGRATABLE_OFFSET, TD_REPLICA_OFFSET};

    repartition = false;  
}

TD_Scheduling_Manager::~TD_Scheduling_Manager(){
    delete affinity_queues;
}

td_task_t *TD_Scheduling_Manager::create_task(intptr_t hostptr, KernelArgsTy *KernelArgs, ident_t *Loc) {
    td_task_t *task = (td_task_t*) malloc(sizeof(td_task_t));
    task->host_base_ptr = apply_image_base_address(hostptr, false);
    task->isReplica = false;
    task->KernelArgs = KernelArgs;
    task->Loc = Loc;

    task->uid = {local_id_tracker.fetch_add(1), comm_man->rank};

    return task;
}

KernelArgsTy *copyKernelArgs(KernelArgsTy *KernelArgs) {
    KernelArgsTy *LocalKernelArgs = new KernelArgsTy();
    LocalKernelArgs->Version = KernelArgs-> Version;
    LocalKernelArgs->NumArgs = KernelArgs->NumArgs;
    LocalKernelArgs->ArgBasePtrs = KernelArgs->ArgBasePtrs;
    LocalKernelArgs->ArgPtrs = KernelArgs->ArgPtrs;
    LocalKernelArgs->ArgSizes = KernelArgs->ArgSizes;
    LocalKernelArgs->ArgTypes = KernelArgs->ArgTypes;
    LocalKernelArgs->ArgNames = KernelArgs->ArgNames;
    LocalKernelArgs->ArgMappers = KernelArgs->ArgMappers;
    LocalKernelArgs->Tripcount = KernelArgs->Tripcount;
    LocalKernelArgs->Flags = KernelArgs->Flags;
    LocalKernelArgs->DynCGroupMem = 0;
    LocalKernelArgs->NumTeams[0] = KernelArgs->NumTeams[0];
    LocalKernelArgs->NumTeams[1] = 0;
    LocalKernelArgs->NumTeams[2] = 0;
    LocalKernelArgs->ThreadLimit[0] = KernelArgs->ThreadLimit[0];
    LocalKernelArgs->ThreadLimit[1] = 0;
    LocalKernelArgs->ThreadLimit[2] = 0;
    return LocalKernelArgs;
}

void TD_Scheduling_Manager::add_task(td_task_t *task, int32_t DeviceID) {
    active_tasks.fetch_add(1);
    task->KernelArgs = copyKernelArgs(task->KernelArgs);
    affinity_queues->at(DeviceID).addTask(task);
    DP("added task (%ld%ld) to device %d\n", task->uid.rank, task->uid.id, DeviceID);
}

void TD_Scheduling_Manager::add_remote_task(td_task_t *task, device_affinity DeviceID) {
    active_tasks.fetch_add(1);
    affinity_queues->at(physical_device_count + 1 + DeviceID + TD_REMOTE_OFFSET).addTask(task);
    DP("added remote task (%ld%ld) to device %d\n", task->uid.rank, task->uid.id, physical_device_count + 1 + DeviceID + TD_REMOTE_OFFSET);
}

void TD_Scheduling_Manager::add_replicated_task(td_task_t *task, device_affinity DeviceID) {
    active_tasks.fetch_add(1);
    affinity_queues->at(physical_device_count + 1 + DeviceID + TD_REPLICA_OFFSET).addTask(task);
    DP("added replicated task (%ld%ld) to device %d\n", task->uid.rank, task->uid.id, physical_device_count + 1 + DeviceID + TD_REPLICA_OFFSET);
}

tdrc TD_Scheduling_Manager::get_task(int32_t PhysicalDeviceID, td_task_t **task) {
    // Prio 0: get fixed device tasks first
    *task = affinity_queues->at(PhysicalDeviceID).getTask();
    if (*task != nullptr) {
        return TARGETDART_SUCCESS;
    }

    int affinity_prio[2];
    affinity_prio[1] = TD_ANY_OFFSET;

    if (PhysicalDeviceID < physical_device_count) {
        affinity_prio[0] = TD_OFFLOAD_OFFSET;
    } else {
        affinity_prio[0] = TD_CPU_OFFSET;
    }

    // Prio 1: fitting device type
    // Prio 2: any device affinity
    for (auto device_offset : affinity_prio) {
        for (auto sub_offset : priorities) {
            *task = affinity_queues->at(physical_device_count + 1 + device_offset + sub_offset).getTask();
            if (*task != nullptr) {
                if (sub_offset == TD_REPLICA_OFFSET) {            
                    started_local_replica.add_task((*task)->uid);
                } else if (sub_offset == TD_REPLICATED_OFFSET && finalized_replicated.task_exists((*task)->uid)) {
                    // TODO: clean up task
                    active_tasks.fetch_sub(1);
                    continue;
                }
                DP("Gotten Task for PhyscalID: %d in queue: %d device count %d device offset %d affinity offset %d\n", PhysicalDeviceID, 1+physical_device_count + device_offset + sub_offset, physical_device_count, device_offset, sub_offset);
                return TARGETDART_SUCCESS;
            }
        }
    }
    return TARGETDART_FAILURE;
}

tdrc TD_Scheduling_Manager::get_migrateable_task(device_affinity affinity, td_task_t **task) {
    *task = affinity_queues->at(physical_device_count + 1 + affinity + TD_MIGRATABLE_OFFSET).getTask();
    if (*task != nullptr) {
        return TARGETDART_SUCCESS;
    }
    return TARGETDART_FAILURE;
}

void TD_Scheduling_Manager::notify_task_completion(td_uid_t taskID, bool isReplica) {
    active_tasks.fetch_sub(1);

    if (isReplica) {        
        finalized_replicated.add_task(taskID);
    }
}

bool TD_Scheduling_Manager::is_empty() {
    return active_tasks.load() == 0;
}

bool TD_Scheduling_Manager::do_repartition(){
    return repartition;
}

void TD_Scheduling_Manager::reset_repatition() {
    repartition = false;
}

/**
* Returns 0 iff local_cost = remote_cost
* Returns the desire load to transfer from local to remote, iff local_cost > remote_cost
* Returns the desire load to transfer from remote to local as a negative value, iff local_cost < remote_cost
*/
COST_DATA_TYPE __compute_transfer_load(COST_DATA_TYPE local_cost, COST_DATA_TYPE remote_cost) {
    COST_DATA_TYPE result;
    if (local_cost == remote_cost) {
        result = 0;
    } else if (local_cost > remote_cost) {
        result = SIMPLE_REACTIVITY_LOAD;
    } else {
        result = -SIMPLE_REACTIVITY_LOAD;
    }
    return result;
}

void TD_Scheduling_Manager::iterative_schedule(device_affinity affinity) {
    std::vector<COST_DATA_TYPE> cost_vector = comm_man->global_cost_vector_propagation(affinity_queues->at(physical_device_count + 1 + affinity + TD_MIGRATABLE_OFFSET).getSize());
    std::vector<td_sort_cost_tuple_t> combined_vector(cost_vector.size());
    //DP("iterative schedule: Rank 0: %lu Rank 1: %ld\n", cost_vector.at(0), cost_vector.at(1));

    for (size_t i = 0; i < combined_vector.size(); i++) {
        combined_vector[i].cost = cost_vector[i];
        combined_vector[i].id = i;
    }
    std::sort(combined_vector.begin(), combined_vector.end(), [](td_sort_cost_tuple_t a, td_sort_cost_tuple_t b) 
                                                                                {                                                                                
                                                                                    return a.cost < b.cost;
                                                                                });

    int local_idx = comm_man->size;
    for (size_t i = 0; i < combined_vector.size(); i++) {
        if (combined_vector.at(i).id == comm_man->rank) {
            local_idx = i;
            break;
        }
    }
    
    if (local_idx == comm_man->size) {
        handle_error_en(1, "local rank not found in index search.");
    }

    // implement Chameleon based victim selection
    int partner_idx;
    if (combined_vector.size() % 2 == 0) {
        int half = combined_vector.size() / 2;
        if (local_idx < half) {
            partner_idx = combined_vector.size() + local_idx - half;
        } else {
            partner_idx = local_idx - half;
        }
    } else {
        int half = combined_vector.size() / 2;
        if (local_idx < half) {
            partner_idx = combined_vector.size() + local_idx - half;
        } else {
            partner_idx = local_idx - half - 1;
        }
    }
    partner_idx = combined_vector.size() - local_idx - 1;
    int partner_proc = combined_vector.at(partner_idx).id;

    COST_DATA_TYPE transfer_load = __compute_transfer_load(combined_vector.at(local_idx).cost, combined_vector.at(partner_idx).cost);
    
    //DP("Found Partner proc: %d for load: %ld\n", partner_proc, transfer_load);
    if (transfer_load == 0) {
        return;
    } else if (transfer_load > 0) {
        for (int i = 0; i < SIMPLE_REACTIVITY_LOAD; i++) {
            td_task_t *task;
            //ensure to not send an empty task, iff the queue becomes empty between the vector exchange and migration
            tdrc ret_code = get_migrateable_task(affinity, &task);
            if (ret_code == TARGETDART_SUCCESS) {
                DP("Preparing send task (%ld%ld) to process %d\n", task->uid.rank, task->uid.id, partner_proc);
                comm_man->signal_task_send(partner_proc, true);
                comm_man->send_task(partner_proc, task);
            } else {
                comm_man->signal_task_send(partner_proc, false);
            }
        }
    } else {
        for (int i = 0; i < SIMPLE_REACTIVITY_LOAD; i++) {
            tdrc ret_code = comm_man->receive_signal_task_send(partner_proc);
            if (ret_code == TARGETDART_SUCCESS) {                
                td_task_t *task = new td_task_t;
                comm_man->receive_task(partner_proc, task);
                DP("Received task (%ld%ld) from process %d\n", task->uid.rank, task->uid.id, partner_proc);
                add_remote_task(task, affinity);
            }
        }
    } 
}


int32_t TD_Scheduling_Manager::public_device_count() {
    return affinity_queues->size() - 6;
}

void TD_Scheduling_Manager::synchronize() {
    while (!is_empty()) {
        // sleep for a few micro seconds to limit contention
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

int64_t TD_Scheduling_Manager::get_active_tasks() {
    return active_tasks.load();
}

/// Find the table information in the map or look it up in the translation
/// tables.
TableMap *TD_Scheduling_Manager::getTableMap(void *HostPtr) {
  std::lock_guard<std::mutex> TblMapLock(PM->TblMapMtx);
  HostPtrToTableMapTy::iterator TableMapIt =
      PM->HostPtrToTableMap.find(HostPtr);

  if (TableMapIt != PM->HostPtrToTableMap.end())
    return &TableMapIt->second;

  // We don't have a map. So search all the registered libraries.
  TableMap *TM = nullptr;
  std::lock_guard<std::mutex> TrlTblLock(PM->TrlTblMtx);
  for (HostEntriesBeginToTransTableTy::iterator Itr =
           PM->HostEntriesBeginToTransTable.begin();
       Itr != PM->HostEntriesBeginToTransTable.end(); ++Itr) {
    // get the translation table (which contains all the good info).
    TranslationTable *TransTable = &Itr->second;
    // iterate over all the host table entries to see if we can locate the
    // host_ptr.
    __tgt_offload_entry *Cur = TransTable->HostTable.EntriesBegin;
    for (uint32_t I = 0; Cur < TransTable->HostTable.EntriesEnd; ++Cur, ++I) {
      if (Cur->addr != HostPtr)
        continue;
      // we got a match, now fill the HostPtrToTableMap so that we
      // may avoid this search next time.
      TM = &(PM->HostPtrToTableMap)[HostPtr];
      TM->Table = TransTable;
      TM->Index = I;
      return TM;
    }
  }

  return nullptr;
}

int32_t TD_Scheduling_Manager::total_device_count() {
    return physical_device_count + public_device_count();
}

// executes a task on a given device
tdrc TD_Scheduling_Manager::invoke_task(td_task_t *task, int64_t Device) {

    int64_t effective_device = Device;

    // get physical device
    auto DeviceOrErr = PM->getDevice(effective_device);
    if (!DeviceOrErr)
        FATAL_MESSAGE(effective_device, "%s", toString(DeviceOrErr.takeError()).c_str());
    
    // create new async info
    AsyncInfoTy TargetAsyncInfo(*DeviceOrErr);

    std::vector<void *> devicePtrs(task->KernelArgs->NumArgs);

    DP("Allocating %d arguments for task (%ld%ld)\n", task->KernelArgs->NumArgs, task->uid.rank, task->uid.id);

    // Allocate data on the device and transfer it from host to device if necessary
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs; i++) {
        if (effective_device == total_device_count()) {
            // Avoid data ttransfers for CPU execution
            devicePtrs[i] = task->KernelArgs->ArgPtrs[i];
        } else {
            // non blocking alloc is not non blocking but rather uses the non-blocking calls internally
            devicePtrs[i] = DeviceOrErr->allocData(task->KernelArgs->ArgSizes[i], task->KernelArgs->ArgPtrs[i], TARGET_ALLOC_DEVICE_NON_BLOCKING);
        }
        const bool hasFlagTo = task->KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_TO;
        if (hasFlagTo) {        
            DeviceOrErr->submitData(devicePtrs[i], task->KernelArgs->ArgPtrs[i], task->KernelArgs->ArgSizes[i], TargetAsyncInfo);
        }
    }

    if (checkDeviceAndCtors(effective_device, task->Loc)) {
        DP("Not offloading to device %" PRId64 "\n", effective_device);
        return TARGETDART_FAILURE;
    }

    // generate a Kernel
    llvm::SmallVector<ptrdiff_t> offsets(task->KernelArgs->NumArgs, 0);

    void *HostPtr = (void *) apply_image_base_address(task->host_base_ptr, true);

    TableMap *TM = getTableMap(HostPtr);
    // No map for this host pointer found!
    if (!TM) {
        REPORT("Host ptr " DPxMOD " does not have a matching target pointer.\n",
            DPxPTR(HostPtr));
        return TARGETDART_FAILURE;
    }

    // get target table.
    __tgt_target_table *TargetTable = nullptr;
    {
        std::lock_guard<std::mutex> TrlTblLock(PM->TrlTblMtx);
        assert(TM->Table->TargetsTable.size() > (size_t)effectiveDevice &&
                "Not expecting a device ID outside the table's bounds!");
        TargetTable = TM->Table->TargetsTable[effective_device];
    }
    assert(TargetTable && "Global data has not been mapped\n");

    // Launch device execution.
    void *TgtEntryPtr = TargetTable->EntriesBegin[TM->Index].addr;
    DP("Launching target execution %s with pointer " DPxMOD " (index=%d).\n",
        TargetTable->EntriesBegin[TM->Index].name, DPxPTR(TgtEntryPtr), TM->Index);
        

    DP("Running kernel for task (%ld%ld)\n", task->uid.rank, task->uid.id);
    auto Err = DeviceOrErr->launchKernel(TgtEntryPtr, devicePtrs.data(), offsets.data(), *task->KernelArgs,
                               TargetAsyncInfo, task->Loc, HostPtr);

    if (Err) {
        DP("Kernel launch of task (%ld%ld) failed\n", task->uid.rank, task->uid.id);
    }

    // Deallocate data on the device and transfer it from device to host if necessary
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs - 1; i++) {
        const bool hasFlagFrom = task->KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_FROM;
        if (hasFlagFrom) {        
            DeviceOrErr->retrieveData(task->KernelArgs->ArgPtrs[i], devicePtrs[i], task->KernelArgs->ArgSizes[i], TargetAsyncInfo);
        }
    }

    // Deallocate data on the device and transfer it from device to host if necessary
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs - 1; i++) {
        DeviceOrErr->deleteData(devicePtrs[i], TARGET_ALLOC_DEVICE_NON_BLOCKING);
    }

    // Synchronization on CPU 
    if (effective_device == total_device_count())
        DeviceOrErr->synchronize(TargetAsyncInfo);

    return TARGETDART_SUCCESS;    
}