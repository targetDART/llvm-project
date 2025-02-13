#include "../include/scheduling.h"
#include "../include/communication.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory.h>
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

    memory_manager = TD_Memory_Manager(physical_device_count);
}

TD_Scheduling_Manager::~TD_Scheduling_Manager(){
    delete affinity_queues;
}

td_task_t *TD_Scheduling_Manager::create_task(intptr_t hostptr, KernelArgsTy *KernelArgs, ident_t *Loc, int32_t DeviceID) {
    td_task_t *task = (td_task_t*) malloc(sizeof(td_task_t));
    task->host_base_ptr = apply_image_base_address(hostptr, false);
    task->isReplica = false;
    task->KernelArgs = KernelArgs;
    task->Loc = Loc;
    // fill with sum of Argsizes (KernelArgs)
    task->cached_total_sizes = 0;
    for (uint32_t i = 0; i < KernelArgs->NumArgs; i++) {
        task->cached_total_sizes += (COST_DATA_TYPE) KernelArgs->ArgSizes[i];
    }

    task->affinity = extract_device_affinity(DeviceID);    

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


device_affinity TD_Scheduling_Manager::extract_device_affinity(int DeviceID) {
    int internalID = DeviceID;
    internalID -= physical_device_count;
    if (internalID < 0)
        return GPU;
    internalID -= 1 + 3; //number of addressable queues per device affinity. The first value after the initial device (CPU) is CPU.
    if  (internalID < 0)
        return CPU;
    internalID -= 3; //number of addressable queues per device affinity
    if (internalID < 0)
        return GPU;
    // if it is not an explicit or managed CPU/GPU only the ANY affinity remains.
    return ANY;
}

void TD_Scheduling_Manager::add_task(td_task_t *task, int32_t DeviceID) {
    active_tasks++;
    task->KernelArgs = copyKernelArgs(task->KernelArgs);
    affinity_queues->at(DeviceID).addTask(task);
    DP("added task (%ld%ld) to device %d\n", task->uid.rank, task->uid.id, DeviceID);
}

void TD_Scheduling_Manager::add_remote_task(td_task_t *task, device_affinity DeviceID) {
    active_tasks++;
    affinity_queues->at(physical_device_count + 1 + DeviceID + TD_REMOTE_OFFSET).addTask(task);
    DP("added remote task (%ld%ld) to device %d\n", task->uid.rank, task->uid.id, physical_device_count + 1 + DeviceID + TD_REMOTE_OFFSET);
}

void TD_Scheduling_Manager::add_replicated_task(td_task_t *task, device_affinity DeviceID) {
    active_tasks++;
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
                    active_tasks--;
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
    active_tasks--;
    DP("completed task (%ld%ld)\n", taskID.rank, taskID.id);
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
    COST_DATA_TYPE result = local_cost - remote_cost;
    /*if (local_cost == remote_cost) {
        result = 0;
    } else if (local_cost > remote_cost) {
        result = SIMPLE_REACTIVITY_LOAD;
    } else {
        result = -SIMPLE_REACTIVITY_LOAD;
    }*/
    return result;
}

void TD_Scheduling_Manager::iterative_schedule(device_affinity affinity) {
    TRACE_START("fine_schedule\n");
    COST_DATA_TYPE local_cost = affinity_queues->at(physical_device_count + 1 + affinity + TD_MIGRATABLE_OFFSET).getSize() + 
                                affinity_queues->at(physical_device_count + 1 + affinity + TD_LOCAL_OFFSET).getSize() + 
                                affinity_queues->at(physical_device_count + 1 + affinity + TD_REMOTE_OFFSET).getSize() +
                                affinity_queues->at(physical_device_count + 1 + affinity + TD_REPLICA_OFFSET).getSize() +
                                affinity_queues->at(physical_device_count + 1 + affinity + TD_REPLICATED_OFFSET).getSize();
    std::vector<COST_DATA_TYPE> cost_vector = comm_man->global_cost_vector_propagation(local_cost);
    std::vector<td_sort_cost_tuple_t> combined_vector(cost_vector.size());
    //DP("iterative schedule: Rank 0: %lu Rank 1: %ld\n", cost_vector.at(0), cost_vector.at(1));

    for (size_t i = 0; i < combined_vector.size(); i++) {
        combined_vector[i].cost = cost_vector[i];
        combined_vector[i].id = i;
    }
    std::stable_sort(combined_vector.begin(), combined_vector.end(), [](td_sort_cost_tuple_t a, td_sort_cost_tuple_t b) 
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
    partner_idx = combined_vector.size() - local_idx - 1;
    int partner_proc = combined_vector.at(partner_idx).id;

    COST_DATA_TYPE transfer_load = __compute_transfer_load(combined_vector.at(local_idx).cost, combined_vector.at(partner_idx).cost);
    
    //DP("Found Partner proc: %d for load: %ld\n", partner_proc, transfer_load);
    if (transfer_load == 0) {
        return;
    } else if (transfer_load > 3) {
        for (int i = 0; i < SIMPLE_REACTIVITY_LOAD; i++) {
            td_task_t *task;
            //ensure to not send an empty task, iff the queue becomes empty between the vector exchange and migration
            tdrc ret_code = get_migrateable_task(affinity, &task);
            if (ret_code == TARGETDART_SUCCESS) {
                DP("Preparing send task (%ld%ld) to process %d\n", task->uid.rank, task->uid.id, partner_proc);
                //comm_man->signal_task_send(partner_proc, true);
                comm_man->send_task(partner_proc, task);
            } else {
                //comm_man->signal_task_send(partner_proc, false);
            }
        }
    } else {
        /*for (int i = 0; i < SIMPLE_REACTIVITY_LOAD; i++) {
            tdrc ret_code = comm_man->receive_signal_task_send(partner_proc);
            if (ret_code == TARGETDART_SUCCESS) {                
                td_task_t *task = new td_task_t;
                comm_man->receive_task(partner_proc, task);
                DP("Received task (%ld%ld) from process %d\n", task->uid.rank, task->uid.id, partner_proc);
                add_remote_task(task, affinity);
            }
        }*/
    } 
    TRACE_END("fine_schedule\n");
}


/**
* Implements the rescheduling of tasks for the local MPI process and its current victim, defined by offset.
* target_load: defines the load the victim should have in total after migration.
* affinity: defines which kinds of tasks should be considered for a rescheduling.
*/
void TD_Scheduling_Manager::partial_global_reschedule(COST_DATA_TYPE target_load, device_affinity affinity, int offset) {
    std::vector<td_task_t*> transferred_tasks;
    COST_DATA_TYPE totalcost = 0;
    while (totalcost < BALANCE_FACTOR * target_load) {
        td_task_t *next_task;
        tdrc return_code = get_migrateable_task(affinity, &next_task);
        if (return_code == TARGETDART_FAILURE) {
            break;
        }
        if (next_task->cached_total_sizes >= target_load/BALANCE_FACTOR) {
            break;
        } else {
            transferred_tasks.push_back(next_task);
        }
    }
    
    //TODO: think about MPI_pack as well
    for (size_t t = 0; t < transferred_tasks.size(); t++) {
        comm_man->send_task(comm_man->rank + offset, transferred_tasks.at(t));
    }
}

void TD_Scheduling_Manager::global_reschedule(device_affinity affinity) {
    TRACE_START("coarse_schedule\n");
    global_sched_params_t params = comm_man->global_cost_communicator(affinity_queues->at(physical_device_count + 1 + affinity + TD_MIGRATABLE_OFFSET).getCost());
    COST_DATA_TYPE target_load = params.total_cost / comm_man->size;

    if (target_load == 0) {
        DP("Skip global reschedule with target load %ld\n", target_load);
        TRACE_END("coarse_schedule\n");
        return;
    }

    DP("Do global reschedule with local load %ld and target load %ld\n", affinity_queues->at(physical_device_count + 1 + affinity + TD_MIGRATABLE_OFFSET).getCost(), target_load);

    COST_DATA_TYPE pre_transfer = 0;
    COST_DATA_TYPE post_transfer = 0;

    //compute pre_transfer
    if (comm_man->rank != 0) {
        COST_DATA_TYPE predecessor_load = params.prefix_sum / comm_man->rank;
        pre_transfer = (target_load - predecessor_load) * comm_man->rank;
    }

    DP("Send a load of %ld to predecessors\n", pre_transfer);

    //compute post_transfer
    if (comm_man->rank != comm_man->size - 1) {
        COST_DATA_TYPE successor_cost = params.total_cost - params.local_cost - params.prefix_sum;
        int num_successors = comm_man->size - 1 - comm_man->rank; //inverted rank
        COST_DATA_TYPE successor_load = successor_cost/num_successors;
        post_transfer = (target_load - successor_load) * num_successors;
    }

    DP("Send a load of %ld to successors\n", post_transfer);

    //calculate num tasks per direktion
    if (pre_transfer < 0) {
        pre_transfer = 0;
    }
    if (post_transfer < 0) {
        post_transfer = 0;
    }

    //compute furthest data transfer
    int pre_distance = pre_transfer/target_load + 1;
    int post_distance = post_transfer/target_load + 1;

    //general case transfers predecessor
    for (int i = 1; i < pre_distance; i++) {
        partial_global_reschedule(target_load, affinity, -i);
    }
    //general case transfers successor
    for (int i = 1; i < post_distance; i++) {
        partial_global_reschedule(target_load, affinity, i);
    }
    
    COST_DATA_TYPE pre_remainder_load = pre_transfer % target_load;    
    partial_global_reschedule(pre_remainder_load, affinity, -pre_distance);
    COST_DATA_TYPE post_remainder_load = post_transfer % target_load;    
    partial_global_reschedule(post_remainder_load, affinity, post_distance);
    TRACE_END("coarse_schedule\n");
}

int32_t TD_Scheduling_Manager::public_device_count() {
    return affinity_queues->size() - 6;
}

void TD_Scheduling_Manager::synchronize() {
    TRACE_START("synchronize\n");
    while (!is_empty()) {
        // sleep for a few micro seconds to limit contention
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    TRACE_END("synchronize\n");
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

    TRACE_START("invoke_task (%ld%ld)\n", task->uid.rank, task->uid.id);

    int64_t effective_device = Device;

    // Work on copy of KernelArgs to avoid differences between MPI processes
    KernelArgsTy *BaseArgs = task->KernelArgs;
    task->KernelArgs = copyKernelArgs(BaseArgs);

    // get physical device
    auto DeviceOrErr = PM->getDevice(effective_device);
    if (!DeviceOrErr)
        FATAL_MESSAGE(effective_device, "%s", toString(DeviceOrErr.takeError()).c_str());

    void *HostPtr = (void *) apply_image_base_address(task->host_base_ptr, true);

    TableMap *TM = getTableMap(HostPtr);
    // No map for this host pointer found!
    if (!TM) {
        REPORT("Host ptr " DPxMOD " does not have a matching target pointer.\n",
            DPxPTR(HostPtr));
        return TARGETDART_FAILURE;
    }
    
    // create new async info
    AsyncInfoTy TargetAsyncInfo(*DeviceOrErr);

    std::vector<void *> devicePtrs(task->KernelArgs->NumArgs);

    DP("Allocating %d arguments for task (%ld%ld)\n", task->KernelArgs->NumArgs, task->uid.rank, task->uid.id);

    // note: the negation here is not that nice, I know...
    const auto noAllocation = [&](auto i) {
        const bool IsLiteral = (task->KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_LITERAL) != 0;
        // copy literals directly; same for everything non-sized (that will fail non-locally anyways), or CPU execution
        return effective_device == total_device_count() || IsLiteral || task->KernelArgs->ArgSizes[i] == 0;
    };

    TRACE_START("H2D_transfer_task (%ld%ld)\n", task->uid.rank, task->uid.id);
    // Allocate data on the device and transfer it from host to device if necessary
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs; i++) {
        if (noAllocation(i)) {
            if (task->KernelArgs->ArgSizes[i] == 0) {
                // Avoid data transfers for pre transfered data
                devicePtrs[i] = memory_manager.get_data_mapping(effective_device, task->KernelArgs->ArgPtrs[i]);
            } 
            if (devicePtrs[i] == nullptr) {                
                // Avoid data transfers for CPU execution
                devicePtrs[i] = task->KernelArgs->ArgPtrs[i];
            }
        } else {
            // allocate data.
            // non blocking alloc is not non blocking but rather uses the non-blocking calls internally
            // (only allocate, if there's something to allocate)
            devicePtrs[i] = DeviceOrErr->allocData(task->KernelArgs->ArgSizes[i], task->KernelArgs->ArgPtrs[i], TARGET_ALLOC_DEVICE_NON_BLOCKING);
        }

        // copied from the default kernel exec
        DP("(%ld%ld) Entry %2d: Host=" DPxMOD ", Device=" DPxMOD ", Size=%" PRId64 ", Type=0x%" PRIx64 "\n",
          task->uid.rank, task->uid.id, i, DPxPTR(task->KernelArgs->ArgPtrs[i]), DPxPTR(devicePtrs[i]), task->KernelArgs->ArgSizes[i], task->KernelArgs->ArgTypes[i]);

        const bool hasFlagTo = task->KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_TO;
        if (hasFlagTo && task->KernelArgs->ArgSizes[i] > 0) {
            DP("(%ld%ld) Entry %2d: H2D copy\n", task->uid.rank, task->uid.id, i);
            DeviceOrErr->submitData(devicePtrs[i], task->KernelArgs->ArgPtrs[i], task->KernelArgs->ArgSizes[i], TargetAsyncInfo);
        }
    }
    TRACE_END("H2D_transfer_task (%ld%ld)\n", task->uid.rank, task->uid.id);

    if (checkDeviceAndCtors(effective_device, task->Loc)) {
        DP("Not offloading to device %" PRId64 "\n", effective_device);
        TRACE_END("invoke_task (%ld%ld)\n", task->uid.rank, task->uid.id);
        return TARGETDART_FAILURE;
    }

    // generate a Kernel
    llvm::SmallVector<ptrdiff_t> offsets(task->KernelArgs->NumArgs, 0);

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
    TRACE_START("kernel_launch_task (%ld%ld)\n", task->uid.rank, task->uid.id);
    auto Err = DeviceOrErr->launchKernel(TgtEntryPtr, devicePtrs.data(), offsets.data(), *task->KernelArgs,
                               TargetAsyncInfo, task->Loc, HostPtr);
    TRACE_END("kernel_launch_task (%ld%ld)\n", task->uid.rank, task->uid.id);

    if (Err) {
        DP("Kernel launch of task (%ld%ld) failed\n", task->uid.rank, task->uid.id);
    }

    TRACE_START("D2H_transfer_task (%ld%ld)\n", task->uid.rank, task->uid.id);
    // Deallocate data on the device and transfer it from device to host if necessary
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs - 1; i++) {
        const bool hasFlagFrom = task->KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_FROM;
        if (hasFlagFrom && task->KernelArgs->ArgSizes[i] > 0) {
            DP("(%ld%ld) Entry %2d: D2H copy\n", task->uid.rank, task->uid.id, i);
            DeviceOrErr->retrieveData(task->KernelArgs->ArgPtrs[i], devicePtrs[i], task->KernelArgs->ArgSizes[i], TargetAsyncInfo);
        }
    }

    // Deallocate data on the device and transfer it from device to host if necessary
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs - 1; i++) {
        if (!noAllocation(i)) {
            DP("(%ld%ld) Entry %2d: data deletion\n", task->uid.rank, task->uid.id, i);
            DeviceOrErr->deleteData(devicePtrs[i], TARGET_ALLOC_DEVICE_NON_BLOCKING);
        }
    }
    TRACE_END("D2H_transfer_task (%ld%ld)\n", task->uid.rank, task->uid.id);

    // Synchronization on CPU 
    if (effective_device != total_device_count()) {
        DeviceOrErr->synchronize(TargetAsyncInfo);
    }

    // Restore original KernelArgs 
    delete task->KernelArgs;
    task->KernelArgs = BaseArgs;

    TRACE_END("invoke_task (%ld%ld)\n", task->uid.rank, task->uid.id);

    return TARGETDART_SUCCESS;    
}

TD_Memory_Manager &TD_Scheduling_Manager::getMemoryManager() {
    return memory_manager;
}
