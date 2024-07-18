#include "../include/communication.h"
#include "../include/task.h"

#include "Shared/Debug.h"
#include "mpi.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>


void TD_Communicator::transfer_setup() {
  MPI_Comm_rank(targetdart_comm, &rank);
  MPI_Comm_size(targetdart_comm, &size);

  declare_KernelArgs_type();
  declare_uid_type();
  declare_task_type();
}

TD_Communicator::TD_Communicator(){

  int rc, flag, valuelen;
  const char pset_name[] = "mpi://WORLD";
  const char mt_key[] = "thread_level";
  const char mt_value[] = "MPI_THREAD_MULTIPLE";
  char out_value[100]; /* large enough */
  MPI_Group wgroup = MPI_GROUP_NULL;
  MPI_Info sinfo = MPI_INFO_NULL;
  MPI_Info tinfo = MPI_INFO_NULL;

  // Create initial MPI Session
  MPI_Info_create(&sinfo);
  MPI_Info_set(sinfo, mt_key, mt_value);
  rc = MPI_Session_init(sinfo, MPI_ERRORS_RETURN, &td_libhandle);
  if (rc != MPI_SUCCESS) {
    DP("Failed setting session info\n");
    ret = -1;
    goto fn_exit;
  }

  /*
   * check we got thread support level foo library needs
   */
  rc = MPI_Session_get_info(td_libhandle, &tinfo);
  if (rc != MPI_SUCCESS) {
    DP("Failed getting session info\n");
    ret = -1;
    goto fn_exit;
  }

  valuelen = sizeof(out_value);
  MPI_Info_get(tinfo, mt_key,  valuelen, out_value, &flag);
  if (0 == flag) {
    DP("Could not find key %s\n", mt_key);
    ret = -1;
    goto fn_exit;
  }

  if (strcmp(out_value, mt_value)) {
    DP("Did not get thread multiple support, got %s\n", out_value);
    ret = -1;
    goto fn_exit;
  }

  /*
   * create a group from the WORLD process set
   */
  rc = MPI_Group_from_session_pset(td_libhandle, pset_name, &wgroup);
  if (rc != MPI_SUCCESS) {
    DP("Failed getting lib group\n");
    ret = -1;
    goto fn_exit;
  }

  /*
   * get a communicator
   */
  rc = MPI_Comm_create_from_group(wgroup,
                                  "org.mpi-forum.mpi-v4_0.example-ex11_8",
                                  MPI_INFO_NULL, MPI_ERRORS_RETURN, &targetdart_comm);
  if (rc != MPI_SUCCESS) {
    DP("Failed getting lib comm\n");
    ret = -1;
    goto fn_exit;
  }
  
  
  /*
   * free group, library doesn't need it.
   */

  transfer_setup();

  if (ret != -1)
    return;
fn_exit:
  DP("Closing lib\n");
  MPI_Group_free(&wgroup);

  if (sinfo != MPI_INFO_NULL) {
    MPI_Info_free(&sinfo);
  }

  if (tinfo != MPI_INFO_NULL) {
    MPI_Info_free(&tinfo);
  }

  if (ret != 0) {
    MPI_Session_finalize(&td_libhandle);
  }

  return;
}

TD_Communicator::~TD_Communicator(){
    DP("lib internal ret = %d\n", ret);
    if (ret == 0) {
        DP("Closing MPI lib\n");
        MPI_Session_finalize(&td_libhandle);
        DP("Closed MPI lib\n");
    }
}

tdrc TD_Communicator::declare_KernelArgs_type() {
    const int nitems = 3;
    int blocklengths[3] = {2,2,7};
    MPI_Datatype types[3] = {MPI_UINT32_T, MPI_UINT64_T,MPI_UINT32_T};
    MPI_Aint offsets[3];
    offsets[0] = (MPI_Aint) offsetof(KernelArgsTy, Version); // also NumArgs
    offsets[1] = (MPI_Aint) offsetof(KernelArgsTy, Tripcount); //also flags
    offsets[2] = (MPI_Aint) offsetof(KernelArgsTy, NumTeams); //also ThreadLimit and DynCGroupMem

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &TD_Kernel_Args);
    MPI_Type_commit(&TD_Kernel_Args);
    return TARGETDART_SUCCESS;
}

tdrc TD_Communicator::declare_task_type() {
    const int nitems = 2;
    int blocklengths[2] = {1,1};
    MPI_Datatype types[2] = {MPI_LONG, TD_TASK_UID};
    MPI_Aint offsets[2];
    offsets[0] = (MPI_Aint) offsetof(td_task_t, host_base_ptr);
    offsets[1] = (MPI_Aint) offsetof(td_task_t, uid);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &TD_MPI_Task);
    MPI_Type_commit(&TD_MPI_Task);

    return TARGETDART_SUCCESS;
}

tdrc TD_Communicator::declare_uid_type() {
    const int nitems = 1;
    int blocklengths[1] = {2};
    MPI_Datatype types[1] = {MPI_INT64_T};
    MPI_Aint offsets[1];
    offsets[0] = (MPI_Aint) offsetof(td_uid_t, id);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &TD_TASK_UID);
    MPI_Type_commit(&TD_TASK_UID);

    return TARGETDART_SUCCESS;
}

tdrc TD_Communicator::send_task(int dest, td_task_t *task) {

    //TODO: Use MPI pack to summarize the messages into a single Send
    //TODO: Use non-blocking send
    DP("Send task (%ld%ld) to process %d\n", task->uid.rank, task->uid.id, dest);

    remote_task_map.insert({task->uid, task});

    //Send Task Data
    MPI_Send(task, 1, TD_MPI_Task, dest, SEND_TASK, targetdart_comm);
    DP("Send task structure for task (%ld%ld) to process %d\n", task->uid.rank, task->uid.id, dest);
    //Send static KernelArgs values excluding pointervalues
    MPI_Send(task->KernelArgs, 1, TD_Kernel_Args, dest, SEND_KERNEL_ARGS, targetdart_comm);
    DP("Send KernelArgs for task (%ld%ld) to process %d\n", task->uid.rank, task->uid.id, dest);
    //Send Argument sizes for actual data transfers
    MPI_Send(task->KernelArgs->ArgSizes, task->KernelArgs->NumArgs, MPI_INT64_T, dest, SEND_PARAM_SIZES, targetdart_comm);
    DP("Send ArgSizes for task (%ld%ld) to process %d\n", task->uid.rank, task->uid.id, dest);
    //Send Argument types for each kernel
    MPI_Send(task->KernelArgs->ArgTypes, task->KernelArgs->NumArgs, MPI_INT64_T, dest, SEND_PARAM_TYPES, targetdart_comm);
    DP("Send ArgTypes for task (%ld%ld) to process %d\n", task->uid.rank, task->uid.id, dest);

    //Send the Base Pointer offsets for all arguments
    std::vector<int64_t> diff(task->KernelArgs->NumArgs);
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs; i++) {
        diff[i] = ((int64_t) task->KernelArgs->ArgBasePtrs[i]) - ((int64_t) task->KernelArgs->ArgPtrs[i]);
    }
    MPI_Send(diff.data(), task->KernelArgs->NumArgs, MPI_INT64_T, dest, SEND_BASE_PTRS, targetdart_comm);

    //Send all parameter values
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs; i++) {
        //Test if data needs to be transfered to the kernel. Defined in omptarget.h (tgt_map_type).
        int64_t IsMapTo = task->KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_TO;
        if (IsMapTo != 0) {
            MPI_Send(task->KernelArgs->ArgPtrs[i], task->KernelArgs->ArgSizes[i], MPI_BYTE, dest, SEND_PARAMS, targetdart_comm);
            DP("Send mem for task (%ld%ld) at " DPxMOD " to process %d\n", task->uid.rank, task->uid.id, DPxPTR(task->KernelArgs->ArgPtrs[i]), dest);
        }
    }

    // Send source location to support debuggin information
    MPI_Send(task->Loc, 4, MPI_INT32_T, dest, SEND_SOURCE_LOCS, targetdart_comm);
    DP("Send Loc for task (%ld%ld) to process %d\n", task->uid.rank, task->uid.id, dest);

    //Send base location
    int64_t Locptr = (int64_t) apply_image_base_address((intptr_t) task->Loc->psource, false);
    MPI_Send(&Locptr, 1, MPI_INT64_T, dest, SEND_LOCS_PSOURCE, targetdart_comm);
    DP("Send Locptr for task (%ld%ld) to process %d\n", task->uid.rank, task->uid.id, dest);

    //Base Pointers == pointers can be assumed for simple cases.
    //For complex combinations of pointers and scalars OMP breaks without our interference

    DP("Send task (%ld%ld) to process %d finished\n", task->uid.rank, task->uid.id, dest);

    return TARGETDART_SUCCESS;
}

tdrc TD_Communicator::receive_task(int source, td_task_t *task) {

    //TODO: use MPI probe for complete receives

    DP("Receive new task from process %d\n", source);

    //Receive Task Data
    MPI_Recv(task, 1, TD_MPI_Task, source , SEND_TASK, targetdart_comm, MPI_STATUS_IGNORE);
    DP("Recv task structure for task (%ld%ld) from process %d\n", task->uid.rank, task->uid.id, source);

    //Receive static KernelArgs values excluding pointervalues
    task->KernelArgs = new KernelArgsTy;
    MPI_Recv(task->KernelArgs, 1, TD_Kernel_Args, source, SEND_KERNEL_ARGS, targetdart_comm, MPI_STATUS_IGNORE);
    DP("Recv KernelArgs for task (%ld%ld) from process %d\n", task->uid.rank, task->uid.id, source);
    //Receive Argument sizes for actual data transfers
    task->KernelArgs->ArgSizes = new int64_t[task->KernelArgs->NumArgs];
    MPI_Recv(task->KernelArgs->ArgSizes, task->KernelArgs->NumArgs, MPI_INT64_T, source, SEND_PARAM_SIZES, targetdart_comm, MPI_STATUS_IGNORE);
    DP("Recv ArgSizes for task (%ld%ld) from process %d\n", task->uid.rank, task->uid.id, source);
    //Receive Argument types for each kernel
    task->KernelArgs->ArgTypes = new int64_t[task->KernelArgs->NumArgs];
    MPI_Recv(task->KernelArgs->ArgTypes, task->KernelArgs->NumArgs, MPI_INT64_T, source, SEND_PARAM_TYPES, targetdart_comm, MPI_STATUS_IGNORE);
    DP("Recv ArgTypes for task (%ld%ld) from process %d\n", task->uid.rank, task->uid.id, source);
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs; i++) {    
        DP("Argument type of arg %d: " DPxMOD "\n", i, DPxPTR(task->KernelArgs->ArgTypes[i]));
        assert(!(OMP_TGT_MAPTYPE_LITERAL & task->KernelArgs->ArgTypes[i]) && "Parameters should not be mapped implicitly as literals to avoid remote GPU errors. Use map clause on literals as well. (map(to:var))\n");
    }

    //Declare Mappers and Names
    task->KernelArgs->ArgMappers = new void*[task->KernelArgs->NumArgs];
    task->KernelArgs->ArgNames = new void*[task->KernelArgs->NumArgs];

    //Receive the Base Pointer offsets for all arguments
    std::vector<int64_t> diff(task->KernelArgs->NumArgs);
    MPI_Recv(diff.data(), task->KernelArgs->NumArgs, MPI_INT64_T, source, SEND_BASE_PTRS, targetdart_comm,MPI_STATUS_IGNORE);
    DP("Recv pointer diff for task (%ld%ld) from process %d\n", task->uid.rank, task->uid.id, source);
    //Receive all parameter values
    task->KernelArgs->ArgPtrs = new void*[task->KernelArgs->NumArgs];
    task->KernelArgs->ArgBasePtrs = new void*[task->KernelArgs->NumArgs];
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs; i++) {
        //Test if data needs to be transfered to the kernel. Defined in omptarget.h (tgt_map_type).
        //TODO: look into datatype and allocation
        task->KernelArgs->ArgBasePtrs[i] = (void *) new int64_t[task->KernelArgs->ArgSizes[i] + diff[i]];
        task->KernelArgs->ArgPtrs[i] = (void *) (((int64_t) task->KernelArgs->ArgBasePtrs[i]) + diff[i]);

        
        DP("Allocated memory for task (%ld%ld) at" DPxMOD " with size %ld bytes\n", task->uid.rank, task->uid.id, DPxPTR(task->KernelArgs->ArgPtrs[i]), task->KernelArgs->ArgSizes[i]);  
        int64_t IsMapTo = task->KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_TO;
        if (IsMapTo != 0) {
            MPI_Recv(task->KernelArgs->ArgPtrs[i], task->KernelArgs->ArgSizes[i], MPI_BYTE, source, SEND_PARAMS, targetdart_comm, MPI_STATUS_IGNORE);
            DP("Recv mem for task (%ld%ld) at " DPxMOD " from process %d\n", task->uid.rank, task->uid.id, DPxPTR(task->KernelArgs->ArgPtrs[i]), source);
        }

        //Fill Mappers and Names with null
        task->KernelArgs->ArgMappers[i] = NULL;
        task->KernelArgs->ArgNames[i] = NULL;
    }



    // Receive source location to support debuggin information
    task->Loc = new ident_t;
    MPI_Recv(task->Loc, 4, MPI_INT32_T, source, SEND_SOURCE_LOCS, targetdart_comm, MPI_STATUS_IGNORE);
    
    //Receive base location
    int64_t Locptr;
    MPI_Recv(&Locptr, 1, MPI_INT64_T, source, SEND_LOCS_PSOURCE, targetdart_comm, MPI_STATUS_IGNORE);
    task->Loc->psource = (const char*) apply_image_base_address((intptr_t) Locptr, true);

    //Base Pointers == pointers can be assumed for simple cases.
    //For complex combinations of pointers and scalars OMP breaks without our interference
    task->KernelArgs->ArgBasePtrs = task->KernelArgs->ArgPtrs;


    DP("Received task (%ld%ld) from process %d, finished\n", task->uid.rank, task->uid.id, source);


    return TARGETDART_SUCCESS;
}

tdrc TD_Communicator::test_and_receive_tasks(td_task_t *task) {

    //test, if a task result can be received
    MPI_Status status;
    int flag;

    MPI_Iprobe(MPI_ANY_SOURCE, SEND_TASK, targetdart_comm, &flag, &status);
    if (flag == true) {
        DP("Task receive signaled\n");
        receive_task(status.MPI_SOURCE, task);
        return TARGETDART_SUCCESS;
    }
    return TARGETDART_FAILURE;
}

tdrc TD_Communicator::signal_task_send(int target, bool value) {
    int flag = value;
    MPI_Send(&flag, 1, MPI_INT, target, SIGNAL_TASK_SEND, targetdart_comm);
    DP("Signal task send to process %d. Will send task %d\n", target, value);
    return TARGETDART_SUCCESS;
}

tdrc TD_Communicator::receive_signal_task_send(int source) {
    int flag;
    MPI_Recv(&flag, 1, MPI_INT, source, SIGNAL_TASK_SEND, targetdart_comm, MPI_STATUS_IGNORE);
    if (flag) {
        DP("Task send signaled by process %d\n", source);
        return TARGETDART_SUCCESS;
    }
    DP("Task send declined by process %d\n", source);
    return TARGETDART_FAILURE;
}

tdrc TD_Communicator::send_task_result(td_task_t *task) {

    //TODO: Use MPI pack to summarize the messages into a single Send
    //TODO: Use non-blocking send
    DP("Start result transfer of task (%ld%ld)\n", task->uid.rank, task->uid.id);
    //Send Task uid
    MPI_Send(&task->uid, 1, TD_TASK_UID, task->uid.rank, SEND_RESULT_UID, targetdart_comm);
    DP("Sending result for task (%ld%ld) to process %ld\n", task->uid.rank, task->uid.id, task->uid.rank);

    //Send Task returncode
    MPI_Send(&task->return_code, 1, MPI_INT, task->uid.rank, SEND_RESULT_RETURN_CODE, targetdart_comm);
    DP("Sending Return Code: %d for task (%ld%ld) in remote task map\n", task->return_code, task->uid.rank, task->uid.id);

    //Send all parameter values
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs; i++) {
        //Test if data needs to be transfered to the kernel. Defined in omptarget.h (tgt_map_type).
        int64_t IsMapFrom = task->KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_FROM;
        if (IsMapFrom != 0) {
            MPI_Send(task->KernelArgs->ArgPtrs[i], task->KernelArgs->ArgSizes[i], MPI_BYTE, task->uid.rank, SEND_RESULT_DATA, targetdart_comm);
            DP("Sending result for task (%ld%ld) at " DPxMOD " \n", task->uid.rank, task->uid.id, DPxPTR(task->KernelArgs->ArgPtrs[i]));
        }
    }

    //Base Pointers == pointers can be assumed for simple cases.
    //For complex combinations of pointers and scalars OMP breaks without our interference
    //TODO: free task data

    DP("Result transfer of task (%ld%ld) finished\n", task->uid.rank, task->uid.id);
    return TARGETDART_SUCCESS;
}

tdrc TD_Communicator::receive_task_result(int source, td_uid_t *uid) {

    DP("Start result receival\n");
    //TODO: use MPI probe for complete receives
    //Receive Task Data
    MPI_Recv(uid, 1, TD_TASK_UID, source, SEND_RESULT_UID, targetdart_comm, MPI_STATUS_IGNORE);
    DP("Receiving result for task (%ld%ld) from process %d\n", uid->rank, uid->id, source);

    td_task_t *task = remote_task_map[*uid];
    remote_task_map.erase(*uid);

    DP("Found task (%ld%ld) in remote task map\n", task->uid.rank, task->uid.id);

    //Receive Task return code
    MPI_Recv(&task->return_code, 1, MPI_INT, source, SEND_RESULT_RETURN_CODE, targetdart_comm, MPI_STATUS_IGNORE);
    DP("Getting Return Code: %d for task (%ld%ld) in remote task map\n", task->return_code, task->uid.rank, task->uid.id);

    //Receive all parameter values
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs; i++) {
        //Test if data needs to be transfered to the kernel. Defined in omptarget.h (tgt_map_type).
        int64_t IsMapFrom = task->KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_FROM;
        if (IsMapFrom != 0) {
            MPI_Recv(task->KernelArgs->ArgPtrs[i], task->KernelArgs->ArgSizes[i], MPI_BYTE, source, SEND_RESULT_DATA, targetdart_comm, MPI_STATUS_IGNORE);
            DP("Recv result for task (%ld%ld) at " DPxMOD " from process %d\n", task->uid.rank, task->uid.id, DPxPTR(task->KernelArgs->ArgPtrs[i]), source);
        }
    }

    DP("Result transfer of task (%ld%ld) finished\n", task->uid.rank, task->uid.id);
    return TARGETDART_SUCCESS;
}

tdrc TD_Communicator::test_and_receive_results(td_uid_t *uid) {

    //test, if a task result can be received
    MPI_Status status;
    int flag;

    MPI_Iprobe(MPI_ANY_SOURCE, SEND_RESULT_UID, targetdart_comm, &flag, &status);
    if (flag == true) {
        DP("Result receival signaled\n");
        receive_task_result(status.MPI_SOURCE, uid);
        return TARGETDART_SUCCESS;
    }
    return TARGETDART_FAILURE;
}

td_global_sched_params_t TD_Communicator::global_cost_communicator(COST_DATA_TYPE local_cost_param) {
    COST_DATA_TYPE reduce = 0;
    COST_DATA_TYPE exScan = 0;
    COST_DATA_TYPE local_cost = local_cost_param;

    //TODO: use efficient combined implementation for allreduce and exscan
    MPI_Allreduce((void*) &local_cost, (void*) &reduce, 1, COST_MPI_DATA_TYPE, MPI_SUM, targetdart_comm);
    MPI_Exscan((void*) &local_cost, (void*) &exScan, 1, COST_MPI_DATA_TYPE, MPI_SUM, targetdart_comm);

    return {reduce, exScan, local_cost};
}

std::vector<COST_DATA_TYPE> TD_Communicator::global_cost_vector_propagation(COST_DATA_TYPE local_cost_param) {
    std::vector<COST_DATA_TYPE> cost_vector(size, 0);
    cost_vector[rank] = local_cost_param; 

    MPI_Allgather(&local_cost_param, 1, COST_MPI_DATA_TYPE, &cost_vector[0], 1, COST_MPI_DATA_TYPE, targetdart_comm);

    /*
    int iterations = std::ceil(std::log2(size));

    //TODO: avoid redundant data elements, iff the size is not a power of 2
    //TODO: utilize MPI_put
    for (int i = 0; i < iterations; i++) {
        int shift = std::pow(2,i);
        int target = (rank + shift) % size;
        //inverted shift to calculate the source rank for receiving
        int source = (rank - shift + size) % size;

        //Calculate the number of elements send for each iteration, covering cornercases
        int send1, send2, recv1, recv2;

        if (target < rank) {
            send1 = size - rank;
            send2 = target;
        } else {
            send1 = shift;
            send2 = 0;
        }

        if (source > rank) {
            recv1 = size - source;
            recv2 = rank;
        } else {
            recv1 = shift;
            recv2 = 0;
        }

        //send data 
        MPI_Request rqsts[4] = {MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL};
        //MPI_Isend(&cost_vector[rank], send1, COST_MPI_DATA_TYPE, target, 1, targetdart_comm, &rqsts[0]);
        if (send2 != 0) {
            //MPI_Isend(&cost_vector[0], send2, COST_MPI_DATA_TYPE, target, 2, targetdart_comm, &rqsts[2]);
        }
        //recv data
        //MPI_Irecv(&cost_vector[source], recv1, COST_MPI_DATA_TYPE, source, 1, targetdart_comm, &rqsts[1]);
        if (recv2 != 0) {
            //MPI_Irecv(&cost_vector[0], recv2, COST_MPI_DATA_TYPE, source, 2, targetdart_comm, &rqsts[3]);
        }

        MPI_Waitall(4, rqsts, MPI_STATUSES_IGNORE);
        // if (recv2 != 0) {
        //     MPI_Wait(&rqsts[3], MPI_STATUS_IGNORE);
        // }
        // if (send2 != 0) {
        //     MPI_Wait(&rqsts[2], MPI_STATUS_IGNORE);
        // }

    }*/

    return cost_vector;
}


bool TD_Communicator::test_finalization(bool local_finalize) {

    bool result = true;

    MPI_Allreduce(&local_finalize, &result, 1, MPI_C_BOOL, MPI_LAND, targetdart_comm);

    return result;
}