#include "../include/communication.h"
#include "../include/task.h"

#include "Shared/Debug.h"
#include <cstdint>

tdrc td_send_task(int dest, td_task_t *task) {

    //TODO: Use MPI pack to summarize the messages into a single Send
    //TODO: Use non-blocking send
    DP("Send task (%d%d) to process %d", task->uid.rank, task->uid.id, dest);

    //Send Task Data
    MPI_Send(task, 1, TD_MPI_Task, dest, SEND_TASK, targetdart_comm);
    //Send static KernelArgs values excluding pointervalues
    MPI_Send(task->KernelArgs, 1, TD_Kernel_Args, dest, SEND_KERNEL_ARGS, targetdart_comm);
    //Send Argument sizes for actual data transfers
    MPI_Send(task->KernelArgs->ArgSizes, task->KernelArgs->NumArgs, MPI_INT64_T, dest, SEND_PARAM_SIZES, targetdart_comm);
    //Send Argument types for each kernel
    MPI_Send(task->KernelArgs->ArgTypes, task->KernelArgs->NumArgs, MPI_INT64_T, dest, SEND_PARAM_TYPES, targetdart_comm);

    //Send the Base Pointer offsets for all arguments
    int64_t diff[task->KernelArgs->NumArgs];
    for (int i = 0; i < task->KernelArgs->NumArgs; i++) {
        diff[i] = ((int64_t) task->KernelArgs->ArgBasePtrs[i]) - ((int64_t) task->KernelArgs->ArgPtrs[i]);
    }
    MPI_Send(diff, task->KernelArgs->NumArgs, MPI_INT64_T, dest, SEND_BASE_PTRS, targetdart_comm);

    //Send all parameter values
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs; i++) {
        //Test if data needs to be transfered to the kernel. Defined in omptarget.h (tgt_map_type).
        int64_t IsMapTo = task->KernelArgs->ArgTypes[i] & 0x001;
        if (IsMapTo != 0)
            MPI_Send(task->KernelArgs->ArgPtrs[i], task->KernelArgs->ArgSizes[i], MPI_BYTE, dest, SEND_PARAMS, targetdart_comm);
    }

    // Send source location to support debuggin information
    MPI_Send(task->Loc, 4, MPI_INT32_T, dest, SEND_SOURCE_LOCS, targetdart_comm);

    //Send base location
    int64_t Locptr = (int64_t) apply_image_base_address((intptr_t) task->Loc->psource, false);
    MPI_Send(&Locptr, 1, MPI_INT64_T, dest, SEND_LOCS_PSOURCE, targetdart_comm);

    //Base Pointers == pointers can be assumed for simple cases.
    //For complex combinations of pointers and scalars OMP breaks without our interference

    DP("Send task (%d%d) to process %d finished", task->uid.rank, task->uid.id, dest);

    return TARGETDART_SUCCESS;
}

tdrc td_receive_task(int source, td_task_t *task) {

    //TODO: use MPI probe for complete receives

    DP("Receive new task from process %d", source);

    //Receive Task Data
    MPI_Recv(task, 1, TD_MPI_Task, source , SEND_TASK, targetdart_comm, MPI_STATUS_IGNORE);

    //Receive static KernelArgs values excluding pointervalues
    task->KernelArgs = new KernelArgsTy;
    MPI_Recv(task->KernelArgs, 1, TD_Kernel_Args, source, SEND_KERNEL_ARGS, targetdart_comm, MPI_STATUS_IGNORE);
    
    //Receive Argument sizes for actual data transfers
    task->KernelArgs->ArgSizes = new int64_t[task->KernelArgs->NumArgs];
    MPI_Recv(task->KernelArgs->ArgSizes, task->KernelArgs->NumArgs, MPI_INT64_T, source, SEND_PARAM_SIZES, targetdart_comm, MPI_STATUS_IGNORE);
    
    //Receive Argument types for each kernel
    task->KernelArgs->ArgTypes = new int64_t[task->KernelArgs->NumArgs];
    MPI_Recv(task->KernelArgs->ArgTypes, task->KernelArgs->NumArgs, MPI_INT64_T, source, SEND_PARAM_TYPES, targetdart_comm, MPI_STATUS_IGNORE);

    for (int i = 0; i < task->KernelArgs->NumArgs; i++) {    
        DP("Argument type of arg %d: " DPxMOD, i, DPxPTR(task->KernelArgs->ArgTypes[i]));
        assert(!(OMP_TGT_MAPTYPE_LITERAL & task->KernelArgs->ArgTypes[i]) && "Parameters should not be mapped implicitly as literals to avoid remote GPU errors. Use map clause on literals as well. (map(to:var))");
    }

    //Declare Mappers and Names
    task->KernelArgs->ArgMappers = new void*[task->KernelArgs->NumArgs];
    task->KernelArgs->ArgNames = new void*[task->KernelArgs->NumArgs];

    //Receive the Base Pointer offsets for all arguments
    int64_t diff[task->KernelArgs->NumArgs];
    MPI_Recv(diff, task->KernelArgs->NumArgs, MPI_INT64_T, source, SEND_BASE_PTRS, targetdart_comm,MPI_STATUS_IGNORE);
    
    //Receive all parameter values
    task->KernelArgs->ArgPtrs = new void*[task->KernelArgs->NumArgs];
    task->KernelArgs->ArgBasePtrs = new void*[task->KernelArgs->NumArgs];
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs; i++) {
        //Test if data needs to be transfered to the kernel. Defined in omptarget.h (tgt_map_type).
        //TODO: look into datatype and allocation
        task->KernelArgs->ArgBasePtrs[i] = std::malloc(task->KernelArgs->ArgSizes[i] + diff[i]);
        task->KernelArgs->ArgPtrs[i] = (void *) (((int64_t) task->KernelArgs->ArgBasePtrs[i]) + diff[i]);

        
        DP("Allocated memory for task (%d%d) at 0x%016x with size %d bytes", task->uid.rank, task->uid.id, task->KernelArgs->ArgPtrs[i], task->KernelArgs->ArgSizes[i]);  
        int64_t IsMapTo = task->KernelArgs->ArgTypes[i] & 0x001;
        if (IsMapTo != 0)
            MPI_Recv(task->KernelArgs->ArgPtrs[i], task->KernelArgs->ArgSizes[i], MPI_BYTE, source, SEND_PARAMS, targetdart_comm, MPI_STATUS_IGNORE);
    
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


    DP("Received task (%d%d) from process %d, finished", task->uid.rank, task->uid.id, source);


    return TARGETDART_SUCCESS;
}

tdrc td_test_and_receive_tasks(td_task_t *task) {

    //test, if a task result can be received
    MPI_Status status;
    int flag;

    MPI_Iprobe(MPI_ANY_SOURCE, SEND_TASK, targetdart_comm, &flag, &status);
    if (flag == true) {
        DP("Task receive signaled");
        td_receive_task(status.MPI_SOURCE, task);
        return TARGETDART_SUCCESS;
    }
    return TARGETDART_FAILURE;
}

tdrc td_signal_task_send(int target, bool value) {
    int flag = value;
    MPI_Send(&flag, 1, MPI_INT, target, SIGNAL_TASK_SEND, targetdart_comm);
    DP("Signal task send to process %d. Will send task %d", target, value);
    return TARGETDART_SUCCESS;
}

tdrc td_receive_signal_task_send(int source) {
    int flag;
    MPI_Recv(&flag, 1, MPI_INT, source, SIGNAL_TASK_SEND, targetdart_comm, MPI_STATUS_IGNORE);
    if (flag) {
        DP("Task send signaled by process %d", source);
        return TARGETDART_SUCCESS;
    }
    DP("Task send declined by process %d", source);
    return TARGETDART_FAILURE;
}

tdrc td_send_task_result(td_task_t *task) {

    //TODO: Use MPI pack to summarize the messages into a single Send
    //TODO: Use non-blocking send
    DP("Start result transfer of task (%d%d)", task->uid.rank, task->uid.id);
    //Send Task uid
    MPI_Send(&task->uid.id, 1, MPI_LONG_LONG, task->uid.rank, SEND_RESULT_UID, targetdart_comm);

    //Send Task returncode
    MPI_Send(&task->return_code, 1, MPI_INT, task->uid.rank, SEND_RESULT_RETURN_CODE, targetdart_comm);

    //Send all parameter values
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs; i++) {
        //Test if data needs to be transfered to the kernel. Defined in omptarget.h (tgt_map_type).
        int64_t IsMapFrom = task->KernelArgs->ArgTypes[i] & 0x002;
        if (IsMapFrom != 0) {
            MPI_Send(task->KernelArgs->ArgPtrs[i], task->KernelArgs->ArgSizes[i], MPI_BYTE, task->uid.rank, SEND_RESULT_DATA, targetdart_comm);
        }
    }

    //Base Pointers == pointers can be assumed for simple cases.
    //For complex combinations of pointers and scalars OMP breaks without our interference

    DP("Result transfer of task (%d%d) finished", task->uid.rank, task->uid.id);
    return TARGETDART_SUCCESS;
}

tdrc td_receive_task_result(int source) {


    DP("Start result receival");
    //TODO: use MPI probe for complete receives
    long long uid;
    //Receive Task Data
    MPI_Recv(&uid, 1, MPI_LONG_LONG, source, SEND_RESULT_UID, targetdart_comm, MPI_STATUS_IGNORE);

    td_task_t *task = td_remote_task_map[uid];
    td_remote_task_map.erase(uid);

    //Receive Task return code
    MPI_Recv(&task->return_code, 1, MPI_INT, source, SEND_RESULT_RETURN_CODE, targetdart_comm, MPI_STATUS_IGNORE);

    //Receive all parameter values
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs; i++) {
        //Test if data needs to be transfered to the kernel. Defined in omptarget.h (tgt_map_type).
        int64_t IsMapFrom = task->KernelArgs->ArgTypes[i] & 0x002;
        if (IsMapFrom != 0)
            MPI_Recv(task->KernelArgs->ArgPtrs[i], task->KernelArgs->ArgSizes[i], MPI_BYTE, source, SEND_RESULT_DATA, targetdart_comm, MPI_STATUS_IGNORE);
    }

    DP("Result transfer of task (%d%d) finished", task->uid.rank, task->uid.id);
    return TARGETDART_SUCCESS;
}

tdrc td_test_and_receive_results() {

    //test, if a task result can be received
    MPI_Status status;
    int flag;

    MPI_Iprobe(MPI_ANY_SOURCE, SEND_RESULT_UID, targetdart_comm, &flag, &status);
    if (flag == true) {
        DP("Result receival signaled");
        td_receive_task_result(status.MPI_SOURCE);
        return TARGETDART_SUCCESS;
    }
    return TARGETDART_FAILURE;
}

td_global_sched_params_t td_global_cost_communicator(COST_DATA_TYPE local_cost_param) {
    COST_DATA_TYPE reduce = 0;
    COST_DATA_TYPE exScan = 0;
    COST_DATA_TYPE local_cost = local_cost_param;

    //TODO: use efficient combined implementation for allreduce and exscan
    MPI_Allreduce((void*) &local_cost, (void*) &reduce, 1, COST_MPI_DATA_TYPE, MPI_SUM, targetdart_comm);
    MPI_Exscan((void*) &local_cost, (void*) &exScan, 1, COST_MPI_DATA_TYPE, MPI_SUM, targetdart_comm);

    return {reduce, exScan, local_cost};
}

std::vector<COST_DATA_TYPE> td_global_cost_vector_propagation(COST_DATA_TYPE local_cost_param) {
    std::vector<COST_DATA_TYPE> cost_vector(td_comm_size, 0);
    cost_vector[td_comm_rank] = local_cost_param; 

    MPI_Allgather(&local_cost_param, 1, COST_MPI_DATA_TYPE, &cost_vector[0], 1, COST_MPI_DATA_TYPE, targetdart_comm);

    /*
    int iterations = std::ceil(std::log2(td_comm_size));

    //TODO: avoid redundant data elements, iff the size is not a power of 2
    //TODO: utilize MPI_put
    for (int i = 0; i < iterations; i++) {
        int shift = std::pow(2,i);
        int target = (td_comm_rank + shift) % td_comm_size;
        //inverted shift to calculate the source rank for receiving
        int source = (td_comm_rank - shift + td_comm_size) % td_comm_size;

        //Calculate the number of elements send for each iteration, covering cornercases
        int send1, send2, recv1, recv2;

        if (target < td_comm_rank) {
            send1 = td_comm_size - td_comm_rank;
            send2 = target;
        } else {
            send1 = shift;
            send2 = 0;
        }

        if (source > td_comm_rank) {
            recv1 = td_comm_size - source;
            recv2 = td_comm_rank;
        } else {
            recv1 = shift;
            recv2 = 0;
        }

        //send data 
        MPI_Request rqsts[4] = {MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL};
        //MPI_Isend(&cost_vector[td_comm_rank], send1, COST_MPI_DATA_TYPE, target, 1, targetdart_comm, &rqsts[0]);
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


bool td_test_finalization(COST_DATA_TYPE cost, bool finalize) {

    // TODO: reimplement
}