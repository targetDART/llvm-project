#include "TD_communication.h"
#include "omptarget.h"
#include "device.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include "private.h"
#include "mpi.h"
#include <link.h>
#include <queue>
#include <dlfcn.h>
#include <unistd.h>
#include "TargetDART.h"

// TODO: implement communication interface for TargetDART

bool td_send_task(int dest, td_task_t &task) {
    //Send Task Data
    MPI_Send(&task, 1, TD_MPI_Task, dest, SEND_TASK, targetdart_comm);
    //Send static KernelArgs values excluding pointervalues
    MPI_Send(task.KernelArgs, 1, TD_Kernel_Args, dest, SEND_KERNEL_ARGS, targetdart_comm);
    //Send Argument sizes for actual data transfers
    MPI_Send(task.KernelArgs->ArgSizes, task.KernelArgs->NumArgs, MPI_INT64_T, dest, SEND_PARAM_SIZES, targetdart_comm);
    //Send Argument types for each kernel
    MPI_Send(task.KernelArgs->ArgTypes, task.KernelArgs->NumArgs, MPI_INT64_T, dest, SEND_PARAM_TYPES, targetdart_comm);

    //Send all parameter values
    for (int i = 0; i < task.KernelArgs->NumArgs; i++) {
        //Test if data needs to be transfered to the kernel. Defined in omptarget.h (tgt_map_type).
        int64_t IsMapTo = task.KernelArgs->ArgTypes[i] & 0x001;
        if (IsMapTo)
            MPI_Send(task.KernelArgs->ArgPtrs[i], task.KernelArgs->ArgSizes[i], MPI_BYTE, dest, SEND_PARAMS, targetdart_comm);
    }

    // Send source location to support debuggin information
    MPI_Send(task.Loc, 4, MPI_INT32_T, dest, SEND_SOURCE_LOCS, targetdart_comm);

    //Base Pointers == pointers can be assumed for simple cases.
    //For complex combinations of pointers and scalars OMP breaks without our interference

    return TARGETDART_SUCCESS;
}

bool td_receive_task(td_task_t *task) {

    //Receive Task Data
    MPI_Recv(task, 1, TD_MPI_Task, MPI_ANY_SOURCE , SEND_TASK, targetdart_comm, MPI_STATUS_IGNORE);

    //Receive static KernelArgs values excluding pointervalues
    KernelArgsTy *KernelArgs = new KernelArgsTy;
    MPI_Recv(KernelArgs, 1, TD_Kernel_Args, MPI_ANY_SOURCE, SEND_KERNEL_ARGS, targetdart_comm, MPI_STATUS_IGNORE);
    task->KernelArgs = KernelArgs;
    
    //Receive Argument sizes for actual data transfers
    int64_t *sizes = new int64_t[task->KernelArgs->NumArgs];
    MPI_Recv(sizes, task->KernelArgs->NumArgs, MPI_INT64_T, MPI_ANY_SOURCE, SEND_PARAM_SIZES, targetdart_comm, MPI_STATUS_IGNORE);
    task->KernelArgs->ArgSizes = sizes;
    
    //Receive Argument types for each kernel
    int64_t *types = new int64_t[task->KernelArgs->NumArgs];
    MPI_Recv(types, task->KernelArgs->NumArgs, MPI_INT64_T, MPI_ANY_SOURCE, SEND_PARAM_TYPES, targetdart_comm, MPI_STATUS_IGNORE);
    task->KernelArgs->ArgTypes = types;

    //Declare Mappers and Names
    task->KernelArgs->ArgMappers = new void*[task->KernelArgs->NumArgs];
    task->KernelArgs->ArgNames = new void*[task->KernelArgs->NumArgs];

    //Receive all parameter values
    task->KernelArgs->ArgPtrs = new void*[task->KernelArgs->NumArgs];
    for (int i = 0; i < task->KernelArgs->NumArgs; i++) {
        //Test if data needs to be transfered to the kernel. Defined in omptarget.h (tgt_map_type).
        task->KernelArgs->ArgPtrs[i] = new int8_t[task->KernelArgs->ArgSizes[i]];
        int64_t IsMapTo = task->KernelArgs->ArgTypes[i] & 0x001;
        if (IsMapTo)
            MPI_Recv(task->KernelArgs->ArgPtrs[i], task->KernelArgs->ArgSizes[i], MPI_BYTE, MPI_ANY_SOURCE, SEND_PARAMS, targetdart_comm, MPI_STATUS_IGNORE);
    
        //Fill Mappers and Names with null
        task->KernelArgs->ArgMappers[i] = NULL;
        task->KernelArgs->ArgNames[i] = NULL;
    }



    // Receive source location to support debuggin information
    task->Loc = new ident_t;
    MPI_Recv(task->Loc, 4, MPI_INT32_T, MPI_ANY_SOURCE, SEND_SOURCE_LOCS, targetdart_comm, MPI_STATUS_IGNORE);

    //Base Pointers == pointers can be assumed for simple cases.
    //For complex combinations of pointers and scalars OMP breaks without our interference
    task->KernelArgs->ArgBasePtrs = task->KernelArgs->ArgPtrs;


    return TARGETDART_SUCCESS;
}