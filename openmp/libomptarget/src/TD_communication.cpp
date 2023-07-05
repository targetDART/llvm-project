#include "TD_communication.h"
#include "omptarget.h"
#include "device.h"
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
    //Send static KernelArgs values excluding pointervalues
    MPI_Send((void *) task.KernelArgs, 1, TD_Kernel_Args, dest, 0, targetdart_comm);
    //Send Argument sizes for actual data transfers
    MPI_Send(task.KernelArgs->ArgSizes, task.KernelArgs->NumArgs, MPI_INT64_T, dest, 1, targetdart_comm);
    //Send Argument types for each kernel
    MPI_Send(task.KernelArgs->ArgTypes, task.KernelArgs->NumArgs, MPI_INT64_T, dest, 2, targetdart_comm);

    for (int i = 0; i < task.KernelArgs->NumArgs; i++) {
        if (task.KernelArgs->ArgTypes)
        MPI_Send(task.KernelArgs->ArgPtrs[i], task.KernelArgs->ArgSizes[i], MPI_BYTE, dest, 4, targetdart_comm);
    }

    //TODO: figure out base pointers in Kernel args

    return TARGETDART_SUCCESS;
}

bool td_receive_task(td_task_t *task) {
    //TODO implement

    //TODO: figure out base pointers in Kernel args

    return TARGETDART_SUCCESS;
}