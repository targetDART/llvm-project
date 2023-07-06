#ifndef _OMPTARGET_TD_COMMUNICATION_H
#define _OMPTARGET_TD_COMMUNICATION_H

//TODO: define communication interface for TargetDART

#include <cstdint>
#include "omptarget.h"
#include "device.h"
#include "mpi.h"

// communicator for remote task requests
extern MPI_Comm targetdart_comm;
// communicator for sending back mapped values
extern MPI_Comm targetdart_comm_mapped;
// communicator for load information
extern MPI_Comm targetdart_comm_load;
// communicator for task cancellation
extern MPI_Comm targetdart_comm_cancel;
// communicator for activating replicated tasks
extern MPI_Comm targetdart_comm_activate;

extern int targetdart_comm_size;
extern int targetdart_comm_rank;


extern MPI_Datatype TD_Kernel_Args;
extern MPI_Datatype TD_MPI_Task;

enum MpiTaskTransferTag {SEND_TASK, SEND_KERNEL_ARGS, SEND_PARAM_SIZES, SEND_PARAM_TYPES, SEND_PARAMS, SEND_SOURCE_LOCS};

#endif // _OMPTARGET_TD_COMMUNICATION_H