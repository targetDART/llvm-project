#ifndef _OMPTARGET_TD_COMMUNICATION_H
#define _OMPTARGET_TD_COMMUNICATION_H

//TODO: define communication interface for TargetDART

#include <cstdint>
#include "omptarget.h"
#include "device.h"
#include "mpi.h"
#include "TD_common.h"




extern MPI_Datatype TD_Kernel_Args;
extern MPI_Datatype TD_MPI_Task;

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

enum MpiTaskTransferTag {SEND_TASK, SEND_KERNEL_ARGS, SEND_PARAM_SIZES, SEND_PARAM_TYPES, SEND_PARAMS, SEND_SOURCE_LOCS, SEND_LOCS_PSOURCE};

int td_send_task(int dest, td_task_t &task);

int td_receive_task(int source, td_task_t *task);

#endif // _OMPTARGET_TD_COMMUNICATION_H