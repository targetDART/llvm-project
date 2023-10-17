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

extern int td_comm_size;
extern int td_comm_rank;

enum MpiTaskTransferTag {SEND_TASK, SEND_KERNEL_ARGS, SEND_PARAM_SIZES, SEND_PARAM_TYPES, SEND_PARAMS, SEND_SOURCE_LOCS, SEND_LOCS_PSOURCE, SEND_RESULT_UID, SEND_RESULT_DATA, SEND_RESULT_RETURN_CODE};

int td_send_task(int dest, td_task_t *task);

int td_receive_task(int source, td_task_t *task);

/**
* implements a global repratitioning of tasks accross all processes.
*/
td_global_sched_params_t td_global_cost_communicator(COST_DATA_TYPE local_cost_param);

/**
* implements a log(n) based vector exchange within MPI
*/
std::vector<COST_DATA_TYPE> td_global_cost_vector_propagation(COST_DATA_TYPE local_cost_param);

/**
* Sends the output data back to the owning process
*/
tdrc td_send_task_result(td_task_t *task);

/**
* Receives the output data from the process the task was migrated to
*/
tdrc td_receive_task_result(int source);

/**
* Returns true, iff all participating processes want to finalize
*/
bool td_test_finalization(std::atomic<bool> local_finalize);

#endif // _OMPTARGET_TD_COMMUNICATION_H