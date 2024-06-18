#include "mpi.h"
#include "task.h"
#include <cstddef>

#define COST_DATA_TYPE size_t
#define COST_MPI_DATA_TYPE MPI_LONG

extern MPI_Datatype TD_Kernel_Args;
extern MPI_Datatype TD_MPI_Task;

// communicator for remote task requests
extern MPI_Comm targetdart_comm;

extern int td_comm_size;
extern int td_comm_rank;

typedef struct td_global_sched_params_t{
    COST_DATA_TYPE        total_cost;
    COST_DATA_TYPE        prefix_sum;
    COST_DATA_TYPE        local_cost;
} td_global_sched_params_t;

enum MpiTaskTransferTag {SIGNAL_TASK_SEND, SEND_TASK, SEND_KERNEL_ARGS, SEND_PARAM_SIZES, SEND_PARAM_TYPES, SEND_BASE_PTRS, SEND_PARAMS, SEND_SOURCE_LOCS, SEND_LOCS_PSOURCE, SEND_RESULT_UID, SEND_RESULT_DATA, SEND_RESULT_RETURN_CODE};

tdrc td_send_task(int dest, td_task_t *task);

tdrc td_receive_task(int source, td_task_t *task);


/**
* Tests if a task can be received, if yes it is received and the function returns success.
* If not the function returns failure.
*/
tdrc td_test_and_receive_tasks(td_task_t *task);

/**
* send the flag boolean value the the process of rank target
*/
tdrc td_signal_task_send(int target, bool value);

/**
* receives the send signal and returns success if the flag is true
*/
tdrc td_receive_signal_task_send(int source);

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
* Tests if a task result can be received, if yes it receives the result and returns success.
* If not the function will return a failure;
*/
tdrc td_test_and_receive_results();

/**
* Returns true, iff all participating processes want to finalize
*/
bool td_test_finalization(COST_DATA_TYPE local_cost, bool finalize);