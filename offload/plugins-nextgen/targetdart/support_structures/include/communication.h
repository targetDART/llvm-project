#ifndef _TARGETDART_COMMUNICTION_H
#define _TARGETDART_COMMUNICTION_H

#include "mpi.h"
#include "task.h"
#include <cstddef>
#include <unordered_map>

#define COST_DATA_TYPE int64_t
#define COST_MPI_DATA_TYPE MPI_LONG

enum MpiTaskTransferTag {SIGNAL_TASK_SEND, SEND_TASK, SEND_KERNEL_ARGS, SEND_PARAM_SIZES, SEND_PARAM_TYPES, SEND_BASE_PTRS, SEND_PARAMS, SEND_SOURCE_LOCS, SEND_LOCS_PSOURCE, SEND_NAME, SEND_RESULT_UID, SEND_RESULT_DATA, SEND_RESULT_RETURN_CODE};

typedef struct td_global_sched_params_t{
    COST_DATA_TYPE        total_cost;
    COST_DATA_TYPE        prefix_sum;
    COST_DATA_TYPE        local_cost;
} td_global_sched_params_t;

// class wrapping MPI and communication specific information
class TD_Communicator {
private:
    MPI_Datatype TD_Kernel_Args;
    MPI_Datatype TD_MPI_Task;

    // communicator for remote task requests
    MPI_Comm targetdart_comm = MPI_COMM_NULL;

    // MPI Session for MPI managment
    MPI_Session td_libhandle = MPI_SESSION_NULL;

    // MPI Return value
    int ret = 0;

    // stores all tasks that are migrated or replicated to simplify receiving results.
    std::unordered_map<td_uid_t, td_task_t*> remote_task_map;

    tdrc declare_KernelArgs_type();

    tdrc declare_task_type();

public:
    // initializes the communication and communcicator
    TD_Communicator();
    ~TD_Communicator();

    int size;
    int rank;

    // sends a task to another process
    tdrc send_task(int dest, td_task_t *task);

    // receives a task from another process
    tdrc receive_task(int source, td_task_t *task);

    /**
    * Tests if a task can be received, if yes it is received and the function returns success.
    * If not the function returns failure.
    */
    tdrc test_and_receive_tasks(td_task_t *task);

    /**
    * send the flag boolean value the the process of rank target
    */
    tdrc signal_task_send(int target, bool value);

    /**
    * receives the send signal and returns success if the flag is true
    */
    tdrc receive_signal_task_send(int source);

    /**
    * implements a global repratitioning of tasks accross all processes.
    */
    td_global_sched_params_t global_cost_communicator(COST_DATA_TYPE local_cost_param);

    /**
    * implements a log(n) based vector exchange within MPI
    */
    std::vector<COST_DATA_TYPE> global_cost_vector_propagation(COST_DATA_TYPE local_cost_param);

    /**
    * Sends the output data back to the owning process
    */
    tdrc send_task_result(td_task_t *task);

    /**
    * Receives the output data from the process the task was migrated to.
    * returns the uid of the received task.
    */
    tdrc receive_task_result(int source, td_uid_t *uid);

    /**
    * Tests if a task result can be received, if yes it receives the result and returns success.
    * If not the function will return a failure;
    * returns the uid of the task, iff one was received.
    */
    tdrc test_and_receive_results(td_uid_t *uid);

    /**
    * Returns true, iff all participating processes want to finalize
    */
    bool test_finalization(bool local_finalize);
};

#endif // _TARGETDART_COMMUNICTION_H