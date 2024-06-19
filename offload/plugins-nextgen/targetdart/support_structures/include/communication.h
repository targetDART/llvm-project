#include "mpi.h"
#include "task.h"
#include <cstddef>
#include <unordered_map>

#define COST_DATA_TYPE size_t
#define COST_MPI_DATA_TYPE MPI_LONG

enum MpiTaskTransferTag {SIGNAL_TASK_SEND, SEND_TASK, SEND_KERNEL_ARGS, SEND_PARAM_SIZES, SEND_PARAM_TYPES, SEND_BASE_PTRS, SEND_PARAMS, SEND_SOURCE_LOCS, SEND_LOCS_PSOURCE, SEND_RESULT_UID, SEND_RESULT_DATA, SEND_RESULT_RETURN_CODE};

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
    MPI_Comm targetdart_comm;


    // test if MPI was initialized by targedart
    bool did_initialize_mpi = false;

    // stores all tasks that are migrated or replicated to simplify receiving results.
    std::unordered_map<td_uid_t, td_task_t*> remote_task_map;

    tdrc declare_KernelArgs_type() {
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

    tdrc declare_task_type() {
        const int nitems = 4;
        int blocklengths[4] = {1,2,2,2};
        MPI_Datatype types[4] = {MPI_LONG, MPI_INT32_T, MPI_INT, MPI_LONG_LONG};
        MPI_Aint offsets[4];
        offsets[0] = (MPI_Aint) offsetof(td_task_t, host_base_ptr);
        offsets[1] = (MPI_Aint) offsetof(td_task_t, num_teams);
        offsets[2] = (MPI_Aint) offsetof(td_task_t, main_affinity);
        offsets[3] = (MPI_Aint) offsetof(td_task_t, uid);

        MPI_Type_create_struct(nitems, blocklengths, offsets, types, &TD_MPI_Task);
        MPI_Type_commit(&TD_MPI_Task);

        return TARGETDART_SUCCESS;
    }

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
    * Receives the output data from the process the task was migrated to
    */
    tdrc receive_task_result(int source);

    /**
    * Tests if a task result can be received, if yes it receives the result and returns success.
    * If not the function will return a failure;
    */
    tdrc test_and_receive_results();

    /**
    * Returns true, iff all participating processes want to finalize
    */
    bool test_finalization(COST_DATA_TYPE local_cost, bool finalize);
};
