#include "targetDART/TargetDART.h"
#include "omptarget.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "private.h"
#include <link.h>
#include <mutex>
#include <pthread.h>
#include <queue>
#include <dlfcn.h>
#include <unistd.h>
#include "targetDART/TD_common.h"

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
    int blocklengths[4] = {1,2,2,1};
    MPI_Datatype types[4] = {MPI_LONG, MPI_INT32_T, MPI_INT, MPI_LONG_LONG};
    MPI_Aint offsets[4];
    offsets[0] = (MPI_Aint) offsetof(td_task_t, host_base_ptr);
    offsets[1] = (MPI_Aint) offsetof(td_task_t, num_teams);
    offsets[2] = (MPI_Aint) offsetof(td_task_t, local_proc);
    offsets[3] = (MPI_Aint) offsetof(td_task_t, uid);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &TD_MPI_Task);
    MPI_Type_commit(&TD_MPI_Task);

    return TARGETDART_SUCCESS;
}


void td_yield(td_task_t *task) {
    td_conditional_wrapper_t *cond_var = conditional_map->get_conditional(task->uid);
    //Lock the function from this point onwards
    {
        std::unique_lock<std::mutex> lock(cond_var->thread_mutex);

        DB_TD("yield OMP hidden helper thread until task (%d%d) is finished", task->local_proc, task->uid);
        //wait until signal and ready to avoid spurrious waiting
        cond_var->conditional.wait(lock, [cond_var](){return cond_var->ready;});
    }

    DB_TD("Helper yield for task (%d%d) finished", task->local_proc, task->uid);
}

void td_signal(td_task_t *task) {
    DB_TD("resume OMP hidden helper thread since task (%d%d) finished", task->local_proc, task->uid);
    td_conditional_wrapper_t *cond_var = conditional_map->get_conditional(task->uid);

    DB_TD("segfault test for task (%d%d)", task->local_proc, task->uid);

    //Lock the function from this point onwards
    {
        std::lock_guard<std::mutex> lock(cond_var->thread_mutex);

        DB_TD("segfault test for task (%d%d) finished", task->local_proc, task->uid);

        cond_var->ready = true;
    
        cond_var->conditional.notify_one();
    }
    DB_TD("Continuation signal for task (%d%d) finished", task->local_proc, task->uid);
}

TD_Conditional_Map::TD_Conditional_Map() {
    conditional_map = new std::unordered_map<td_uid_t, td_conditional_wrapper_t*>();
    DB_TD("Initialized Conditional Map");
}

TD_Conditional_Map::~TD_Conditional_Map() {
    //cleanup contents 
    for (auto& entry : *conditional_map) {
        delete entry.second;
    }
    delete conditional_map;
}

// Thread safe map access
td_conditional_wrapper_t* TD_Conditional_Map::add_conditional(td_uid_t tid){
    td_conditional_wrapper_t *cond_var = new td_conditional_wrapper_t();
    DB_TD("Entering mutex for conditional creation for task (%d%d)", td_comm_rank, tid);
    mapmutex.lock();
    DB_TD("Entered mutex for conditional creation for task (%d%d)", td_comm_rank, tid);
    conditional_map[0][tid] = cond_var;
    mapmutex.unlock();
    return cond_var;
}

// Thread safe map access
td_conditional_wrapper_t* TD_Conditional_Map::get_conditional(td_uid_t tid){
    td_conditional_wrapper_t* res;
    DB_TD("Entering mutex for conditional read for task (%d%d)", td_comm_rank, tid);
    mapmutex.lock();
    DB_TD("Entered mutex for conditional read for task (%d%d)", td_comm_rank, tid);
    res = conditional_map[0][tid];
    mapmutex.unlock();

    return res;
}