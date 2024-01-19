#include "TargetDART.h"
#include "omptarget.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include "private.h"
#include <link.h>
#include <queue>
#include <dlfcn.h>
#include <unistd.h>
#include "TD_common.h"

tdrc declare_KernelArgs_type() {
    const int nitems = 3;
    int blocklengths[3] = {2,2,3};
    MPI_Datatype types[3] = {MPI_UINT32_T, MPI_UINT64_T,MPI_UINT32_T};
    MPI_Aint offsets[3];
    offsets[0] = (MPI_Aint) offsetof(KernelArgsTy, Version);
    offsets[1] = (MPI_Aint) offsetof(KernelArgsTy, Tripcount);
    offsets[2] = (MPI_Aint) offsetof(KernelArgsTy, NumTeams);

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
    //TODO: potentially empty or not defined look into setup
    td_pthread_conditional_wrapper_t *cond_var = conditional_map.get_conditional(task->uid);

    DB_TD("yield OMP hidden helper thread until task (%d%d) is finished", task->local_proc, task->uid);
    pthread_cond_wait(&cond_var->conditional,&cond_var->thread_mutex);
    pthread_mutex_unlock(&cond_var->thread_mutex);

    DB_TD("Helper yield for task (%d%d) finished", task->local_proc, task->uid);
}

void td_signal(td_task_t *task) {
    DB_TD("resume OMP hidden helper thread since task (%d%d) finished", task->local_proc, task->uid);
    td_pthread_conditional_wrapper_t *cond_var = conditional_map.get_conditional(task->uid);

    DB_TD("segfault test for task (%d%d)", task->local_proc, task->uid);

    pthread_mutex_lock(&cond_var->thread_mutex);
    DB_TD("segfault test for task (%d%d) finished", task->local_proc, task->uid);
    
    pthread_cond_signal(&cond_var->conditional);
    pthread_mutex_unlock(&cond_var->thread_mutex);
    DB_TD("Continuation signal for task (%d%d) finished", task->local_proc, task->uid);
}

TD_Conditional_Map::TD_Conditional_Map() {
    conditional_map = new std::unordered_map<td_uid_t, td_pthread_conditional_wrapper_t*>();
    mapmutex = PTHREAD_MUTEX_INITIALIZER;
}

TD_Conditional_Map::~TD_Conditional_Map() {
    //cleanup contents 
    for (auto& entry : *conditional_map) {
        delete entry.second;
    }
    delete conditional_map;
}

// Thread safe map access
void TD_Conditional_Map::add_conditional(td_uid_t tid){
    td_pthread_conditional_wrapper_t *cond_var = new td_pthread_conditional_wrapper_t({PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER});
    pthread_mutex_lock(&mapmutex);
    conditional_map[0][tid] = cond_var;
    pthread_mutex_unlock(&mapmutex);
}

// Thread safe map access
td_pthread_conditional_wrapper_t* TD_Conditional_Map::get_conditional(td_uid_t tid){
    td_pthread_conditional_wrapper_t* res;
    pthread_mutex_lock(&mapmutex);
    res = conditional_map[0][tid];
    pthread_mutex_unlock(&mapmutex);

    return res;
}