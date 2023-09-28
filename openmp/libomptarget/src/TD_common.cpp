#include "TargetDART.h"
#include "omptarget.h"
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
    int blocklengths[4] = {1,2,1,1};
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


void td_yield(long long task_uid) {
    td_pthread_conditional_wrapper_t *cond_var = td_task_conditional_map[task_uid];
    pthread_mutex_lock(&cond_var->thread_mutex);
    pthread_cond_wait(&cond_var->conditional,&cond_var->thread_mutex);
    pthread_mutex_unlock(&cond_var->thread_mutex);
}

void td_signal(long long task_uid) {
    td_pthread_conditional_wrapper_t *cond_var = td_task_conditional_map[task_uid];
    pthread_mutex_lock(&cond_var->thread_mutex);
    pthread_cond_signal(&cond_var->conditional);
    pthread_mutex_unlock(&cond_var->thread_mutex);
}
