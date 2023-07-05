#ifndef _OMPTARGET_TARGETDART_H
#define _OMPTARGET_TARGETDART_H

//TODO: define interface

#include <cstdint>
#include "mpi.h"
#include "omptarget.h"
#include "device.h"
#include "TD_communication.h"

enum tdrc {TARGETDART_FAILURE, TARGETDART_SUCCESS};

#ifndef DBP
#ifdef TD_DEBUG
#define DBP( ... ) { RELP(__VA_ARGS__); }
#else
#define DBP( ... ) { }
#endif
#endif

typedef struct td_task_t{
    void*           host_ptr;
    KernelArgsTy*   KernelArgs;
    int32_t         num_teams;
    int32_t         thread_limit;
    ident_t*        Loc;
} td_task_t;

extern MPI_Datatype TD_Kernel_Args;

// Outsources a target construct to the targetDART runtime
int addTargetDARTTask( ident_t *Loc, int32_t NumTeams,
                        int32_t ThreadLimit, void *HostPtr,
                        KernelArgsTy *KernelArgs, int64_t *DeviceId);

tdrc get_base_address(void * main_ptr);

int32_t set_image_base_address(int idx_image, intptr_t base_address);

tdrc declare_KernelArgs_type();


extern "C" int initTargetDART(int *argc, char ***argv, void* main_ptr);

extern "C" int finalizeTargetDART();

extern "C" int testFunction(int *, char ***);

#endif // _OMPTARGET_TARGETDART_H