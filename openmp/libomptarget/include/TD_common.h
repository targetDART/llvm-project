
#ifndef _OMPTARGET_TD_COMMON_H
#define _OMPTARGET_TD_COMMON_H

//TODO: define communication interface for TargetDART

#include <cstdint>
#include "omp.h"
#include "omptarget.h"
#include "device.h"
#include "mpi.h"


#define handle_error_en(en, msg) \
           do { errno = en; DBP("ERROR: %s : %s\n", msg, strerror(en)); exit(EXIT_FAILURE); } while (0)

enum tdrc {TARGETDART_FAILURE, TARGETDART_SUCCESS};

#ifndef DBP
#ifdef TD_DEBUG
#define DBP( ... ) { RELP(__VA_ARGS__); }
#else
#define DBP( ... ) { }
#endif
#endif

//TODO: Add support for more accelerators (FPGA, Aurora etc.)
enum td_device_affinity {TD_CPU=TARGETDART_CPU, TD_GPU=TARGETDART_GPU, TD_ANY=TARGETDART_ANY};

typedef struct td_task_t{
    intptr_t            host_base_ptr;
    KernelArgsTy*       KernelArgs;
    int32_t             num_teams;
    int32_t             thread_limit;
    ident_t*            Loc;
    int                 local_proc;
    td_device_affinity  affinity;
} td_task_t;


extern MPI_Datatype TD_Kernel_Args;
extern MPI_Datatype TD_MPI_Task;

tdrc declare_KernelArgs_type();

tdrc declare_task_type();

#endif // _OMPTARGET_TD_COMMON_H

