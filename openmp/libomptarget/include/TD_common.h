
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

#define DEFAULT_TASK_COST 1
#define COST_DATA_TYPE long
#define COST_MPI_DATA_TYPE MPI_LONG
#define ITER_TILL_REPARTITION 100

//TODO: Add support for more accelerators (FPGA, Aurora etc.)
enum td_device_affinity {TD_CPU_AF=TARGETDART_CPU, TD_GPU_AF=TARGETDART_GPU, TD_ANY_AF=TARGETDART_ANY, TD_FPGA_AF=TARGETDART_FPGA, TD_VECTOR_AF=TARGETDART_VEC};

enum td_device_type {TD_CPU=0, TD_GPU=1, TD_FPGA=2, TD_VECTOR=3};


typedef struct td_task_t{
    intptr_t            host_base_ptr;
    KernelArgsTy*       KernelArgs;
    int32_t             num_teams;
    int32_t             thread_limit;
    ident_t*            Loc;
    int                 local_proc;
    td_device_affinity  affinity;
} td_task_t;

typedef struct td_global_sched_params_t{
    COST_DATA_TYPE        total_cost;
    COST_DATA_TYPE        prefix_sum;
    COST_DATA_TYPE        local_cost;
} td_global_sched_params_t;

extern MPI_Datatype TD_Kernel_Args;
extern MPI_Datatype TD_MPI_Task;


extern int td_comm_size;
extern int td_comm_rank;

tdrc declare_KernelArgs_type();

tdrc declare_task_type();

td_device_type get_device_from_affinity(td_device_affinity affinity);

#endif // _OMPTARGET_TD_COMMON_H

