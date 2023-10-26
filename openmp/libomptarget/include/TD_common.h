
#ifndef _OMPTARGET_TD_COMMON_H
#define _OMPTARGET_TD_COMMON_H

//TODO: define communication interface for TargetDART

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <pthread.h>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <sstream>
#include "omp.h"
#include "omptarget.h"
#include "device.h"
#include "mpi.h"
#include "pthread.h"
#include <time.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <stdarg.h>


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
#define SPECIFIC_DEVICE_RANGE_START TARGETDART_DEVICE(0)
#define NUM_FLEXIBLE_AFFINITIES 5

template< typename... Args >
static std::string format2string(const char* format, Args... args) {
    const size_t SIZE = std::snprintf( NULL, 0, format, args...);

    std::string res;
    res.resize(SIZE+1);
    std::snprintf( &(res[0]), SIZE+1, format, args...);
    return std::move(res);
}

template <typename... Args>
static void __td_dbg_print_help(int print_prefix, const char * prefix, int rank, const char* format, Args... args) {
    timeval curTime;
    gettimeofday(&curTime, NULL);
    int milli = curTime.tv_usec / 1000;
    int micro_sec = curTime.tv_usec % 1000;
    char buffer [80];
    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", localtime(&curTime.tv_sec));
    char currentTime[84] = "";
    sprintf(currentTime, "%s.%03d.%03d", buffer, milli, micro_sec);
    std::string message = format2string(format, args...);

    std::ostringstream stream;
    stream << currentTime << " " << prefix << " #R" << rank << ": --> " << message << std::endl;
    std::cerr << stream.str();
}

template <typename... Args>
static void __td_dbg_print(int rank, const char* format, Args... args) {
    __td_dbg_print_help(1, "targeDARTLib", rank, format, args...);
}

#ifndef DB_TD
#ifdef TARGETDART_DEBUG
#define DBP( ... )  __td_dbg_print(td_comm_rank, __VA_ARGS__) 
#else
#define DB_TD( ... ) __td_dbg_print(td_comm_rank, __VA_ARGS__)
#endif
#endif

//TODO: Add support for more accelerators (FPGA, Aurora etc.)
enum td_device_affinity {TD_CPU=TARGETDART_CPU - DEVICE_BASE, TD_GPU=TARGETDART_GPU - DEVICE_BASE, 
                        TD_ANY=TARGETDART_ANY - DEVICE_BASE, TD_FPGA=TARGETDART_FPGA - DEVICE_BASE, 
                        TD_VECTOR=TARGETDART_VEC - DEVICE_BASE, TD_FIXED};

enum td_queue_class {TD_LOCAL=0, TD_REMOTE=1, TD_REPLICA=2};

#define TD_AFFINITIES {TD_ANY, TD_GPU, TD_CPU, TD_VECTOR, TD_FPGA}
#define TD_NUM_AFFINITIES 5


typedef struct td_task_t{
    intptr_t            host_base_ptr;
    KernelArgsTy*       KernelArgs;
    int32_t             num_teams;
    int32_t             thread_limit;
    ident_t*            Loc;
    int                 local_proc;
    td_device_affinity  affinity;
    long long           uid;
    int                 return_code;
} td_task_t;

typedef struct td_global_sched_params_t{
    COST_DATA_TYPE        total_cost;
    COST_DATA_TYPE        prefix_sum;
    COST_DATA_TYPE        local_cost;
} td_global_sched_params_t;

typedef struct td_pthread_conditional_wrapper_t {
    pthread_mutex_t thread_mutex;
    pthread_cond_t  conditional;
} td_pthread_conditional_wrapper_t;

extern MPI_Datatype TD_Kernel_Args;
extern MPI_Datatype TD_MPI_Task;
extern std::atomic<bool> *td_finalize;
extern std::unordered_map<long long, td_pthread_conditional_wrapper_t*> td_task_conditional_map;


extern int td_comm_size;
extern int td_comm_rank;
extern std::unordered_map<long long, td_task_t*> td_remote_task_map;
extern std::atomic<long> num_offloaded_tasks;

tdrc declare_KernelArgs_type();

tdrc declare_task_type();

/**
* lets the current thread sleep until a signal for the task with uid task_uid sends a signal, indicating, that it finished
*/
void td_yield(td_task_t *task);

/**
* Sends a signal that the task with uid task_uid has finished execution, so OMP can resume its management
*/
void td_signal(td_task_t *task);

#endif // _OMPTARGET_TD_COMMON_H

