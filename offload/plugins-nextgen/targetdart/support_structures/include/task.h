#ifndef _TARGETDART_TASK_H
#define _TARGETDART_TASK_H


#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include "omptarget.h"

#include "PluginInterface.h"
#include "Shared/Environment.h"

#define gettid() syscall(SYS_gettid)

#define COST_DATA_TYPE int64_t

#define handle_error_en(en, msg) \
           do { errno = en; DP("ERROR: %s : %s\n", msg, strerror(en)); exit(EXIT_FAILURE); } while (0)

inline uint32_t getTraceLevel() {
  static uint32_t TraceLevel = 0;
  static std::once_flag Flag{};
  std::call_once(Flag, []() {
    if (char *EnvStr = getenv("TD_TRACE"))
      TraceLevel = std::stoi(EnvStr);
  });

  return TraceLevel;
}

#ifdef TD_TRACE

extern int trace_rank; // this MPI rank
extern FILE *trace_file; // if tracing is enabled, this will provide a handle for the tracefile (for this MPI rank)
extern unsigned long start_of_trace; // initial time stamp in microseconds => trace time-stamps are relative to this

#define TRACE(prefix, ...)                                                              \
    do {                                                                                \
        char buffer[100];                                                               \
        char buffer2[100];                                                              \
        snprintf(buffer, 99, "%s --> ", prefix);                                        \
        snprintf(buffer2, 99, __VA_ARGS__);                                             \
        if (trace_file != NULL) {						\
            fprintf(trace_file, "%s%s", buffer, buffer2);                       \
        }										\
  } while (false);

#define TRACE_START(...)                                                                          \
    if (getTraceLevel() > 0) {                                                                    \
        char bufferl[100];                                                                        \
        pid_t pid = gettid();                                                                     \
        struct timeval tv;                                                                        \
        gettimeofday(&tv,NULL);                                                                   \
        unsigned long usecs = (1000000 * tv.tv_sec) + tv.tv_usec;                                 \
        usecs -= start_of_trace;						          \
        sprintf(bufferl, "Starting targetDART trace on thread: %d at: %lu for type", pid, usecs); \
        TRACE(bufferl, __VA_ARGS__);                                                              \
    }
    
#define TRACE_END(...)                                                                            \
    if (getTraceLevel() > 0) {                                                                    \
        char bufferl[100];                                                                        \
        pid_t pid = gettid();                                                                     \
        struct timeval tv;                                                                        \
        gettimeofday(&tv,NULL);                                                                   \
        unsigned long usecs = (1000000 * tv.tv_sec) + tv.tv_usec;                                 \
        usecs -= start_of_trace;							  \
        sprintf(bufferl, "Ending targetDART trace on thread: %d at: %lu for type", pid, usecs);   \
        TRACE(bufferl, __VA_ARGS__);                                                              \
    }


#else
#define TRACE(...) {}
#define TRACE_START(...) {}
#define TRACE_END(...) {}
#endif

enum tdrc {TARGETDART_FAILURE, TARGETDART_SUCCESS};

// Main affinities
#define TD_CPU_OFFSET 2  // Task that should only run on the CPU
#define TD_OFFLOAD_OFFSET 5 // Task that should only run on an offload device
#define TD_ANY_OFFSET TD_CPU_OFFSET + TD_OFFLOAD_OFFSET + 1 // Task that can run on any hardware
// Sub affinities/priorities
#define TD_REPLICA_OFFSET -2 // The task was created on the local process and exists multiple times. Additional communication for cancelation necessary
#define TD_LOCAL_OFFSET -1 // The task must not be migrated to another process
#define TD_MIGRATABLE_OFFSET 0 // The task may be migrated to another process
#define TD_REMOTE_OFFSET 7 // The task was created on another device and exist only a single time accross the runtime
#define TD_REPLICATED_OFFSET 8 // The task was created on another device and exists multiple times. Additional communication for cancelation necessary

/* QUEUE Order:
physical_devices + 1 => Available GPUs + host device
=> Addressable scheduling queues by the user (also available as devices)
+ CPU_affinity(Local, Migratable, Replica) 
+ Offload_affinity(Local, Migratable, Replica) 
+ Any_affinity(Local, Migratable, Replica)
=> Internal scheduling queues required for managing remote/replicated data
+ CPU_affinity(Replicated, Remote) 
+ Offload_affinity(Replicated, Remote) 
+ Any_affinity(Replicated, Remote)
// two additional padding slots are required to ensure that GPU/ANY replicated, remote queues are accessed correctly.
*/

enum sub_affinity{MIGRATEABLE = TD_MIGRATABLE_OFFSET, LOCAL = TD_LOCAL_OFFSET, REPLICA = TD_REPLICA_OFFSET, REMOTE = TD_REMOTE_OFFSET, REPLICATED = TD_REPLICATED_OFFSET};
enum device_affinity{CPU = TD_CPU_OFFSET, GPU = TD_OFFLOAD_OFFSET, ANY = TD_ANY_OFFSET};

typedef struct td_uid_t{
    int64_t      id;
    int64_t      rank;

    bool operator == (const td_uid_t& t) const {
        return id == t.id && rank == t.rank;
    }
} td_uid_t;

typedef struct td_task_t{
    intptr_t            host_base_ptr;
    KernelArgsTy*       KernelArgs;
    ident_t*            Loc;
    td_uid_t            uid;
    COST_DATA_TYPE      cached_total_sizes;
    int                 return_code;
    device_affinity     affinity;
    bool                isReplica;
} td_task_t;

template <>
struct std::hash<td_uid_t>
{
    std::size_t operator()(const td_uid_t& uid) const {
        using std::size_t;
        using std::hash;
        using std::string;

        // Compute individual hash values for first,
        // second and combine them using XOR
        // and bit shifting:
        return ((hash<int>()(uid.rank) ^ (hash<int>()(uid.id) << 1)) >> 1);
    }
};

// array that holds image base addresses
extern std::vector<intptr_t> *_image_base_addresses;

tdrc init_task_structures();
tdrc finalize_task_structes();

/*
* Frees all memory assoziated with the task.
* This should onle be used to free data structures depending on locality
*/
tdrc delete_task(td_task_t *task, bool local);

/*
* Function set_image_base_address
* Sets base address of particular image index.
* This is necessary to determine the entry point for functions that represent a target construct
*/
tdrc set_image_base_address(size_t idx_image, intptr_t base_address);

/*
* Function apply_image_base_address
* Adds the base address to the address if iBaseAddress == true
* Else it creates a base address
*/
intptr_t apply_image_base_address(intptr_t base_address, bool isBaseAddress);

/*
* Executes the task on the given hardware device.
* TODO: test how a direct execution of the plugin may affect the 
*/
tdrc invoke_task(td_task_t *task, int64_t Device);

/*
* Initializes the reference pointer to normalize the task pointer.
* main_ptr needs to be the same function in the same binary in all processes.
*/
int add_main_ptr(void* main_ptr);


/**
* Function get_base_address
* Generates the base address for the current process
* 
* Works only for identical BINARIES
*/
tdrc get_base_address(void * main_ptr);


extern bool checkDeviceAndCtors(int64_t &DeviceID, ident_t *Loc);
#endif //_TARGETDART_TASK_H
