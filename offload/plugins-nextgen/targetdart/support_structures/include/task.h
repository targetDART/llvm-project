#ifndef _TARGETDART_TASK_H
#define _TARGETDART_TASK_H


#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include "omptarget.h"

#include "PluginInterface.h"

#include "Shared/Environment.h"

enum tdrc {TARGETDART_FAILURE, TARGETDART_SUCCESS};

// Main affinities
#define TD_CPU_OFFSET 3  // Task that should only run on the CPU
#define TD_OFFLOAD_OFFSET 6 // Task that should only run on an offload device
#define TD_ANY_OFFSET TD_CPU_OFFSET + TD_OFFLOAD_OFFSET // Task that can run on any hardware
// Sub affinities/priorities
#define TD_MIGRATABLE_OFFSET 0 // The task may be migrated to another process
#define TD_LOCAL_OFFSET -1 // The task must not be migrated to another process
#define TD_REPLICA_OFFSET -2 // The task was created on the local process and exists multiple times. Additional communication for cancelation necessary
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
*/

enum sub_affinity{KMIGRATEABLE = TD_MIGRATABLE_OFFSET, LOCAL = TD_LOCAL_OFFSET, REPLICA = TD_REPLICA_OFFSET, REMOTE = TD_REMOTE_OFFSET, REPLICATED = TD_REPLICATED_OFFSET};
enum device_affinity{CPU = TD_CPU_OFFSET, GPU = TD_OFFLOAD_OFFSET, ANY = TD_ANY_OFFSET};

typedef struct td_uid_t{
    int      id;
    int      rank;
} td_uid_t;

typedef struct td_task_t{
    intptr_t            host_base_ptr;
    KernelArgsTy*       KernelArgs;
    bool                isReplica;
    int32_t             num_teams;
    int32_t             thread_limit;
    ident_t*            Loc;
    device_affinity     main_affinity;
    sub_affinity        sub_affinity;
    td_uid_t            uid;
    int                 return_code;
} td_task_t;

#endif //_TARGETDART_TASK_H