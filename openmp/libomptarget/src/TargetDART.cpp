#include "TargetDART.h"
#include "omp.h"
#include "omptarget.h"
#include "device.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "private.h"
#include "mpi.h"
#include <link.h>
#include <pthread.h>
#include <queue>
#include <dlfcn.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include "TD_communication.h"
#include "TD_common.h"
#include "TD_cost_estimation.h"
#include "TD_scheduling.h"
#include "TD_comm_thread.h"


// TODO: implement interface

// Mixed endianness is not supported

static bool __td_initialized = false;

static bool __td_did_initialize_mpi = false;

// stores all tasks that are migrated or replicated to simplify receiving results.
std::unordered_map<long long, td_task_t*> td_remote_task_map;


std::vector<long long> __td_tasks_generated_per_thread;

// communicator for remote task requests
MPI_Comm targetdart_comm;

int td_comm_size;
int td_comm_rank;

bool td_finalize;

MPI_Datatype TD_Kernel_Args;
MPI_Datatype TD_MPI_Task;

// array that holds image base addresses
std::vector<intptr_t> _image_base_addresses;
std::unordered_map<long long, td_pthread_conditional_wrapper_t*> td_task_conditional_map;

// initial scheduler
// TODO: replace with correct scheduler
static inline int DeviceIdGenerator(KernelArgsTy *KernelArgs) {
  return KernelArgs->NumArgs % (omp_get_num_devices() + 1);
}

//Adds a task to the TargetDART runtime
int td_add_task( ident_t *Loc, int32_t NumTeams,
                        int32_t ThreadLimit, void *HostPtr,
                        KernelArgsTy *KernelArgs, int64_t *DeviceId) 
{
  *DeviceId = DeviceIdGenerator(KernelArgs);
  // create internal task data structure
  td_task_t *task = (td_task_t*) std::malloc(sizeof(td_task_t));
  task->host_base_ptr = apply_image_base_address((intptr_t) HostPtr, false);
  task->KernelArgs = KernelArgs;
  task->Loc = Loc;
  task->num_teams = NumTeams;
  task->thread_limit = ThreadLimit;
  task->local_proc = td_comm_rank;
  task->uid = (__td_tasks_generated_per_thread[omp_get_thread_num()] << 4) + omp_get_thread_num();
  __td_tasks_generated_per_thread[omp_get_thread_num()]++;

  td_pthread_conditional_wrapper_t cond_var = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER};
  td_task_conditional_map[task->uid] = &cond_var;

  //initial assignment
  if (*DeviceId >= SPECIFIC_DEVICE_RANGE_START) {
    task->affinity = TD_FIXED;
    //TODO: add to fixed local queue. Tasks in this queue are not supposed to be migrated, but help the cost estimation.
  } else  {
    task->affinity = (td_device_affinity) (*DeviceId - DEVICE_BASE);
    td_add_to_load_local(task);
  }

  //yields until the current task has finished.
  td_yield(task->uid);
  /*
  Number of hidden_helper_threads is defined by __kmp_hidden_helper_threads_num in kmp_runtime.cpp line 9142
  Default is currently 8
  */

  /*if (targetdart_comm_rank == 0) {
    td_send_task(1, task);
  } else {
    td_receive_task(0, task);
  }*/

  // TODO: make sure the current thread sleeps here until the task is executed.
  // Run a dedicated device thread which consumes the task for its assigned device. 
  //(think about multiple threads per device, at least for GPUs to overlap com/comp)
  // ensure that the local thread cannot progress until the task is finished and the Data is available again 
  // Ideas: yield + ping, Barrier, Mutex, busy-waiting
  // requires additional parameters in task

  return task->return_code;
}

// initializes the targetDART lib
/*
main_ptr needs to be the same function in the same binary in all processes.
Preferably main().
If MPI is used in the user code, MPI must be initialized before TargetDART.
*/
int initTargetDART(void* main_ptr) {
  if (__td_initialized) {
    return TARGETDART_SUCCESS;
  }

  // check whether MPI is initialized, otherwise do so
  int mpi_initialized, err;
  mpi_initialized = 0;
  int provided;
  err = MPI_Initialized(&mpi_initialized);
  if(!mpi_initialized) {
      // MPI_Init(NULL, NULL);
      MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
      __td_did_initialize_mpi = true;
  }
  MPI_Query_thread(&provided);
  if(provided != MPI_THREAD_MULTIPLE) {
    //TODO: Fehler meldung
  }

  //decare KernelArgs,task as MPI Type
  declare_KernelArgs_type();
  declare_task_type();

  // create separate communicator for targetdart
  //TODO: reduce to single communicator for coordination (Deadlock danger?)
  err = MPI_Comm_dup(MPI_COMM_WORLD, &targetdart_comm);
  MPI_Comm_size(targetdart_comm, &td_comm_size);
  MPI_Comm_rank(targetdart_comm, &td_comm_rank);

  //Initialize the data structures for scheduling
  td_local_task_queues = std::vector<TD_Task_Queue>(omp_get_num_devices() + NUM_FLEXIBLE_AFFINITIES);
  __td_tasks_generated_per_thread = std::vector<long long>(omp_get_num_threads(), 0);
  std::unordered_map<intptr_t,std::vector<double>> td_cost;
  // Initialize the map of remote and replicated tasks
  td_remote_task_map = std::unordered_map<long long, td_task_t*>();
  td_task_conditional_map = std::unordered_map<long long, td_pthread_conditional_wrapper_t*>();


  // define the base address of the current process
  get_base_address(main_ptr);

  // Init devices during installation
  for (long i = 0; i < omp_get_num_devices(); i++) {
    if (checkDeviceAndCtors(i, nullptr)) {
      DP("Not offloading to device %" PRId64 "\n", DeviceId);
      return TARGETDART_FAILURE;
    }
  }

  td_finalize = true;

  return TARGETDART_SUCCESS;
}

/* finalizes targetDART lib, iff TD initilized MPI it also needs to finalize it.
 * Should be one of the last functions in your program.
*/
int finalizeTargetDART() {
  if (__td_did_initialize_mpi) {
    MPI_Finalize();
  }
  //TODO: synchronize all threads and processes
  td_finalize = true;
  return TARGETDART_SUCCESS;
}

/*
 * Function set_image_base_address
 * Sets base address of particular image index.
 * This is necessary to determine the entry point for functions that represent a target construct
 */
int32_t set_image_base_address(int idx_image, intptr_t base_address) {
    if(_image_base_addresses.size() < idx_image+1) {
        _image_base_addresses.resize(idx_image+1);
    }
    // set base address for image (last device wins)
    DBP("set_image_base_address (enter) Setting base_address: " DPxMOD " for img: %d\n", DPxPTR((void*)base_address), idx_image);
    _image_base_addresses[idx_image] = base_address;
    return TARGETDART_SUCCESS;
}

/*
 * Function apply_image_base_address
 * Adds the base address to the address if iBaseAddress == true
 * Else it creates a base address
 */
intptr_t apply_image_base_address(intptr_t base_address, bool isBaseAddress) {
    if (isBaseAddress) {
      return base_address + _image_base_addresses[99];
    }
    return base_address - _image_base_addresses[99];
}

/**
* Function get_base_address
* Generates the base address for the current process
* 
* Works only for identical BINARIES
*/
tdrc get_base_address(void * main_ptr) {
  Dl_info info;
  int rc;
  link_map * map = (link_map *)malloc(1000*sizeof(link_map));
  void *start_ptr = (void*)map;
  // struct link_map map;
  rc = dladdr1(main_ptr, &info, (void**)&map, RTLD_DL_LINKMAP);
  // Store base pointer for lokal process
  // TODO: is the image base address necessary
  set_image_base_address(99, (intptr_t)info.dli_fbase);    
  // TODO: keep it simply for now and assume that target function is in main binary
  // If it is necessary to apply different behavior each loaded library has to be covered and analyzed
  free(start_ptr);
  return TARGETDART_SUCCESS;
}