#include "TargetDART.h"
#include "omptarget.h"
#include "device.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include "private.h"
#include "mpi.h"
#include <link.h>
#include <queue>
#include <dlfcn.h>
#include <unistd.h>
#include "TD_communication.h"
#include "TD_common.h"

// TODO: implement interface

// Mixed endianness is not supported

static bool __td_initialized = false;

static bool __td_did_initialize_mpi = false;

// communicator for remote task requests
MPI_Comm targetdart_comm;
// communicator for sending back mapped values
MPI_Comm targetdart_comm_mapped;
// communicator for cancelling offloaded tasks
MPI_Comm targetdart_comm_cancel;
// communicator for load information
MPI_Comm targetdart_comm_load;
// communicator for activating replicated tasks
MPI_Comm targetdart_comm_activate;

int targetdart_comm_size;
int targetdart_comm_rank;

MPI_Datatype TD_Kernel_Args;
MPI_Datatype TD_MPI_Task;

// array that holds image base addresses
std::vector<intptr_t> _image_base_addresses;

static inline int DeviceIdGenerator(KernelArgsTy *KernelArgs) {
  return KernelArgs->NumArgs % (omp_get_num_devices() + 1);
}

static inline void printInfo(KernelArgsTy *KernelArgs) {
  for (int i = 0; i < KernelArgs->NumArgs; i++) {  
    std::cout << i << ". pointer " << KernelArgs->ArgPtrs[i] << std::endl;
    std::cout << i << ". base pointer " << KernelArgs->ArgBasePtrs[i] << std::endl;
    std::cout << i << ". size " << KernelArgs->ArgSizes[i] << std::endl;    
  }  
}

static inline void printLocInfo(ident_t *Loc) {
  std::cout << "file: " << Loc->reserved_1 << std::endl;
  std::cout << "function: " << Loc->flags << std::endl;
  std::cout << "line: " << Loc->reserved_2 << std::endl;
  std::cout << "col: " << Loc->reserved_3 << std::endl;  
}

//Adds a task to the TargetDART runtime
int td_add_task( ident_t *Loc, int32_t NumTeams,
                        int32_t ThreadLimit, void *HostPtr,
                        KernelArgsTy *KernelArgs, int64_t *DeviceId) 
{
  *DeviceId = DeviceIdGenerator(KernelArgs);

  td_task_t task;
  task.host_base_ptr = apply_image_base_address((intptr_t) HostPtr, false);
  task.KernelArgs = KernelArgs;
  task.Loc = Loc;
  task.num_teams = NumTeams;
  task.thread_limit = ThreadLimit;
  task.local_proc = targetdart_comm_rank;

  if (targetdart_comm_rank == 0) {
    td_send_task(1, task);
  } else {
    td_receive_task(0, &task);
  }

  return __td_invoke_task(*DeviceId, &task);
}

// initializes the targetDART lib
/*
main_ptr needs to be the same function in the same binary in all processes.
Preferably main().
If MPI is used in the user code, MPI must be initialized before TargetDART.
*/
int initTargetDART(int *argc, char ***argv, void* main_ptr) {
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
      MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
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
  if(err != 0) handle_error_en(err, "MPI_Comm_dup - targetdart_comm");
  MPI_Comm_size(targetdart_comm, &targetdart_comm_size);
  MPI_Comm_rank(targetdart_comm, &targetdart_comm_rank);
  err = MPI_Comm_dup(MPI_COMM_WORLD, &targetdart_comm_mapped);
  if(err != 0) handle_error_en(err, "MPI_Comm_dup - targetdart_comm_mapped");
  err = MPI_Comm_dup(MPI_COMM_WORLD, &targetdart_comm_load);
  if(err != 0) handle_error_en(err, "MPI_Comm_dup - targetdart_comm_load");
  err = MPI_Comm_dup(MPI_COMM_WORLD, &targetdart_comm_cancel);
  if(err != 0) handle_error_en(err, "MPI_Comm_dup - targetdart_comm_cancel");
  err = MPI_Comm_dup(MPI_COMM_WORLD, &targetdart_comm_activate);
  if(err != 0) handle_error_en(err, "MPI_Comm_dup - targetdart_comm_activate");

  /**
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  MPI_Errhandler_set(targetdart_comm, MPI_ERRORS_RETURN);
  MPI_Errhandler_set(targetdart_comm_mapped, MPI_ERRORS_RETURN);
  MPI_Errhandler_set(targetdart_comm_cancel, MPI_ERRORS_RETURN);
  MPI_Errhandler_set(targetdart_comm_load, MPI_ERRORS_RETURN);*/

  // define the base address of the current process
  get_base_address(main_ptr);

  // Init devices during installation
  for (long i = 0; i < omp_get_num_devices(); i++) {
    if (checkDeviceAndCtors(i, nullptr)) {
      DP("Not offloading to device %" PRId64 "\n", DeviceId);
      return TARGETDART_FAILURE;
    }
  }

  return TARGETDART_SUCCESS;
}

/* finalizes targetDART lib, iff TD initilized MPI it also needs to finalize it.
 * Should be one of the last functions in your program.
*/
int finalizeTargetDART() {
  if (__td_did_initialize_mpi) {
    MPI_Finalize();
  }
  return TARGETDART_SUCCESS;
}

// executes the task on the targeted Device 
int __td_invoke_task(int DeviceId, td_task_t* task) {
  return __tgt_target_kernel(task->Loc, DeviceId, task->num_teams, task->thread_limit, (void *) apply_image_base_address(task->host_base_ptr, true), task->KernelArgs);
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