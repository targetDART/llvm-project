#include "TargetDART.h"
#include "omptarget.h"
#include "device.h"
#include <iostream>
#include "private.h"
#include "mpi.h"
#include <link.h>
#include <queue>
#include <dlfcn.h>

// TODO: implement interface

int test = 0;

// array that holds image base addresses
std::vector<intptr_t> _image_base_addresses;

static inline int DeviceIdGenerator(KernelArgsTy *KernelArgs) {
  return KernelArgs->NumArgs % (omp_get_num_devices() + 1);
}

static inline void printInfo(KernelArgsTy *KernelArgs) {
  for (int i = 0; i < KernelArgs->NumArgs; i++) {  
    std::cout << i << ". pointer " << KernelArgs->ArgPtrs[i] << std::endl;
    std::cout << i << ". size " << KernelArgs->ArgSizes[i] << std::endl;    
  }
  std::cout << "test: " << test << std::endl;
  test++;
  
}


int testFunction(int *argc, char ***argv) {
  int provided = 0;
  MPI_Init_thread(argc, argv, MPI_THREAD_SINGLE, &provided);
  return 3;
}

static inline void printLocInfo(ident_t *Loc) {
  std::cout << "file: " << Loc->reserved_1 << std::endl;
  std::cout << "function: " << Loc->flags << std::endl;
  std::cout << "line: " << Loc->reserved_2 << std::endl;
  std::cout << "col: " << Loc->reserved_3 << std::endl;
  
}

int addTargetDARTTask( ident_t *Loc, int32_t NumTeams,
                        int32_t ThreadLimit, void *HostPtr,
                        KernelArgsTy *KernelArgs, int64_t *DeviceId) 
{
  *DeviceId = DeviceIdGenerator(KernelArgs);
  std::cout << "libomptarget on device: " << *DeviceId << std::endl;
  std::cout << "Host pointer " << HostPtr << std::endl;
  printInfo(KernelArgs);
  printLocInfo(Loc);

  return __tgt_target_kernel(Loc, *DeviceId, NumTeams, ThreadLimit, HostPtr, KernelArgs);
}

// initializes the targetDART lib
int initTargetDART(int *argc, char ***argv, void* main_ptr) {

  get_base_address(main_ptr);

  int provided = 0;
  MPI_Init_thread(argc, argv, MPI_THREAD_SINGLE, &provided);

  if (provided == MPI_THREAD_SINGLE) 
    return TARGETDART_SUCCESS;
  return TARGETDART_FAILURE;
}

// finalizes targetDART lib
int finalizeTargetDART() {
  MPI_Finalize();
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

/**
* 
* 
* 
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