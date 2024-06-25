
#include <__atomic/atomic.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <sys/types.h>
#include "../include/task.h"
#include "../../../src/private.h"

std::vector<intptr_t> *_image_base_addresses;

bool initialized = false;

tdrc init_task_stuctures(){
    _image_base_addresses = new std::vector<intptr_t>(100);
    return TARGETDART_SUCCESS;
}

tdrc finalize_task_structes() {
    delete _image_base_addresses;
    return TARGETDART_SUCCESS;
}

/*
 * Function set_image_base_address
 * Sets base address of particular image index.
 * This is necessary to determine the entry point for functions that represent a target construct
 */
tdrc set_image_base_address(size_t idx_image, intptr_t base_address) {
    if(_image_base_addresses->size() < idx_image+1) {
        _image_base_addresses->resize(idx_image+1);
    }
    // set base address for image (last device wins)
    DP("set_image_base_address (enter) Setting base_address: " DPxMOD " for img: %lu\n", DPxPTR((void*)base_address), idx_image);
    (*_image_base_addresses)[idx_image] = base_address;
    return TARGETDART_SUCCESS;
}

/*
 * Function apply_image_base_address
 * Adds the base address to the address if iBaseAddress == true
 * Else it creates a base address
 */
intptr_t apply_image_base_address(intptr_t base_address, bool isBaseAddress) {
    if (isBaseAddress) {
      return base_address + (*_image_base_addresses)[99];
    }
    return base_address - (*_image_base_addresses)[99];
}

tdrc invoke_task(td_task_t *task, int64_t Device) {
    auto Ret = targetKernelWrapper(task->Loc, Device, task->KernelArgs->NumTeams[0], task->KernelArgs->ThreadLimit[0], (void *) apply_image_base_address(task->host_base_ptr, true), task->KernelArgs);
    if(Ret) {
      return TARGETDART_FAILURE;
    }   
    
    return TARGETDART_SUCCESS;
}

int add_main_ptr(void *main_ptr) {    
    get_base_address(main_ptr);
    return 0;
}

tdrc get_base_address(void *main_ptr) {
  Dl_info info;
  int rc;
  //link_map * map = (link_map *)malloc(1000*sizeof(link_map));
  //void *start_ptr = (void*)map;
  // struct link_map map;
  //rc = dladdr1(main_ptr, &info, (void**)&map, RTLD_DL_LINKMAP);
  rc = dladdr(main_ptr, &info);
  // Store base pointer for lokal process
  // TODO: is the image base address necessary
  set_image_base_address(99, (intptr_t)info.dli_fbase);    
  // TODO: keep it simply for now and assume that target function is in main binary
  // If it is necessary to apply different behavior each loaded library has to be covered and analyzed
  //free(start_ptr);
  initialized = true;
  return TARGETDART_SUCCESS;
}