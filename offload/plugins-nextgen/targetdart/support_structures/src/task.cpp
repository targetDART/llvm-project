
#include <cstdint>
#include <cstdlib>
#include <sys/types.h>
#include "../include/task.h"
#include "../../../src/private.h"

std::vector<intptr_t> *_image_base_addresses;

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
tdrc set_image_base_address(ulong idx_image, intptr_t base_address) {
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

