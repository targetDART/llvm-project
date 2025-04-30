#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include "../include/task.h"
#include "../../../src/private.h"
#include "PluginManager.h"
#include "device.h"
#include "omptarget.h"

#include "llvm/ADT/SmallVector.h"

#include <execinfo.h>

std::vector<intptr_t> *_image_base_addresses;

bool initialized = false;

tdrc init_task_structures(){
    char **strings; //every entry is an entire line of the stacktrace
    size_t i, size;
    const int max_size = 128;
    void *array[max_size];
    size = backtrace(array, max_size);
    strings = backtrace_symbols(array, size); //returns the entire stack
    

    int target_idx = size-1;
    std::string s(strings[target_idx]);
    int idx = s.find("[")+1+2; // +1 to skip [, +2 to skip 0x, so first number, in char* takes 12 excluding 0x...
    std::string main_nmb_as_str = s.substr(idx, 12);
    long main_nmb = std::stol(main_nmb_as_str, 0, 16);

    _image_base_addresses = new std::vector<intptr_t>(100);
    add_main_ptr((void*) main_nmb);
    
    return TARGETDART_SUCCESS;
}

tdrc finalize_task_structes() {
    delete _image_base_addresses;
    return TARGETDART_SUCCESS;
}

/*
* Frees all memory assoziated with the task.
* This should onle be used to free data structures explicitly allocated on remote nodes
*/
tdrc delete_task(td_task_t *task, bool local) {
    if (!local) {
        DP("num args: %d\n",task->KernelArgs->NumArgs);
        for (uint32_t i = 0; i < task->KernelArgs->NumArgs - 1; i++) {
            const auto notLiteral = (task->KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_LITERAL) == 0;
            const auto notPrivate = (task->KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_PRIVATE) == 0;
            if (notLiteral && notPrivate) {
                // zero-size data should have a zero pointer, or hidden data near the base arg
                DP("(%ld%ld) Entry %d: delete data at " DPxMOD " \n", task->uid.rank, task->uid.id, i, DPxPTR(task->KernelArgs->ArgBasePtrs[i]));
                std::free(task->KernelArgs->ArgBasePtrs[i]);
            }
        }
        delete task->Loc;
        delete task->KernelArgs->ArgBasePtrs;
        delete task->KernelArgs->ArgMappers;
        delete task->KernelArgs->ArgNames;
        delete task->KernelArgs->ArgTypes;
        delete task->KernelArgs->ArgSizes;
    }
    delete task->KernelArgs;
    delete task;
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
    DP("Base address: " DPxMOD " Image start: " DPxMOD "\n", DPxPTR(base_address), DPxPTR((*_image_base_addresses)[99]));
    if (isBaseAddress) { 
      return base_address + (*_image_base_addresses)[99];
    }
    return base_address - (*_image_base_addresses)[99];
}

int add_main_ptr(void *main_ptr) {
    get_base_address(main_ptr);
    return 0;
}

tdrc get_base_address(void *main_ptr) {
  Dl_info info;
  //int rc;
  //link_map * map = (link_map *)malloc(1000*sizeof(link_map));
  //void *start_ptr = (void*)map;
  // struct link_map map;
  //rc = dladdr1(main_ptr, &info, (void**)&map, RTLD_DL_LINKMAP);
  dladdr(main_ptr, &info);
  // Store base pointer for lokal process
  // TODO: is the image base address necessary
  set_image_base_address(99, (intptr_t)info.dli_fbase);    
  // TODO: keep it simply for now and assume that target function is in main binary
  // If it is necessary to apply different behavior each loaded library has to be covered and analyzed
  //free(start_ptr);
  initialized = true;
  return TARGETDART_SUCCESS;
}


#ifdef TD_TRACE
  int trace_rank = -1; // this MPI rank
  FILE *trace_file = NULL; // if tracing is enabled, this will provide a handle for the tracefile (for this MPI rank)
  unsigned long start_of_trace = 0ul; // initial time stamp in microseconds => trace time-stamps are relative to this
#endif // TD_TRACE

