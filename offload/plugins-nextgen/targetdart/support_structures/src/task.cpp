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

/// Find the table information in the map or look it up in the translation
/// tables.
TableMap *getTableMap(void *HostPtr) {
  std::lock_guard<std::mutex> TblMapLock(PM->TblMapMtx);
  HostPtrToTableMapTy::iterator TableMapIt =
      PM->HostPtrToTableMap.find(HostPtr);

  if (TableMapIt != PM->HostPtrToTableMap.end())
    return &TableMapIt->second;

  // We don't have a map. So search all the registered libraries.
  TableMap *TM = nullptr;
  std::lock_guard<std::mutex> TrlTblLock(PM->TrlTblMtx);
  for (HostEntriesBeginToTransTableTy::iterator Itr =
           PM->HostEntriesBeginToTransTable.begin();
       Itr != PM->HostEntriesBeginToTransTable.end(); ++Itr) {
    // get the translation table (which contains all the good info).
    TranslationTable *TransTable = &Itr->second;
    // iterate over all the host table entries to see if we can locate the
    // host_ptr.
    __tgt_offload_entry *Cur = TransTable->HostTable.EntriesBegin;
    for (uint32_t I = 0; Cur < TransTable->HostTable.EntriesEnd; ++Cur, ++I) {
      if (Cur->addr != HostPtr)
        continue;
      // we got a match, now fill the HostPtrToTableMap so that we
      // may avoid this search next time.
      TM = &(PM->HostPtrToTableMap)[HostPtr];
      TM->Table = TransTable;
      TM->Index = I;
      return TM;
    }
  }

  return nullptr;
}

tdrc invoke_task(td_task_t *task, int64_t Device) {

    if (Device == -1 || Device == omp_get_initial_device()) {
        auto Ret = targetKernelWrapper(task->Loc, Device, task->KernelArgs->NumTeams[0], task->KernelArgs->ThreadLimit[0], (void *) apply_image_base_address(task->host_base_ptr, true), task->KernelArgs);
        if(Ret) {
        return TARGETDART_FAILURE;
        }   
    
        return TARGETDART_SUCCESS;
    }

    // get physical device
    auto DeviceOrErr = PM->getDevice(Device);
    if (!DeviceOrErr)
        FATAL_MESSAGE(Device, "%s", toString(DeviceOrErr.takeError()).c_str());
    
    // create new async info
    AsyncInfoTy TargetAsyncInfo(*DeviceOrErr);

    void *devicePtrs[task->KernelArgs->NumArgs];

    DP("Allocating %d arguments for task (%ld%ld)\n", task->KernelArgs->NumArgs, task->uid.rank, task->uid.id);

    // Allocate data on the device and transfer it from host to device if necessary
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs; i++) {
        // non blocking alloc is not non blocking but rather uses the non-blocking calls internally
        devicePtrs[i] = DeviceOrErr->allocData(task->KernelArgs->ArgSizes[i], task->KernelArgs->ArgPtrs[i], TARGET_ALLOC_DEVICE_NON_BLOCKING);
        const bool hasFlagTo = task->KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_TO;
        if (hasFlagTo) {        
            DeviceOrErr->submitData(devicePtrs[i], task->KernelArgs->ArgPtrs[i], task->KernelArgs->ArgSizes[i], TargetAsyncInfo);
        }
    }

    if (checkDeviceAndCtors(Device, task->Loc)) {
        DP("Not offloading to device %" PRId64 "\n", Device);
        return TARGETDART_FAILURE;
    }

    // generate a Kernel
    llvm::SmallVector<ptrdiff_t> offsets(task->KernelArgs->NumArgs, 0);

    void *HostPtr = (void *) apply_image_base_address(task->host_base_ptr, true);

    TableMap *TM = getTableMap(HostPtr);
    // No map for this host pointer found!
    if (!TM) {
        REPORT("Host ptr " DPxMOD " does not have a matching target pointer.\n",
            DPxPTR(HostPtr));
        return TARGETDART_FAILURE;
    }

    // get target table.
    __tgt_target_table *TargetTable = nullptr;
    {
        std::lock_guard<std::mutex> TrlTblLock(PM->TrlTblMtx);
        assert(TM->Table->TargetsTable.size() > (size_t)Device &&
                "Not expecting a device ID outside the table's bounds!");
        TargetTable = TM->Table->TargetsTable[Device];
    }
    assert(TargetTable && "Global data has not been mapped\n");

    // Launch device execution.
    void *TgtEntryPtr = TargetTable->EntriesBegin[TM->Index].addr;
    DP("Launching target execution %s with pointer " DPxMOD " (index=%d).\n",
        TargetTable->EntriesBegin[TM->Index].name, DPxPTR(TgtEntryPtr), TM->Index);
        

    DP("Running kernel for task (%ld%ld)\n", task->uid.rank, task->uid.id);
    auto Err = DeviceOrErr->launchKernel(TgtEntryPtr, devicePtrs, offsets.data(), *task->KernelArgs,
                               TargetAsyncInfo, task->Loc, HostPtr);

    // Deallocate data on the device and transfer it from device to host if necessary
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs - 1; i++) {
        const bool hasFlagFrom = task->KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_FROM;
        if (hasFlagFrom) {        
            DeviceOrErr->retrieveData(task->KernelArgs->ArgPtrs[i], devicePtrs[i], task->KernelArgs->ArgSizes[i], TargetAsyncInfo);
        }
    }

    DeviceOrErr->synchronize(TargetAsyncInfo);
    // Deallocate data on the device and transfer it from device to host if necessary
    for (uint32_t i = 0; i < task->KernelArgs->NumArgs - 1; i++) {
        DeviceOrErr->deleteData(devicePtrs[i], TARGET_ALLOC_DEVICE_NON_BLOCKING);
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