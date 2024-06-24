//===----RTLs/targetDART/src/rtl.cpp - Target RTLs Implementation - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL NextGen for targetDART runtime scheduling
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <iostream>
#include <ffi.h>
#include "PluginInterface.h"
#include "Shared/Environment.h"
#include "omptarget.h"
#include "PluginManager.h"
#include "../../../src/private.h"


#include "../support_structures/include/task.h"
#include "../support_structures/include/threading.h"
#include "../support_structures/include/communication.h"
#include "../support_structures/include/scheduling.h"



#include "Utils/ELF.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DynamicLibrary.h"


#ifndef TARGET_NAME
#define TARGET_NAME TARGETDART
#endif
#ifndef DEBUG_PREFIX
#define DEBUG_PREFIX "Target " GETNAME(TARGET_NAME) " RTL"
#endif

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

using llvm::sys::DynamicLibrary;

/// Forward declarations for all specialized data structures.
struct targetDARTKernelTy;
struct targetDARTDeviceTy;
struct targetDARTPluginTy;
struct targetDARTDeviceImageTy;
struct targetDARTGlobalHandlerTy;

struct targetDARTDeviceImageTy : public DeviceImageTy {
  /// Create the targetDART image with the id and the target image pointer.
  targetDARTDeviceImageTy(int32_t ImageId, GenericDeviceTy &Device,
                        const __tgt_device_image *TgtImage)
      : DeviceImageTy(ImageId, Device, TgtImage), DynLib() {}

  /// Getter and setter for the dynamic library.
  DynamicLibrary &getDynamicLibrary() { return DynLib; }
  void setDynamicLibrary(const DynamicLibrary &Lib) { DynLib = Lib; }

private:
  /// The dynamic library that loaded the image.
  DynamicLibrary DynLib;
};

struct targetDARTGlobalHandlerTy final : public GenericGlobalHandlerTy {
public:
  Error getGlobalMetadataFromDevice(GenericDeviceTy &GenericDevice,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    targetDARTDeviceImageTy &TDImage = static_cast<targetDARTDeviceImageTy &>(Image);

    const char *GlobalName = DeviceGlobal.getName().data();

    // Get dynamic library that has loaded the device image.
    DynamicLibrary &DynLib = TDImage.getDynamicLibrary();

    // Get the address of the symbol.
    void *Addr = DynLib.getAddressOfSymbol(GlobalName);
    if (Addr == nullptr) {
      return Plugin::error("Failed to load global '%s'\n", GlobalName);
    }

    // Save the pointer to the symbol.
    DeviceGlobal.setPtr(Addr);

    return Plugin::success();
  }
};

/// Class implementing common functionalities of offload kernels. Each plugin
/// should define the specific kernel class, derive from this generic one, and
/// implement the necessary virtual function members.
struct targetDARTKernelTy: public GenericKernelTy {
  /// Construct a kernel with a name and a execution mode.
  targetDARTKernelTy(const char *Name) : GenericKernelTy(Name) {}

  ~targetDARTKernelTy() {}

  /// Initialize the kernel object from a specific device.
  Error initImpl(GenericDeviceTy &GenericDevice, DeviceImageTy &Image) override {
    // Functions have zero size.
    GlobalTy Global(getName(), 0);

    // Get the metadata (address) of the kernel function.
    GenericGlobalHandlerTy &GHandler = GenericDevice.Plugin.getGlobalHandler();
    if (auto Err = GHandler.getGlobalMetadataFromDevice(GenericDevice, Image, Global))
      return Err;

    // Check that the function pointer is valid.
    if (!Global.getPtr())
      return Plugin::error("Invalid function for kernel %s\n", getName());

    // Save the function pointer.
    CPUFunc = (void (*)())Global.getPtr();
    DP("Function for kernel %s\n", getName());
    std::cout << "Function Pointer: " << Global.getPtr() << std::endl;

    // TODO: Check which settings are appropriate for the mpi plugin
    // for now we are using the Elf64 plugin configuration
    KernelEnvironment.Configuration.ExecMode = OMP_TGT_EXEC_MODE_GENERIC;
    KernelEnvironment.Configuration.MayUseNestedParallelism = /* Unknown */ 2;
    KernelEnvironment.Configuration.UseGenericStateMachine = /* Unknown */ 2;

    // Set the maximum number of threads to a single.
    MaxNumThreads = 1;
    return Plugin::success();
  }

  Error addHostInfo(ident_t *HostLoc, void *HostEntryPtr) override {
    HostPtr = HostEntryPtr;
    Loc = HostLoc;
    return Plugin::success();
  }

  /// Launch the kernel on the specific device. The device must be the same
  /// one used to initialize the kernel.
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads, uint64_t NumBlocks, KernelArgsTy &KernelArgs, void *Args, AsyncInfoWrapperTy &AsyncInfoWrapper) const override {

    DP("targetDART Kernel launch\n");

    KernelArgs.NumArgs--;



    // TODO: add task to queue
    /* auto Ret = targetKernelWrapper(Loc, 0, 1, 1, HostPtr, &KernelArgs);
    if(Ret) {
      return Plugin::error("targetDART offloading failed\n");
    }   */ 
    
    return Plugin::success();
  }

protected:
  /// Prints plugin-specific kernel launch information after generic kernel
  /// launch information
  Error printLaunchInfoDetails(GenericDeviceTy &GenericDevice,
                                       KernelArgsTy &KernelArgs,
                                       uint32_t NumThreads,
                                       uint64_t NumBlocks) const override {
  //TODO Extend for targetDART
  return Plugin::success();
  }

private: 

  void (*CPUFunc)(void);
  void *HostPtr;
  ident_t *Loc;

};

/// Class implementing common functionalities of offload devices. Each plugin
/// should define the specific device class, derive from this generic one, and
/// implement the necessary virtual function members.
struct targetDARTDeviceTy : public GenericDeviceTy {
  /// Construct a device with its device id within the plugin, the number of
  /// devices in the plugin and the grid values for that kind of device.
  targetDARTDeviceTy(GenericPluginTy &Plugin, int32_t DeviceId, int32_t NumDevices)
      : GenericDeviceTy(Plugin, DeviceId, NumDevices, targetDARTGridValues) {
        deviceID = DeviceId;
      }

  /// Set the context of the device if needed, before calling device-specific
  /// functions. Plugins may implement this function as a no-op if not needed.
  Error setContext() override {
    return Plugin::success();
  }

  /// Initialize the device. After this call, the device should be already
  /// working and ready to accept queries or modifications.
  Error initImpl(GenericPluginTy &Plugin) override {
    //TODO: setup Queues
    return Plugin::success();
  }

  /// Deinitialize the device and free all its resources. After this call, the
  /// device is no longer considered ready, so no queries or modifications are
  /// allowed.
  Error deinitImpl() override {
    //TODO: cleanup queues
    return Plugin::success();
  }

  /// Load the binary image into the device and return the target table.
  Expected<DeviceImageTy *> loadBinaryImpl(const __tgt_device_image *TgtImage, int32_t ImageId) override {
    // Allocate and initialize the image object.
    targetDARTDeviceImageTy *Image = Plugin.allocate<targetDARTDeviceImageTy>();
    new (Image) targetDARTDeviceImageTy(ImageId, *this, TgtImage);

    // Create a temporary file.
    char TmpFileName[] = "/tmp/tmpfile_XXXXXX";
    int TmpFileFd = mkstemp(TmpFileName);
    if (TmpFileFd == -1)
      return Plugin::error("Failed to create tmpfile for loading target image\n");

    // Open the temporary file.
    FILE *TmpFile = fdopen(TmpFileFd, "wb\n");
    if (!TmpFile)
      return Plugin::error("Failed to open tmpfile %s for loading target image\n",
                           TmpFileName);

    // Write the image into the temporary file.
    size_t Written = fwrite(Image->getStart(), Image->getSize(), 1, TmpFile);
    if (Written != 1)
      return Plugin::error("Failed to write target image to tmpfile %s\n",
                           TmpFileName);

    // Close the temporary file.
    int Ret = fclose(TmpFile);
    if (Ret)
      return Plugin::error("Failed to close tmpfile %s with the target image\n",
                           TmpFileName);

    // Load the temporary file as a dynamic library.
    std::string ErrMsg;
    DynamicLibrary DynLib =
        DynamicLibrary::getPermanentLibrary(TmpFileName, &ErrMsg);

    // Check if the loaded library is valid.
    if (!DynLib.isValid())
      return Plugin::error("Failed to load target image: %s\n", ErrMsg.c_str());

    // Save a reference of the image's dynamic library.
    Image->setDynamicLibrary(DynLib);

    return Image;
  }
  
  /// This plugin should not setup the device environment or memory pool.
  virtual bool shouldSetupDeviceEnvironment() const override { return false; };
  virtual bool shouldSetupDeviceMemoryPool() const override { return false; };

  /// Synchronize the current thread with the pending operations on the
  /// __tgt_async_info structure.
  Error synchronizeImpl(__tgt_async_info &AsyncInfo) override{
    //TODO
  }

  /// Query for the completion of the pending operations on the __tgt_async_info
  /// structure in a non-blocking manner.
  Error queryAsyncImpl(__tgt_async_info &AsyncInfo) override{
    //TODO: 
  }

  /// Get the total device memory size
  Error getDeviceMemorySize(uint64_t &DSize)override{
    DSize = 0;
    return Plugin::success();
  }

  /// Allocates \p RSize bytes (rounded up to page size) and hints the driver to
  /// map it to \p VAddr. The obtained address is stored in \p Addr. At return
  /// \p RSize contains the actual size which can be equal or larger than the
  /// requested size.
  Error memoryVAMap(void **Addr, void *VAddr, size_t *RSize)override {
    DP("targetDART devices cannot perform memory management\n");    
    return Plugin::success();
  }

  /// De-allocates device memory and unmaps the virtual address \p VAddr
  Error memoryVAUnMap(void *VAddr, size_t Size) override{    
    DP("targetDART devices cannot perform memory management\n");    
    return Plugin::success();
  }

  /// Lock the host buffer \p HstPtr with \p Size bytes with the vendor-specific
  /// API and return the device accessible pointer.
  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override{     
    DP("targetDART devices cannot perform memory management providing nullptr\n");    
    return nullptr;
  }

  /// Unlock a previously locked host buffer starting at \p HstPtr.
  Error dataUnlockImpl(void *HstPtr) override{ 
    DP("targetDART devices cannot perform memory management\n");    
    return Plugin::success();
  }

  /// Check whether the host buffer with address \p HstPtr is pinned by the
  /// underlying vendor-specific runtime (if any). Retrieve the host pointer,
  /// the device accessible pointer and the size of the original pinned buffer.
  Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr, void *&BaseDevAccessiblePtr, size_t &BaseSize) const override{
    return false;
  }

  /// Submit data to the device (host to device transfer).
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size, AsyncInfoWrapperTy &AsyncInfoWrapper) override{
    DP("targetDART submit Data, not supported\n");
    return Plugin::success();
  }

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size, AsyncInfoWrapperTy &AsyncInfoWrapper) override{    
    DP("targetDART retrieve Data, not supported\n");
    return Plugin::success();
  }

  /// Exchange data between devices (device to device transfer). Calling this
  /// function is only valid if GenericPlugin::isDataExchangable() passing the
  /// two devices returns true.
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev, void *DstPtr, int64_t Size, AsyncInfoWrapperTy &AsyncInfoWrapper) override{
    DP("Data management not supported for targetDART devices, exchange\n");
    return Plugin::error("Data management not supported for targetDART devices, exchange\n");
  }

  /// Initialize a __tgt_async_info structure. Related to interop features.
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::error("initAsyncInfoImpl not supported\n");
  }

  /// Initialize a __tgt_device_info structure. Related to interop features.
  Error initDeviceInfoImpl(__tgt_device_info *DeviceInfo) override{
    return Plugin::error("initDeviceInfoImpl not supported\n");
  }

  /// Allocate memory. Use std::malloc in all cases.
  void *allocate(size_t Size, void *, TargetAllocTy Kind) override {

    DP("targetDART alloc Data, not supported\n");
    return nullptr;
  }

  /// Free the memory. Use std::free in all cases.
  int free(void *TgtPtr, TargetAllocTy Kind) override {
    DP("targetDART free Data, not supported\n");
    return OFFLOAD_SUCCESS;
  }

  /// Create an event.
  Error createEventImpl(void **EventPtrStorage) override{
    *EventPtrStorage = nullptr;
    return Plugin::success();
  }

  /// Destroy an event.
  Error destroyEventImpl(void *EventPtr) override{
    return Plugin::success();
  }

  /// Start the recording of the event.
  Error recordEventImpl(void *EventPtr, AsyncInfoWrapperTy &AsyncInfoWrapper) override{
    return Plugin::success();
  }

  /// Wait for an event to finish. Notice this wait is asynchronous if the
  /// __tgt_async_info is not nullptr.
  virtual Error waitEventImpl(void *EventPtr, AsyncInfoWrapperTy &AsyncInfoWrapper)override{
    return Plugin::success();
  }

  /// Synchronize the current thread with the event.
  Error syncEventImpl(void *EventPtr) override{
    return Plugin::success();
  }

  /// Print information about the device.
  Error obtainInfoImpl(InfoQueueTy &Info) override{
    // TODO: Add more information about the device.
    Info.add("targetDART plugin\n");
    Info.add("targetDART OpenMP Device Number\n", DeviceId);

    return Plugin::success();
  }

  /// Allocate and construct a kernel object.
  Expected<GenericKernelTy &> constructKernel(const char *Name) override {
    // Allocate and construct the kernel.
    targetDARTKernelTy *TDKernel = Plugin.allocate<targetDARTKernelTy>();

    if (!TDKernel)
      return Plugin::error("Failed to allocate memory for targetDART kernel\n");

    new (TDKernel) targetDARTKernelTy(Name);

    return *TDKernel;
  }

  Error getDeviceStackSize(uint64_t &V)  override{
    V = 0;
    return Plugin::success();
  }

  private: 

  Error setDeviceStackSize(uint64_t V)  override{
    return Plugin::success();
  }
  Error getDeviceHeapSize(uint64_t &V)  override{
    V = 0;
    return Plugin::success();
  }
  Error setDeviceHeapSize(uint64_t V) override{
    return Plugin::success();
  }

  int32_t deviceID;

  /// Grid values for the targetDART plugin.
  static constexpr GV targetDARTGridValues = {
      1, // GV_Slot_Size
      1, // GV_Warp_Size
      1, // GV_Max_Teams
      1, // GV_Default_Num_Teams
      1, // GV_SimpleBufferSize
      1, // GV_Max_WG_Size
      1, // GV_Default_WG_Size
  };

  //TODO: add queue per device
  //TODO: add remote queue
  //TODO: add replica queue
};

/// Class implementing common functionalities of offload plugins. Each plugin
/// should define the specific plugin class, derive from this generic one, and
/// implement the necessary virtual function members.
struct targetDARTPluginTy : public GenericPluginTy {

  /// Construct a plugin instance.
  targetDARTPluginTy() : GenericPluginTy(getTripleArch()) {
    std::cout << "create targetDART plugin" << std::endl;
  }

  ~targetDARTPluginTy() {}

  /// This class should not be copied.
  targetDARTPluginTy(const targetDARTPluginTy &) = delete;
  targetDARTPluginTy(targetDARTPluginTy &&) = delete;

  bool requiresDataManagement() override {
    return false;
  }

  /// Initialize the plugin and return the number of available devices.
  Expected<int32_t> initImpl() override{
    DP("init targetDART\n");

    DP("detected prior devices: %d\n", getDeviceIdStartIndex());

    //TODO: get size from other plugins
    int32_t external_devices = 4;

    td_comm = new TD_Communicator();
    td_sched = new TD_Scheduling_Manager(external_devices, td_comm);
    td_thread = new TD_Thread_Manager(external_devices, td_comm, td_sched);
    return td_sched->public_device_count();
  }

  /// Deinitialize the plugin and release the resources.
  Error deinitImpl() override{
    //TODO cleanup
    std::cout << "finalize targetDART" << std::endl;

    // TODO: fix me
    //delete td_thread;
    //delete td_sched;
    delete td_comm;
    return Plugin::success();
  }

  /// Create a new device for the underlying plugin.
  GenericDeviceTy *createDevice(GenericPluginTy &Plugin,
                                        int32_t DeviceID,
                                        int32_t NumDevices) override{
    std::cout << "create targetDART device: " << DeviceID << std::endl;
    return new targetDARTDeviceTy(Plugin, DeviceID, NumDevices);
  }

  /// Create a new global handler for the underlying plugin.
  GenericGlobalHandlerTy *createGlobalHandler() override{
    return new targetDARTGlobalHandlerTy();
  }

  /// Get the ELF code to recognize the binary image of this plugin.
  uint16_t getMagicElfBits() const override {
    return utils::elf::getTargetMachine();
  }

  /// Get the target triple of this plugin.
  Triple::ArchType getTripleArch() const override {
    #if defined(__x86_64__)
      return llvm::Triple::x86_64;
    #elif defined(__s390x__)
      return llvm::Triple::systemz;
    #elif defined(__aarch64__)
    #ifdef LITTLEENDIAN_CPU
      return llvm::Triple::aarch64;
    #else
      return llvm::Triple::aarch64_be;
    #endif
    #elif defined(__powerpc64__)
    #ifdef LITTLEENDIAN_CPU
      return llvm::Triple::ppc64le;
    #else
      return llvm::Triple::ppc64;
    #endif
    #else
      return llvm::Triple::UnknownArch;
    #endif
  }

  /// Get the constant name identifier for this plugin.
  const char *getName() const override{
    return GETNAME(TARGET_NAME);
  }

  /// Indicate if an image is compatible with the plugin devices. Notice that
  /// this function may be called before actually initializing the devices. So
  /// we could not move this function into GenericDeviceTy.
  Expected<bool> isELFCompatible(StringRef Image) const override{
    return true;
  }

  private:
    TD_Communicator *td_comm;
    TD_Scheduling_Manager *td_sched;
    TD_Thread_Manager *td_thread;

    int getPhysicalDevices() {
      
    }
};

template <typename... ArgsTy>
static Error Plugin::check(int32_t Code, const char *ErrMsg, ArgsTy... Args) {
  if (Code == 0)
    return Error::success();

  return createStringError<ArgsTy..., const char *>(
      inconvertibleErrorCode(), ErrMsg, Args..., std::to_string(Code).data());
}

extern "C" {
llvm::omp::target::plugin::GenericPluginTy *createPlugin_targetdart() {
  return new llvm::omp::target::plugin::targetDARTPluginTy();
}
}

} //namespace plugin
} //namespace target
} //namespace omp
} //namespace llvm