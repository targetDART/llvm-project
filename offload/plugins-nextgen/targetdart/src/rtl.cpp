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

#include "llvm/ADT/SmallVector.h"
#include <iostream>
#include "PluginInterface.h"
#include "omptarget.h"
#include "Utils/ELF.h"


// TODO: Move somewhere else
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

/// Forward declarations for all specialized data structures.
struct targetDARTKernelTy;
struct targetDARTDeviceTy;
struct targetDARTPluginTy;
struct targetDARTDeviceImageTy;
struct targetDARTGlobalHandlerTy;

/// Class implementing the MPI device images properties.
struct targetDARTDeviceImageTy : public DeviceImageTy {
  /// Create the MPI image with the id and the target image pointer.
  targetDARTDeviceImageTy(int32_t ImageId, GenericDeviceTy &Device,
                   const __tgt_device_image *TgtImage)
      : DeviceImageTy(ImageId, Device, TgtImage), DeviceImageAddrs(getSize()) {}

  llvm::SmallVector<void *> DeviceImageAddrs;
};

class targetDARTGlobalHandlerTy final : public GenericGlobalHandlerTy {
public:
  Error getGlobalMetadataFromDevice(GenericDeviceTy &GenericDevice,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
   const char *GlobalName = DeviceGlobal.getName().data();
    targetDARTDeviceImageTy &TDImage = static_cast<targetDARTDeviceImageTy &>(Image);

    if (GlobalName == nullptr) {
      return Plugin::error("Failed to get name for global %p", &DeviceGlobal);
    }

    void *EntryAddress = nullptr;

    __tgt_offload_entry *Begin = TDImage.getTgtImage()->EntriesBegin;
    __tgt_offload_entry *End = TDImage.getTgtImage()->EntriesEnd;

    int I = 0;
    for (auto &Entry = Begin; Entry < End; ++Entry) {
      if (!strcmp(Entry->name, GlobalName)) {
        EntryAddress = TDImage.DeviceImageAddrs[I];
        break;
      }
      I++;
    }

    if (EntryAddress == nullptr) {
      return Plugin::error("Failed to find global %s", GlobalName);
    }

    // Save the pointer to the symbol.
    DeviceGlobal.setPtr(EntryAddress);

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
      return Plugin::error("Invalid function for kernel %s", getName());

    // Save the function pointer.
    //Func = (void (*)())Global.getPtr();
    std::cout << "Function Pointer" << Global.getPtr() << std::endl;

    // TODO: Check which settings are appropriate for the mpi plugin
    // for now we are using the Elf64 plugin configuration
    KernelEnvironment.Configuration.ExecMode = OMP_TGT_EXEC_MODE_GENERIC;
    KernelEnvironment.Configuration.MayUseNestedParallelism = /* Unknown */ 2;
    KernelEnvironment.Configuration.UseGenericStateMachine = /* Unknown */ 2;

    // Set the maximum number of threads to a single.
    MaxNumThreads = 1;
    return Plugin::success();
  }

  /// Launch the kernel on the specific device. The device must be the same
  /// one used to initialize the kernel.
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads, uint64_t NumBlocks, KernelArgsTy &KernelArgs, void *Args, AsyncInfoWrapperTy &AsyncInfoWrapper) const override {
    //TODO
  }

protected:
  /// Prints plugin-specific kernel launch information after generic kernel
  /// launch information
  Error printLaunchInfoDetails(GenericDeviceTy &GenericDevice,
                                       KernelArgsTy &KernelArgs,
                                       uint32_t NumThreads,
                                       uint64_t NumBlocks) const override {
  //TODO
  }
};

/// Class implementing common functionalities of offload devices. Each plugin
/// should define the specific device class, derive from this generic one, and
/// implement the necessary virtual function members.
struct targetDARTDeviceTy : public GenericDeviceTy {
  /// Construct a device with its device id within the plugin, the number of
  /// devices in the plugin and the grid values for that kind of device.
  targetDARTDeviceTy(GenericPluginTy &Plugin, int32_t DeviceId, int32_t NumDevices)
      : GenericDeviceTy(Plugin, DeviceId, NumDevices, targetDARTGridValues) {}

  /// Set the context of the device if needed, before calling device-specific
  /// functions. Plugins may implement this function as a no-op if not needed.
  Error setContext() override{
    return Plugin::success();
  }

  /// Initialize the device. After this call, the device should be already
  /// working and ready to accept queries or modifications.
  Error initImpl(GenericPluginTy &Plugin) override{
    //TODO: Inititialize device specific queues
    std::cout << "init targetDART device" << std::endl;

    return Plugin::success();
  }

  /// Deinitialize the device and free all its resources. After this call, the
  /// device is no longer considered ready, so no queries or modifications are
  /// allowed.
  Error deinitImpl() override{
    //TODO
    std::cout << "finalize targetDART device" << std::endl;
    return Plugin::success();
  }

  /// Load the binary image into the device and return the target table.
  Expected<DeviceImageTy *> loadBinaryImpl(const __tgt_device_image *TgtImage, int32_t ImageId)override {
    // Allocate and initialize the image object.
    targetDARTDeviceImageTy *Image = Plugin.allocate<targetDARTDeviceImageTy>();
    new (Image) targetDARTDeviceImageTy(ImageId, *this, TgtImage);

    std::cout << "load targetDART image" << std::endl;

    return Image;
  }

  /// Synchronize the current thread with the pending operations on the
  /// __tgt_async_info structure.
  Error synchronizeImpl(__tgt_async_info &AsyncInfo) override{
    //TODO
  }

  /// Query for the completion of the pending operations on the __tgt_async_info
  /// structure in a non-blocking manner.
  Error queryAsyncImpl(__tgt_async_info &AsyncInfo) override{
    //TODO
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
    //TODO
  }

  /// De-allocates device memory and unmaps the virtual address \p VAddr
  Error memoryVAUnMap(void *VAddr, size_t Size) override{
    //TODO
  }

  /// Lock the host buffer \p HstPtr with \p Size bytes with the vendor-specific
  /// API and return the device accessible pointer.
  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override{
    //TODO
  }

  /// Unlock a previously locked host buffer starting at \p HstPtr.
  Error dataUnlockImpl(void *HstPtr) override{
    //TODO
  }

  /// Check whether the host buffer with address \p HstPtr is pinned by the
  /// underlying vendor-specific runtime (if any). Retrieve the host pointer,
  /// the device accessible pointer and the size of the original pinned buffer.
  Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr,
                                         void *&BaseDevAccessiblePtr,
                                         size_t &BaseSize) const override{
    return false;
  }

  /// Submit data to the device (host to device transfer).
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                               AsyncInfoWrapperTy &AsyncInfoWrapper) override{
    DP("Data management not supported for targetDART devices");
    return Plugin::success();
  }

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                                 AsyncInfoWrapperTy &AsyncInfoWrapper) override{
    DP("Data management not supported for targetDART devices");
    return Plugin::success();
  }

  /// Exchange data between devices (device to device transfer). Calling this
  /// function is only valid if GenericPlugin::isDataExchangable() passing the
  /// two devices returns true.
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev,
                                 void *DstPtr, int64_t Size,
                                 AsyncInfoWrapperTy &AsyncInfoWrapper) override{
    DP("Data management not supported for targetDART devices");
    return Plugin::success();
  }

  /// Initialize a __tgt_async_info structure. Related to interop features.
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override{
    return Plugin::error("initAsyncInfoImpl not supported");
  }

  /// Initialize a __tgt_device_info structure. Related to interop features.
  Error initDeviceInfoImpl(__tgt_device_info *DeviceInfo) override{
    return Plugin::error("initDeviceInfoImpl not supported");
  }

    /// Allocate memory on the device or related to the device.
  void *allocate(size_t Size, void *, TargetAllocTy Kind) override {
    REPORT("Offloading to abstract targetDART devices is not supported\n");
    return nullptr;
  }

  /// Deallocate memory on the device or related to the device.
  int free(void *TgtPtr, TargetAllocTy Kind) override {
    REPORT("Offloading to abstract targetDART devices is not supported\n");
    return OFFLOAD_FAIL;
  }

  /// Create an event.
  Error createEventImpl(void **EventPtrStorage) override{
    //TODO
  }

  /// Destroy an event.
  Error destroyEventImpl(void *EventPtr) override{
    //TODO
  }

  /// Start the recording of the event.
  Error recordEventImpl(void *EventPtr,
                                AsyncInfoWrapperTy &AsyncInfoWrapper) override{
    //TODO
  }

  /// Wait for an event to finish. Notice this wait is asynchronous if the
  /// __tgt_async_info is not nullptr.
  virtual Error waitEventImpl(void *EventPtr,
                              AsyncInfoWrapperTy &AsyncInfoWrapper)override{
    //TODO
  }

  /// Synchronize the current thread with the event.
  Error syncEventImpl(void *EventPtr) override{
    //TODO
  }

  /// Print information about the device.
  Error obtainInfoImpl(InfoQueueTy &Info) override{
    //TODO
  }

  /// Allocate and construct a kernel object.
  Expected<GenericKernelTy &> constructKernel(const char *Name) override{
    //TODO
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

  /// Initialize the plugin and return the number of available devices.
  Expected<int32_t> initImpl() override{
    //TODO Initialize MPI Sessions and all necessary queues
    std::cout << "init targetDART" << std::endl;
    return 3;
  }

  /// Deinitialize the plugin and release the resources.
  Error deinitImpl() override{
    //TODO cleanup
    std::cout << "finalize targetDART" << std::endl;
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
  Triple::ArchType getTripleArch() const override{
    std::cout << "get target Arch" << std::endl;
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
};

extern "C" {
llvm::omp::target::plugin::GenericPluginTy *createPlugin_targetdart() {
  return new llvm::omp::target::plugin::targetDARTPluginTy();
}
}

} //namespace plugin
} //namespace target
} //namespace omp
} //namespace llvm