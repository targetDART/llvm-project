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
#include "PluginInterface.h"
#include "omptarget.h"


namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// Forward declarations for all specialized data structures.
struct targetDARTKernelTy;
struct targetDARTDeviceTy;
struct targetDARTPluginTy;
struct targetDARTDeviceImageTy;

/// Class implementing the MPI device images properties.
struct targetDARTDeviceImageTy : public DeviceImageTy {
  /// Create the MPI image with the id and the target image pointer.
  targetDARTDeviceImageTy(int32_t ImageId, GenericDeviceTy &Device,
                   const __tgt_device_image *TgtImage)
      : DeviceImageTy(ImageId, Device, TgtImage), DeviceImageAddrs(getSize()) {}

  llvm::SmallVector<void *> DeviceImageAddrs;
};


} //namespace plugin
} //namespace target
} //namespace omp
} //namespace llvm