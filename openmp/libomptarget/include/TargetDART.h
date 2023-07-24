#ifndef _OMPTARGET_TARGETDART_H
#define _OMPTARGET_TARGETDART_H

//TODO: define interface

#include <cstdint>
#include "mpi.h"
#include "omptarget.h"
#include "device.h"
#include "TD_common.h"

// Outsources a target construct to the targetDART runtime
int td_add_task( ident_t *Loc, int32_t NumTeams,
                        int32_t ThreadLimit, void *HostPtr,
                        KernelArgsTy *KernelArgs, int64_t *DeviceId);

tdrc get_base_address(void * main_ptr);

int32_t set_image_base_address(int idx_image, intptr_t base_address);

tdrc declare_KernelArgs_type();
tdrc declare_task_type();

int __td_invoke_task(int DeviceId, td_task_t* task);

intptr_t apply_image_base_address(intptr_t base_address, bool isBaseAddress);


extern "C" int initTargetDART(void* main_ptr);

extern "C" int finalizeTargetDART();

#endif // _OMPTARGET_TARGETDART_H