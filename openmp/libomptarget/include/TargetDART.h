#ifndef _OMPTARGET_TARGETDART_H
#define _OMPTARGET_TARGETDART_H

//TODO: define interface

#include <cstdint>
#include "omptarget.h"
#include "device.h"

enum tdrc {TARGETDART_FAILURE, TARGETDART_SUCCESS};

#ifndef DBP
#ifdef CHAM_DEBUG
#define DBP( ... ) { RELP(__VA_ARGS__); }
#else
#define DBP( ... ) { }
#endif
#endif

// Outsources a target construct to the targetDART runtime
int addTargetDARTTask( ident_t *Loc, int32_t NumTeams,
                        int32_t ThreadLimit, void *HostPtr,
                        KernelArgsTy *KernelArgs, int64_t *DeviceId);

tdrc get_base_address(void * main_ptr);

int32_t set_image_base_address(int idx_image, intptr_t base_address);


extern "C" int initTargetDART(int *argc, char ***argv, void* main_ptr);

extern "C" int finalizeTargetDART();

extern "C" int testFunction(int *, char ***);

#endif // _OMPTARGET_TARGETDART_H