#include "TargetDART.h"
#include "omptarget.h"
#include "device.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include "private.h"
#include "mpi.h"
#include <link.h>
#include <queue>
#include <dlfcn.h>
#include <unistd.h>
#include "TD_common.h"


tdrc declare_KernelArgs_type() {
  const int nitems = 3;
  int blocklengths[3] = {2,2,3};
  MPI_Datatype types[3] = {MPI_UINT32_T, MPI_UINT64_T,MPI_UINT32_T};
  MPI_Aint offsets[3];
  offsets[0] = (MPI_Aint) offsetof(KernelArgsTy, Version);
  offsets[1] = (MPI_Aint) offsetof(KernelArgsTy, Tripcount);
  offsets[2] = (MPI_Aint) offsetof(KernelArgsTy, NumTeams);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &TD_Kernel_Args);
  MPI_Type_commit(&TD_Kernel_Args);

  return TARGETDART_SUCCESS;
}

tdrc declare_task_type() {
  const int nitems = 3;
  int blocklengths[3] = {1,2,1};
  MPI_Datatype types[3] = {MPI_LONG, MPI_INT32_T, MPI_INT};
  MPI_Aint offsets[3];
  offsets[0] = (MPI_Aint) offsetof(td_task_t, host_base_ptr);
  offsets[1] = (MPI_Aint) offsetof(td_task_t, num_teams);
  offsets[2] = (MPI_Aint) offsetof(td_task_t, local_proc);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &TD_MPI_Task);
  MPI_Type_commit(&TD_MPI_Task);

  return TARGETDART_SUCCESS;
}