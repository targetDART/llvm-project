#include "omptarget.h"
#include "TargetDART.h"
#include "TD_common.h"
#include "TD_scheduling.h"

extern bool doRepartition;

/**
* triggers a global repratitioning of tasks accross all processes.
*/
void td_trigger_global_repartitioning(td_device_affinity affinity);