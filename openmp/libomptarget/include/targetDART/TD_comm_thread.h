#include "omptarget.h"
#include "targetDART/TargetDART.h"
#include "targetDART/TD_common.h"
#include "targetDART/TD_scheduling.h"

extern bool doRepartition;

/**
* triggers a global repratitioning of tasks accross all processes.
*/
void td_trigger_global_repartitioning(td_device_affinity affinity);

/*
* initializes the scheduling and executor threads.
* The threads are pinned to the cores defined in the parameters.
* the number of exec placements must be equal to the omp_get_num_devices() + 1.
*/
tdrc td_init_threads(std::vector<int> *assignments);

/**
* synchronizes all processes to ensure all tasks are finished.
*/
tdrc td_finalize_threads();