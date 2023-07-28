
#ifndef _OMPTARGET_TD_COST_ESTIMATION_H
#define _OMPTARGET_TD_COST_ESTIMATION_H

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>
#include "omptarget.h"
#include "device.h"

#define DEFAULT_TASK_COST 1

// a map of costs for each known task.
// the value defines the cost as pair (CPU_cost, GPU_cost)
// only add base pointers to this table to allow communication of the table
extern std::unordered_map<intptr_t, std::pair<double, double>> td_cost;

// returns the cost of the given task for the chosen device
// returns a default value if the task is unknown
double td_get_task_cost(intptr_t hostptr, bool onGPU);


#endif