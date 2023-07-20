
#ifndef _OMPTARGET_TD_COST_ESTIMATION_H
#define _OMPTARGET_TD_COST_ESTIMATION_H

//TODO: define scheduling interface for TargetDART

#include <cstdint>
#include <unordered_map>
#include <vector>
#include "omptarget.h"
#include "device.h"

enum cost_index {CPU=0, GPU=1};

static std::unordered_map<intptr_t, double[2]> td_cost_estimation;



#endif