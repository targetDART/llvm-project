#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "omptarget.h"
#include "device.h"
#include "TD_cost_estimation.h"


std::unordered_map<intptr_t,std::pair<double, double>> td_cost;

double td_get_task_cost(intptr_t hostptr, bool onGPU) {
    if (td_cost.find(hostptr) == td_cost.end()) {
        return DEFAULT_TASK_COST;
    }
}