#include <cstdint>
#include <unordered_map>
#include <vector>
#include "omptarget.h"
#include "device.h"


std::unordered_map<intptr_t, std::vector<double>> td_cost;