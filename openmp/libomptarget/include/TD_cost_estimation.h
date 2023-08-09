
#ifndef _OMPTARGET_TD_COST_ESTIMATION_H
#define _OMPTARGET_TD_COST_ESTIMATION_H

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>
#include "omptarget.h"
#include "device.h"
#include "TD_common.h"

#define DEFAULT_TASK_COST 1
#define COST_DATA_TYPE long

typedef struct td_progression_t{
    uint64_t    advancement;
    uint64_t    phase_progress;
    uint64_t    compute_load;
    uint64_t    memory_load;
    double      time_load;
};

typedef struct td_state_stamp_t{
    uint64_t    compute_load;
    uint64_t    memory_load;
    double      current_time;
};

// increases the local advancement by value 
void td_advance(uint64_t value);

// Must be called the same number of times by every process
void td_phase_progress(uint64_t progress);

// a map of costs for each known task.
// the value defines the cost as vector, index is defined by td_device_cost
// only add base pointers to this table to allow communication of the table
// Access must happen in a critical section (e.g. omp critical) to avoid data races
extern std::unordered_map<intptr_t,std::vector<double>> td_cost;

// returns the cost of the given task for the chosen device
// returns a default value if the task is unknown
COST_DATA_TYPE td_get_task_cost(intptr_t hostptr, td_device_type device);

// Sets the cost on a specific device 
// returns false if a value is already present
tdrc td_set_task_cost(intptr_t hostptr, td_device_type device, COST_DATA_TYPE cost);

// Starts the cost gathering for a given task
td_state_stamp_t td_start_timer();

// Stops the cost gathering for a given task
td_state_stamp_t td_stop_timer();

// Sums up two timestamps into a single progression state
td_progression_t td_sum_states(td_state_stamp_t start_counter, td_state_stamp_t stop_counter);

// Sums up two progression states into a single progression state
td_progression_t td_sum_progression(td_progression_t device0, td_progression_t device1);

#endif