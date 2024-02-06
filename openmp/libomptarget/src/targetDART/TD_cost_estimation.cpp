#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "omp.h"
#include "omptarget.h"
#include "device.h"
#include "targetDART/TD_cost_estimation.h"
#include "targetDART/TD_common.h"

 
td_progression_t node_progression = {0, 0, 0, 0};

std::unordered_map<intptr_t,std::vector<double>> td_cost;

COST_DATA_TYPE td_get_task_cost(intptr_t hostptr, td_device_affinity device) {
    if (td_cost.find(hostptr) == td_cost.end()) {
        return DEFAULT_TASK_COST;
    }
    return td_cost[hostptr][device];
}


tdrc td_set_task_cost(intptr_t hostptr, td_device_affinity device, COST_DATA_TYPE cost){
    #pragma omp critical
    {
        if (td_cost.find(hostptr) == td_cost.end()) {
           
            std::vector<double> cost = {DEFAULT_TASK_COST, DEFAULT_TASK_COST, DEFAULT_TASK_COST, DEFAULT_TASK_COST, DEFAULT_TASK_COST};
            td_cost[hostptr] = cost;
        }        
        if (td_cost[hostptr][device] == DEFAULT_TASK_COST) {            
            td_cost[hostptr][device] = cost;
        }
    }
    return TARGETDART_SUCCESS;
}

// Starts the cost gathering for a given task
td_state_stamp_t td_start_timer(){
    td_state_stamp_t current_state;
    //TODO: handle performance counter
    //current_state.current_time = omp_get_wtime();
    return current_state;
}

// Stops the cost gathering for a given task
td_state_stamp_t td_stop_timer(){
    td_state_stamp_t current_state;
    //TODO: handle performance counter
    //current_state.current_time = omp_get_wtime();
    return current_state;
}


td_progression_t td_sum_states(td_state_stamp_t start_counter, td_state_stamp_t stop_counter){
    td_progression_t progress = {0, 0, 0, 0};
    double time = stop_counter.current_time - start_counter.current_time;
    progress.time_load += time;
    progress.compute_load += stop_counter.compute_load;
    progress.memory_load += stop_counter.memory_load;
    return progress;
}


td_progression_t td_update_progression(td_progression_t original, td_progression_t update){
    td_progression_t progress = {0,0,0,0};
    progress.advancement = std::min(original.advancement, update.advancement);
    progress.compute_load = original.compute_load + update.compute_load;
    progress.memory_load = original.memory_load + update.memory_load;
    progress.time_load = original.time_load + update.time_load;
    return progress;
}

void td_advance(uint64_t value) {
    node_progression.advancement += value;
}

void td_phase_progress(uint64_t progress){
    node_progression.phase_progress = progress;
}