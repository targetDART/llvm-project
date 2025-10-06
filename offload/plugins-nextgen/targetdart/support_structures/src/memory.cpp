#include "../include/memory.h"
#include "../include/task.h"
#include "Shared/Debug.h"
#include <cstdint>
#include <mutex>

TD_Memory_Manager::TD_Memory_Manager(int32_t physical_device_count) {
    //this->physical_device_count = physical_device_count;
}

void TD_Memory_Manager::add_data_mapping(void *TgtPtr, void const *HstPtr) {
    TRACE_START("Add_hst_tgt_mapping\n");
    DP("Adding mapping for host pointer: " DPxMOD " to target pointer: " DPxMOD "\n", DPxPTR(HstPtr), DPxPTR(TgtPtr));

    std::lock_guard<std::mutex> lock(map_mutex);

    if (device_data_grouping.find(TgtPtr) != device_data_grouping.end()) {
        device_data_grouping[TgtPtr].host_ptr = HstPtr;
        host_data_mapping.insert({HstPtr,  device_data_grouping[TgtPtr]});
    } else {
        DP("Error: Device pointer does not exist in mapping\n");
    }
    
    TRACE_END("Add_hst_tgt_mapping\n");
}

void *TD_Memory_Manager::get_data_mapping(int32_t deviceID, void const *HstPtr) {
    TRACE_START("Get_address_from_mapping\n");
    DP("Getting mapping for host pointer: " DPxMOD " on device: %d\n", DPxPTR(HstPtr), deviceID);

    void *ret = nullptr;

    std::lock_guard<std::mutex> lock(map_mutex);
    if (auto iter = host_data_mapping.find(HstPtr); iter != host_data_mapping.end()) {
        if (auto iter2 = iter->second.device_ptrs.find(deviceID); iter2 != iter->second.device_ptrs.end()) {
            ret = iter2->second;
        } else {
            DP("Error: Mapping does not include device\n");
        }
    } else {
        DP("Error: Host pointer does not exist in mapping\n");
    }
    
    TRACE_END("Get_address_from_mapping\n");
    return ret;
}

size_t TD_Memory_Manager::get_data_mapping_size(const void *HstPtr) {
    TRACE_START("Get_size_from_mapping\n");
    DP("Getting size for host pointer: " DPxMOD "\n", DPxPTR(HstPtr));

    
    TRACE_END("Get_size_from_mapping\n");
    return 0;
}

void *TD_Memory_Manager::get_data_grouping(int32_t deviceID, void *TgtPtr){
    TRACE_START("Get_data_from_grouping\n");
    DP("Getting data for key: " DPxMOD " on device: %d\n", DPxPTR(TgtPtr), deviceID);

    void *ret = nullptr;

    std::lock_guard<std::mutex> lock(map_mutex);
    if (auto iter = device_data_grouping.find(TgtPtr); iter != device_data_grouping.end()) {
        DP("Found data for key: " DPxMOD " on device: %d\n", DPxPTR(TgtPtr), deviceID);
        ret = iter->second.device_ptrs[deviceID];
    } else {
        DP("Error: Device pointer not found in grouping\n");
    }
   
    TRACE_END("Get_data_from_grouping\n");
    return ret;
}

void TD_Memory_Manager::register_allocation(void *base_ptr, void *device_ptr, size_t size, int32_t deviceID) {
    TRACE_START("Add_allocation_to_mapping\n");
    DP("Registering allocation at key: " DPxMOD " with value: " DPxMOD " with size: %ld for device: %d\n", DPxPTR(base_ptr), DPxPTR(device_ptr), size, deviceID);

    std::lock_guard<std::mutex> lock(map_mutex);

    if (device_data_grouping.find(base_ptr) == device_data_grouping.end()) {
        device_data_grouping[base_ptr] = {nullptr, size, std::unordered_map<int32_t, void *>()};
    }

    if (device_data_grouping[base_ptr].device_ptrs.find(deviceID) == device_data_grouping[base_ptr].device_ptrs.end()) {
        device_data_grouping[base_ptr].device_ptrs[deviceID] = device_ptr;
    }

    TRACE_END("Add_allocation_to_mapping\n");
}

void TD_Memory_Manager::register_deallocation(void *TgtPtr){
    TRACE_START("Remove_data_from_mapping\n");
    DP("Registering deallocation at: " DPxMOD "\n", DPxPTR(TgtPtr));

    std::lock_guard<std::mutex> lock(map_mutex);
    if (auto iter = device_data_grouping.find(TgtPtr); iter != device_data_grouping.end()) {
        void const *HstPtr = iter->second.host_ptr;
        if (HstPtr != nullptr) {
            host_data_mapping.erase(HstPtr);        
        }
        device_data_grouping.erase(TgtPtr);
    } else {
        DP("Error: Device pointer not found in grouping\n");
    }

    TRACE_END("Remove_data_from_mapping\n");
}
