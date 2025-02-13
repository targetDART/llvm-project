#include "../include/memory.h"
#include "../include/task.h"
#include "Shared/Debug.h"
#include <cstdint>

TD_Memory_Manager::TD_Memory_Manager(int32_t physical_device_count) {
    this->physical_device_count = physical_device_count;
}

void TD_Memory_Manager::add_data_mapping(void *TgtPtr, const void* HstPtr) {
    TRACE_START("Add_hst_tgt_mapping\n");
    std::lock_guard<std::mutex> lock(map_mutex);
    if (device_data_grouping.find(TgtPtr) == device_data_grouping.end()) {
        device_data_grouping.at(TgtPtr).host_ptr = HstPtr;
        hst_data_mapping.insert({HstPtr, device_data_grouping.at(TgtPtr)});
    } else {
        DP("Error: Device pointer already exists in mapping\n");
    }
    TRACE_END("Add_hst_tgt_mapping\n");
}

void *TD_Memory_Manager::get_data_mapping(int32_t deviceID, const void* HstPtr) {
    TRACE_START("Get_address_from_mapping\n");
    std::lock_guard<std::mutex> lock(map_mutex);
    if (hst_data_mapping.find(HstPtr) != hst_data_mapping.end()) {
        TRACE_END("Get_address_from_mapping\n");
        return hst_data_mapping.at(HstPtr).device_ptrs.at(deviceID);
    } else {
        DP("Error: Host pointer not found in mapping\n");
        TRACE_END("Get_address_from_mapping\n");
        return nullptr;
    }
}

size_t TD_Memory_Manager::get_data_mapping_size(const void* HstPtr) {
    TRACE_START("Get_size_from_mapping\n");
    std::lock_guard<std::mutex> lock(map_mutex);
    if (hst_data_mapping.find(HstPtr) != hst_data_mapping.end()) {
        TRACE_END("Get_size_from_mapping\n");
        return hst_data_mapping.at(HstPtr).size;
    } else {
        DP("Error: Host pointer not found in mapping\n");
        TRACE_END("Get_size_from_mapping\n");
        return 0;
    }
}

void TD_Memory_Manager::register_allocation(void* TgtPtr_key, void* TgtPtr_payload, size_t size, int32_t deviceID) {
    TRACE_START("Add_allocation_to_mapping\n");
    std::lock_guard<std::mutex> lock(map_mutex);
    if (device_data_grouping.find(TgtPtr_key) != device_data_grouping.end()) {
        device_data_grouping.insert({TgtPtr_key, {nullptr, size, std::vector<void *>(physical_device_count, nullptr)}});
    } else {
        if (device_data_grouping.at(TgtPtr_key).device_ptrs.at(deviceID) == nullptr) {
            device_data_grouping.at(TgtPtr_key).device_ptrs.at(deviceID) = TgtPtr_payload;
        } else {
            DP("Error: Device pointer already exists in grouping\n");        
        }
    }
    TRACE_END("Add_allocation_to_mapping\n");
}

void TD_Memory_Manager::register_deallocation(void *TgtPtr){
    TRACE_START("Remove_data_from_mapping\n");
    std::lock_guard<std::mutex> lock(map_mutex);
    if (device_data_grouping.find(TgtPtr) != device_data_grouping.end()) {
        const void *HstPtr = device_data_grouping.at(TgtPtr).host_ptr;
        if (HstPtr != nullptr) {
            hst_data_mapping.erase(HstPtr);        
        }
        device_data_grouping.erase(TgtPtr);
    } else {
        DP("Error: Device pointer not found in grouping\n");
    }
    TRACE_END("Remove_data_from_mapping\n");
}
