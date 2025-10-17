#include "../include/memory.h"
#include "../include/task.h"
#include "Shared/Debug.h"
#include <cstdint>
#include <mutex>


TD_Memory_Manager::TD_Memory_Manager() {}

void TD_Memory_Manager::add_data_mapping(void const *host_ptr, void *base_ptr, int32_t base_deviceID, int base_rank) {
    TRACE_START("Add_hst_tgt_mapping\n");
    DP("Adding mapping for host pointer: " DPxMOD " to base pointer: " DPxMOD " on device %d on rank %d\n", 
       DPxPTR(host_ptr), DPxPTR(base_ptr), base_deviceID, base_rank);

    std::lock_guard<std::mutex> lock(map_mutex);

    global_ptr global_base_ptr = {{base_rank, base_deviceID}, base_ptr};

    if (data_grouping.find(global_base_ptr) != data_grouping.end()) {
        data_grouping[global_base_ptr].host_ptr = host_ptr;
        host_data_mapping.insert({host_ptr,  data_grouping[global_base_ptr]});
    } else {
        DP("Error: Device pointer does not exist in mapping\n");
    }
    
    TRACE_END("Add_hst_tgt_mapping\n");
}

void *TD_Memory_Manager::get_data_mapping(void const *host_ptr, int32_t deviceID, int rank) {
    TRACE_START("Get_address_from_mapping\n");
    DP("Getting mapping for host pointer: " DPxMOD " on device %d on rank %d\n", DPxPTR(host_ptr), deviceID, rank);

    void *ret = nullptr;

    global_device_id gdi{ rank, deviceID };

    std::lock_guard<std::mutex> lock(map_mutex);
    if (auto iter = host_data_mapping.find(host_ptr); iter != host_data_mapping.end()) {
        if (auto iter2 = iter->second.device_ptrs.find(gdi); iter2 != iter->second.device_ptrs.end()) {
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

size_t TD_Memory_Manager::get_data_mapping_size(void const *host_ptr) {
    TRACE_START("Get_size_from_mapping\n");
    DP("Getting size for host pointer: " DPxMOD "\n", DPxPTR(host_ptr));

    size_t ret = 0;

    if (auto iter = host_data_mapping.find(host_ptr); iter != host_data_mapping.end()) {
        ret = iter->second.size;
    } else {
        DP("host_ptr not found in mapping\n");
    }

    
    TRACE_END("Get_size_from_mapping\n");
    return ret;
}

global_ptr TD_Memory_Manager::get_global_base_ptr(void const *host_ptr) {
    DP("Getting global base pointer of host_ptr " DPxMOD "\n", DPxPTR(host_ptr));

    global_ptr ret = {{0, 0}, nullptr};

    if(auto iter = host_data_mapping.find(host_ptr); iter != host_data_mapping.end()) {
        global_device_id base_deviceID = iter->second.base_deviceID;
        ret = {base_deviceID, iter->second.device_ptrs[base_deviceID]};
        DP("Found base_ptr: " DPxMOD " (deviceID: %d, rank: %d) for host_ptr: " DPxMOD "\n", DPxPTR(ret.ptr), ret.gdi.deviceID, ret.gdi.rank, DPxPTR(host_ptr));
    } else {
        DP("Found no mapping for host_ptr " DPxMOD ", return nullptr\n", DPxPTR(host_ptr));
    }

    return ret;
}

void *TD_Memory_Manager::get_data_grouping(void *base_ptr, int32_t base_deviceID, int base_rank, int32_t deviceID, int rank) {
    TRACE_START("Get_data_from_grouping\n");
    DP("Getting data for base_ptr: " DPxMOD " (base_deviceID %d, base_rank %d) on device %d on rank %d\n", DPxPTR(base_ptr), base_deviceID, base_rank, deviceID, rank);

    void *ret = nullptr;

    global_device_id gdi{rank, deviceID};
    global_ptr global_base_ptr{{base_rank, base_deviceID}, base_ptr};

    std::lock_guard<std::mutex> lock(map_mutex);
    if (auto iter = data_grouping.find(global_base_ptr); iter != data_grouping.end()) {
        DP("Found data for key: " DPxMOD "\n", DPxPTR(base_ptr));
        ret = iter->second.device_ptrs[gdi];
    } else {
        DP("Error: Device pointer not found in grouping\n");
    }
   
    TRACE_END("Get_data_from_grouping\n");
    return ret;
}

void const *TD_Memory_Manager::get_host_ptr(void *base_ptr, int32_t base_deviceID, int base_rank) {
    void const *ret = nullptr;
    DP("Getting host_ptr for base_ptr " DPxMOD "(deviceID: %d, base_rank: %d)\n", DPxPTR(base_ptr), base_deviceID, base_rank);

    global_ptr global_base_ptr = {{base_rank, base_deviceID}, base_ptr};

    if (auto iter = data_grouping.find(global_base_ptr); iter != data_grouping.end()) {
        ret = iter->second.host_ptr;
        DP("Found host ptr:" DPxMOD "\n", DPxPTR(ret));
    } else {
        DP("Warning: host ptr not found, return nullptr\n");
    }

    return ret;
}

void TD_Memory_Manager::register_allocation(void *base_ptr, void *device_ptr, size_t size, int32_t base_deviceID, int base_rank, int deviceID, int rank) {
    TRACE_START("Add_allocation_to_mapping\n");
    DP("Registering allocation at base_ptr: " DPxMOD " (base_deviceID: %d, base_rank: %d) with value: " DPxMOD " with size: %ld for device %d on rank %d\n", 
       DPxPTR(base_ptr), base_deviceID, base_rank, DPxPTR(device_ptr), size, deviceID, rank);

    std::lock_guard<std::mutex> lock(map_mutex);

    global_device_id gdi = {rank, deviceID};
    global_device_id global_base_deviceID = {base_rank, base_deviceID};
    global_ptr global_base_ptr = {global_base_deviceID, base_ptr};

    // Is base_ptr already grouped with other addresses?
    if (data_grouping.find(global_base_ptr) == data_grouping.end()) {
        data_grouping[global_base_ptr] = {nullptr, size, std::unordered_map<global_device_id, void *>(), global_base_deviceID};
    }

    // Is device already part of the base_ptr group?
    if (data_grouping[global_base_ptr].device_ptrs.find(gdi) == data_grouping[global_base_ptr].device_ptrs.end()) {
        data_grouping[global_base_ptr].device_ptrs[gdi] = device_ptr;
    }

    TRACE_END("Add_allocation_to_mapping\n");
}

void TD_Memory_Manager::register_deallocation(void *base_ptr, int32_t base_deviceID, int base_rank) {
    TRACE_START("Remove_data_from_mapping\n");
    DP("Registering deallocation at: " DPxMOD " on device %d on rank %d\n", DPxPTR(base_ptr), base_deviceID, base_rank);

    global_ptr global_base_ptr{{base_rank, base_deviceID}, base_ptr};

    std::lock_guard<std::mutex> lock(map_mutex);
    if (auto iter = data_grouping.find(global_base_ptr); iter != data_grouping.end()) {
        void const *host_ptr = iter->second.host_ptr;
        if (host_ptr != nullptr) {
            host_data_mapping.erase(host_ptr);        
        }
        data_grouping.erase(global_base_ptr);
    } else {
        DP("Error: Device pointer not found in grouping\n");
    }

    TRACE_END("Remove_data_from_mapping\n");
}
