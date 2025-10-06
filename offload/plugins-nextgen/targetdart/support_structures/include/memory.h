#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

struct mapping {
    void const *host_ptr;
    size_t size;
    std::unordered_map<int32_t, void *> device_ptrs;
};

class TD_Memory_Manager {
    private:
        // Number of physical devices managed by other plugins
        int32_t physical_device_count;

        std::unordered_map<void const *, mapping&> host_data_mapping;

        std::unordered_map<void *, mapping> device_data_grouping;

        // Mutex for the data mapping
        std::mutex map_mutex;

    public:

        TD_Memory_Manager(int32_t physical_device_count);

        // Add host device pointer pair to mapping
        void add_data_mapping(void *TgtPtr, const void *HstPtr);

        // Get a device ptr for a host device mapping on a given device
        void *get_data_mapping(int32_t deviceID, const void *HstPtr);

        // Get the size for a host device mapping on a given device
        size_t get_data_mapping_size(const void* HstPtr);

        // Get the device ptr for a specific device in a grouping
        void *get_data_grouping(int32_t deviceID, void *TgtPtr);

        void register_allocation(void *base_ptr, void *device_ptr, size_t size, int32_t deviceID);

        // Remove device data from grouping
        void register_deallocation(void *TgtPtr);
};
