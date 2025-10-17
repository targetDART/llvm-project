#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

struct global_device_id {
    int rank;
    int32_t deviceID;

    bool operator==(global_device_id const &other) const {
        return rank == other.rank && deviceID == other.deviceID;
    }
};
template<>
struct std::hash<global_device_id>
{
    std::size_t operator()(global_device_id const &gdi) const noexcept
    {
        std::size_t h1 = std::hash<int>{}(gdi.rank);
        std::size_t h2 = std::hash<int32_t>{}(gdi.deviceID);
        return h1 ^ (h2 << 1);
    }
};

struct global_ptr {
    global_device_id gdi;
    void *ptr;

    bool operator==(global_ptr const &other) const {
        return gdi == other.gdi && ptr == other.ptr;
    }
};
template<>
struct std::hash<global_ptr>
{
    std::size_t operator()(global_ptr const &gptr) const noexcept
    {
        std::size_t h1 = std::hash<global_device_id>{}(gptr.gdi);
        std::size_t h2 = std::hash<void *>{}(gptr.ptr);
        return h1 ^ (h2 << 1);
    }
};

struct global_host_ptr {
    int rank;
    void const *ptr;

    bool operator==(global_host_ptr const &other) const {
        return rank == other.rank && ptr == other.ptr;
    }
};
template<>
struct std::hash<global_host_ptr>
{
    std::size_t operator()(global_host_ptr const &gptr) const noexcept
    {
        std::size_t h1 = std::hash<int>{}(gptr.rank);
        std::size_t h2 = std::hash<void const *>{}(gptr.ptr);
        return h1 ^ (h2 << 1);
    }
};


struct mapping {
    void const *host_ptr;
    size_t size;
    std::unordered_map<global_device_id, void *> device_ptrs;
    global_device_id base_deviceID;
};

class TD_Memory_Manager {
    private:

        std::unordered_map<void const *, mapping&> host_data_mapping;

        std::unordered_map<global_ptr, mapping> data_grouping;

        // Mutex for the data mapping
        std::mutex map_mutex;

    public:
        TD_Memory_Manager();
        void add_data_mapping(void const *host_ptr, void *base_ptr, int32_t base_deviceID, int base_rank);
        void *get_data_mapping(void const *host_ptr, int32_t deviceID, int rank);
        size_t get_data_mapping_size(void const *host_ptr);
        global_ptr get_global_base_ptr(void const *host_ptr);
        void *get_data_grouping(void *base_ptr, int32_t base_deviceID, int base_rank, int32_t deviceId, int rank);
        void const *get_host_ptr(void *base_ptr, int32_t base_deviceID, int base_rank);
        void register_allocation(void *base_ptr, void *device_ptr, size_t size, int32_t base_deviceID, int base_rank, int32_t deviceID, int rank);
        void register_deallocation(void *base_ptr, int32_t base_deviceID, int base_rank);
};
