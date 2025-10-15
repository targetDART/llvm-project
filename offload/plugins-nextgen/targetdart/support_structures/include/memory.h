#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

struct global_device_id {
    int rank;
    int32_t device_id;

    bool operator==(global_device_id const &other) const {
        return rank == other.rank && device_id == other.device_id;
    }
};
template<>
struct std::hash<global_device_id>
{
    std::size_t operator()(global_device_id const &gdi) const noexcept
    {
        std::size_t h1 = std::hash<int>{}(gdi.rank);
        std::size_t h2 = std::hash<int32_t>{}(gdi.device_id);
        return h1 ^ (h2 << 1);
    }
};

struct global_ptr {
    global_device_id gid;
    void *ptr;

    bool operator==(global_ptr const &other) const {
        return gid == other.gid && ptr == other.ptr;
    }
};
template<>
struct std::hash<global_ptr>
{
    std::size_t operator()(global_ptr const &gptr) const noexcept
    {
        std::size_t h1 = std::hash<global_device_id>{}(gptr.gid);
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
};

class TD_Memory_Manager {
    private:

        std::unordered_map<void const *, mapping&> host_data_mapping;

        std::unordered_map<global_ptr, mapping> data_grouping;

        // Mutex for the data mapping
        std::mutex map_mutex;

    public:
        TD_Memory_Manager();
        void add_data_mapping(void const *host_ptr, void *base_ptr, int32_t deviceID, int rank);
        void *get_data_mapping(void const *host_ptr, int32_t deviceID, int rank);
        size_t get_data_mapping_size(const void* HstPtr);
        void *get_data_grouping(void *base_ptr, int32_t base_deviceID, int base_rank, int32_t deviceId, int rank);
        void register_allocation(void *base_ptr, void *device_ptr, size_t size, int32_t base_deviceID, int base_rank, int32_t deviceID, int rank);
        void register_deallocation(void *base_ptr, int32_t base_deviceID, int base_rank);
};
