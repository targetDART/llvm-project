
#ifndef _OMPTARGET_TD_QUEUE_H
#define _OMPTARGET_TD_QUEUE_H

//TODO: define communication interface for TargetDART

#include <cstdint>
#include <vector>
#include "omptarget.h"
#include "device.h"
#include <atomic>
#include <memory>
#include "TD_common.h"

#define BUFFER_SIZE 16536
#if __cplusplus >= 202002L
    #define MEM_ORDER std::memory_order::relaxed
#else
    #define MEM_ORDER std::memory_order::memory_order_relaxed
#endif

class TD_Task_Queue {
private:
    //TODO ALIGNMENT might be super important for performance
    //TODO double check all memory ordering semantics or just switch to sequential consistency
    //For head and tail this ring buffer uses uint64_t. They start at 0 and are incremented until they overflow.
    // When accessing the buffer a % BUFFER_SIZE needs to be done on the index to stay in bounds.
    // Setting the head or tail to a lower value will make the ABA problem very likely to cause issues in practice.
    // By not repeating values in head and tail
    // the ABA problem will only occur after several years of runtime with another thread being unscheduled at the
    // worst possible place for the whole time.

    //head incremented when polling
    alignas(64) std::atomic<uint64_t> head{0};
    //TODO Padding is possible here, could also move tail next to head with/without padding.

    //TODO Maybe avoid false sharing of neighboring atomics by translating the index using ((index << 3) | (index & 0b111)) % BUFFER_SIZE
    // instead of using % BUFFER_SIZE directly. This way the next index should be on a different cache line (assuming 8 atomics fit into a cache line together)
    //Vector size is a power of two. Atomics are used to allow having multiple readers/writers.
    alignas(64) std::vector<std::atomic<td_task_t*>> workBuffer = std::vector<std::atomic<td_task_t*>> (BUFFER_SIZE);
public:
    //tail incremented when offering
    alignas(64) std::atomic<uint64_t> tail{0};
    //having a separate cache line for busy waiting seems to reduce cache line ping pong
    alignas(64) std::atomic<uint64_t> size{0};

public:
    TD_Task_Queue();

    [[nodiscard]] tdrc offerTask(td_task_t* task);

    [[nodiscard]] td_task_t* pollTask(std::function<bool(std::atomic<uint64_t>&, uint64_t)>* blockingFunction);
};

#endif