
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

    [[nodiscard]] bool offerTask(td_task_t* task);

    [[nodiscard]] td_task_t* pollTask(std::function<bool(std::atomic<uint64_t>&, uint64_t)>* blockingFunction);
};

[[nodiscard]] bool TD_Task_Queue::offerTask(td_task_t* task) {
    uint64_t oldTail = tail.load(std::memory_order_relaxed);
    bool noSuccess = true;
    while(noSuccess) {
        uint64_t currentTail = oldTail;
        while ((currentTail + 1) % BUFFER_SIZE == head.load(std::memory_order_relaxed) % BUFFER_SIZE) {
            currentTail = tail.load(std::memory_order_relaxed);
            if (currentTail == oldTail) {
                //The queue is probably full
                return false;
            }
            oldTail = currentTail;
        }
        //The queue is known to not be full, assuming currentTail is still equal to tail.

        std::atomic<td_task_t*>& atomicEntry = workBuffer.at(currentTail % BUFFER_SIZE);
        td_task_t* previousEntry = nullptr;
        noSuccess = !atomicEntry.compare_exchange_strong(previousEntry, task, std::memory_order_release);
        //Integer overflow possible. Note that overflow is well-defined for unsigned int and BUFFER_SIZE is a power of two.
        currentTail = currentTail + 1;
        //ABA Problem in case of integer overflow possible. Currently, this is ignored.
        if (tail.compare_exchange_strong(oldTail, currentTail, std::memory_order_relaxed)) {
//            tail.notify_all(); //Blocking on tail is a bad idea, 100x worse performance than busy waiting or so.
            oldTail = currentTail;
        }
    }
    size.fetch_add(1, MEM_ORDER);
    return true;
}

/**
 * Poll from the queue.
 * Fails when queue is empty (returns null).
 * WARNING: Spurious failures possible while multiple threads are polling and at least one thread is offering.
 * @tparam T Type of elements stored in the Task Queue (usually T = Task)
 * @return The first element of the queue which is removed from the queue.
 */
[[nodiscard]] td_task_t* TD_Task_Queue::pollTask(std::function<bool(std::atomic<uint64_t>&, uint64_t)>* blockingFunction) {
    while (size.load(MEM_ORDER) == 0) {
        if (blockingFunction != nullptr) {
            bool shutdown = (*blockingFunction)(size, 0);
            if (shutdown) {
                return nullptr;
            }
        }
        return nullptr;
    }
    td_task_t* entry = nullptr;
    uint64_t oldHead = head.load(std::memory_order_relaxed);
    while(entry == nullptr) {
        uint64_t currentHead = oldHead;
        //prevent popping from "empty" queue
        //This is affected by the ABA problem. However, this does not cause any issues here.
        // If this mistakenly assumes that the queue is empty, there will be a spurious failure of pollTask. This only
        // happens when tail was updated recently. As the thread pool continuously polls until it is terminated (no more new tasks coming in),
        // the thread pool can avoid correctness issues by polling after termination without the risk of spurious failures.
        while (currentHead == tail.load(std::memory_order_relaxed)) {
            currentHead = head.load(std::memory_order_relaxed);
            if (currentHead == oldHead) {
                //The queue is probably empty
                if (blockingFunction != nullptr) {
                    bool shutdown = (*blockingFunction)(tail, oldHead);
                    if (shutdown) {
                        return nullptr;
                    }
                } else {
                    return nullptr;
                }
            }
            oldHead = currentHead;
        }

        std::atomic<td_task_t*>& atomicEntry = workBuffer.at(currentHead % BUFFER_SIZE);
        td_task_t* previousEntry = atomicEntry.load(std::memory_order_relaxed);
        if (previousEntry != nullptr) { //This line doesn't change the outcome but maybe the performance
            if (atomicEntry.compare_exchange_strong(previousEntry, nullptr, std::memory_order_acquire)) {
                entry = previousEntry;
            }
        }
        currentHead = currentHead + 1;
//        if (currentHead < oldHead) {
        //Integer overflow. Note that overflow is well-defined for unsigned int and BUFFER_SIZE is a power of two.
        // However, this can lead to the ABA problem. Another thread could be setting head from n to n+1, but
        //the element n wasn't removed from the queue in this generation yet.
        //TODO handle ABA problem when the head/tail overflow + another thread was frozen here for a long time (a lot more than 10 years with 2022's clock speeds).
//        }
        //CAS strong guarantees progress, weak should be fine in sane systems though.
        if (head.compare_exchange_strong(oldHead, currentHead, std::memory_order_relaxed)) {
            oldHead = currentHead;
        }
    }
    size.fetch_add(-1, MEM_ORDER);
    return entry;
}

inline TD_Task_Queue::TD_Task_Queue() {
    head.store((uint64_t)0);
    tail.store((uint64_t)0);
    //Probably useless and takes too much time
//    for (std::atomic<T*>& atomic : this->workBuffer) {
//        atomic.store(nullptr);
//    }
}

#endif