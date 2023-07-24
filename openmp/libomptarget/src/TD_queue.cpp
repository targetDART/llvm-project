//TODO: define communication interface for TargetDART

#include <cstdint>
#include <vector>
#include "omptarget.h"
#include "device.h"
#include <atomic>
#include <memory>
#include "TD_common.h"
#include "TD_queue.h"

[[nodiscard]] tdrc TD_Task_Queue::offerTask(td_task_t* task) {
    uint64_t oldTail = tail.load(std::memory_order_relaxed);
    bool noSuccess = TARGETDART_SUCCESS;
    while(noSuccess) {
        uint64_t currentTail = oldTail;
        while ((currentTail + 1) % BUFFER_SIZE == head.load(std::memory_order_relaxed) % BUFFER_SIZE) {
            currentTail = tail.load(std::memory_order_relaxed);
            if (currentTail == oldTail) {
                //The queue is probably full
                return TARGETDART_FAILURE;
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
    return TARGETDART_SUCCESS;
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