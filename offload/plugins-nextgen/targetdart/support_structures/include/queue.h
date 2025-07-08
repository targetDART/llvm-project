#ifndef _TARGETDART_QUEUE_H
#define _TARGETDART_QUEUE_H

#include <cstdint>
#include <functional>
#include <queue>
#include <mutex>
#include <atomic>
#include <memory>
#include "communication.h"
#include "task.h"
#include "llvm/Support/Error.h"

#define BUFFER_SIZE 16536
#if __cplusplus >= 202002L
    #define MEM_ORDER std::memory_order::relaxed
#else
    #define MEM_ORDER std::memory_order::memory_order_relaxed
#endif

/// Data structure that defines a single queue
class TD_Task_Queue {
private:

    alignas(64) std::atomic<uint64_t> head{0};
    
    //Vector size is a power of two. Atomics are used to allow having multiple readers/writers.
    alignas(64) std::vector<std::atomic<td_task_t*>> workBuffer = std::vector<std::atomic<td_task_t*>> (BUFFER_SIZE);

    //tail incremented when offering
    alignas(64) std::atomic<uint64_t> tail{0};
    //having a separate cache line for busy waiting seems to reduce cache line ping pong
    alignas(64) std::atomic<uint64_t> size{0};
    //stores the current load on the queue
    alignas(64) std::atomic<uint64_t> cost{0};

    [[nodiscard]] tdrc offer_task(td_task_t* task);
    [[nodiscard]] td_task_t* poll_task(std::function<bool(std::atomic<uint64_t>&, uint64_t)>* blockingFunction);

public:

    TD_Task_Queue();
    ~TD_Task_Queue();

    td_task_t *getTask();
    void addTask(td_task_t *task);
    size_t getSize();
    COST_DATA_TYPE getCost();


};

#endif //_TARGETDART_QUEUE_H