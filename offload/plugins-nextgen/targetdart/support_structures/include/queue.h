#ifndef _TARGETDART_QUEUE_H
#define _TARGETDART_QUEUE_H

#include <cstdint>
#include <queue>
#include <mutex>
#include "communication.h"
#include "task.h"
#include "llvm/Support/Error.h"

/// Data structure that defines a single queue
class TD_Task_Queue {
private:
    std::queue<td_task_t*> queue;
    std::mutex queue_mutex;
    // lookout for potential overflows
    COST_DATA_TYPE total_cost;

public:

    TD_Task_Queue();
    ~TD_Task_Queue();

    td_task_t *getTask();
    void addTask(td_task_t *task);
    size_t getSize();
    COST_DATA_TYPE getCost();


};

#endif //_TARGETDART_QUEUE_H