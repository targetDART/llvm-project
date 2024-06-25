#ifndef _TARGETDART_QUEUE_H
#define _TARGETDART_QUEUE_H

#include <queue>
#include <mutex>
#include "task.h"
#include "llvm/Support/Error.h"

/// Data structure that defines a single queue
class TD_Task_Queue {
private:
    std::queue<td_task_t*> queue;
    std::mutex queue_mutex;

public:

    TD_Task_Queue();
    ~TD_Task_Queue();

    td_task_t *getTask();
    void addTask(td_task_t *task);
    size_t getSize();

};

#endif //_TARGETDART_QUEUE_H