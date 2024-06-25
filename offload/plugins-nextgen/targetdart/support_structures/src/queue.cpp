#include "../include/queue.h"
#include "llvm/Support/Error.h"

#include <queue>

TD_Task_Queue::TD_Task_Queue() {
}

TD_Task_Queue::~TD_Task_Queue() {
}

td_task_t *TD_Task_Queue::getTask() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    if (queue.empty()) {
        return nullptr;
    }
    td_task_t *task = queue.front();
    queue.pop();
    return task;
}

void TD_Task_Queue::addTask(td_task_t *task) {
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue.push(task);
}

size_t TD_Task_Queue::getSize() {
    return queue.size();
}