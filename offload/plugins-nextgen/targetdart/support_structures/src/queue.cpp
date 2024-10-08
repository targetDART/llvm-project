#include "../include/queue.h"
#include "llvm/Support/Error.h"

#include <queue>

TD_Task_Queue::TD_Task_Queue() {
    total_cost = 0;
}

TD_Task_Queue::~TD_Task_Queue() {
}

td_task_t *TD_Task_Queue::getTask() {
    TRACE_START("get_task_from_base_queue\n");
    std::unique_lock<std::mutex> lock(queue_mutex);
    if (queue.empty()) {
        TRACE_END("get_task_from_base_queue\n");
        return nullptr;
    }
    td_task_t *task = queue.front();
    queue.pop();
    total_cost -= task->cached_total_sizes;
    TRACE_END("get_task_from_base_queue\n");
    return task;
}

void TD_Task_Queue::addTask(td_task_t *task) {
    TRACE_START("add_task_to_base_queue\n");
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue.push(task);
    total_cost += task->cached_total_sizes;
    TRACE_END("add_task_to_base_queue\n");
}

size_t TD_Task_Queue::getSize() {
    return queue.size();
}

COST_DATA_TYPE TD_Task_Queue::getCost() {
    return total_cost;
}