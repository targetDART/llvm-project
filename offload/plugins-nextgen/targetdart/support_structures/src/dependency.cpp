#include "../include/dependency.h"
#include "omptarget.h"
#include <mutex>
#include <unordered_set>
#include <vector>


struct filtered_helper {
  void *address;
  size_t read;
  size_t write;
};

// Remove duplicates from the dependency list
void filter_addresses(KernelArgsTy const *KernelArgs, 
                     std::vector<filtered_helper> &filtered) {
  
  std::unordered_set<void *> seen;
  for (size_t i = 0; i < KernelArgs->NumArgs; i++) {
    if (!(KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_IMPLICIT)) {
      // filter out implicit mappings
      void *address = KernelArgs->ArgPtrs[i];
      size_t read = KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_TO ? 1 : 0;
      size_t write = KernelArgs->ArgTypes[i] & OMP_TGT_MAPTYPE_FROM ? 1 : 0;

      if (seen.insert(address).second) {
        filtered.push_back({address, read, write});
        DP("Dependency: Address: " DPxMOD " Read: %lu, Write %lu\n", DPxPTR(address), read, write);
      }
      //else
      //  DP("Address: " DPxMOD " was already inserted\n", DPxPTR(address));
    }
  }
}

void TD_Dependency_Manager::process_deps(td_task_t *task) {
  std::vector<filtered_helper> filtered;
  filter_addresses(task->KernelArgs, filtered);

  std::lock_guard<std::mutex> map_lock(map_mutex);

  for (filtered_helper &helper : filtered) {
    void *address = helper.address;
    size_t read = helper.read;
    size_t write = helper.write;
    DP("Dependency: Task (%ld%ld): Checking address " DPxMOD " for possible confilcts (r%luw%lu)\n", task->uid.rank, task->uid.id, DPxPTR(address), read, write);
    auto map_iter = in_use_map.find(address);
    if (map_iter != in_use_map.end()) {
      // Address already inserted before
      if (write) {
        // This task wants to write to address
        // -> Mark all tasks that currenly write or 
        //    read from this address as predecessors
        if (map_iter->second.total_write + map_iter->second.total_read > 0) {
          // At least one task currently writes or reads the address
          std::vector<td_task_t *> *tasks = &map_iter->second.tasks;
          std::vector<size_t> *writes = &map_iter->second.writes;
          std::vector<size_t> *reads = &map_iter->second.reads;

          for (size_t i = 0; i < tasks->size(); i++) {
            if (writes->at(i) || reads->at(i)) {
              DP("Dependency: Added task (%ld%ld) as a predecessor of task (%ld%ld), due to write dependency on " DPxMOD "\n", tasks->at(i)->uid.rank, tasks->at(i)->uid.id, task->uid.rank, task->uid.id, DPxPTR(address));
              task->n_predecessors++;
              tasks->at(i)->successors.push_back(task);
              // if we found one predecessor the task is blocked and can only be run after the predecessor is finished
              break;
            }
          }
          if (task->n_predecessors != 0)
            break;
        } else {
          // No task currently reads or writes the address
          // -> This address does not block the execution
        }
      } else if (read) {
        // This task wants to only read from address 
        // -> If address is already written to, 
        //    make this task a successor of the writing task
        if (map_iter->second.total_write) {
          // Address is being written to
          std::vector<td_task_t *> *tasks = &map_iter->second.tasks;
          std::vector<size_t> *writes = &map_iter->second.writes;
          for (size_t i = 0; i < writes->size(); i++) {
            if (writes->at(i)) {
              // tasks[i] is the one that writes to the address
              DP("Dependency: Added task (%ld%ld) as a predecessor of task (%ld%ld), due to read dependency on " DPxMOD "\n", tasks->at(i)->uid.rank, tasks->at(i)->uid.id, task->uid.rank, task->uid.id, DPxPTR(address));
              task->n_predecessors++;
              tasks->at(i)->successors.push_back(task);
              // if we found one predecessor the task is blocked and can only be run after the predecessor is finished
              break;
            }
          }
          if (task->n_predecessors != 0)
            break;
        }
      } else {
        // This task wants to neither read nor write on address
        // -> ignore
      }
    } else {
      // Address not in use yet
      // -> This address does not block the execution of the task
    }
  }

  if (task->n_predecessors == 0) {
    // This task has no_predeccesors and can run
    // The addresses are assigned to this task
    DP("Dependency: Task (%ld%ld) has no predecessors and can run\n", task->uid.rank, task->uid.id);
    for (filtered_helper &helper: filtered) {
      void *address = helper.address;
      size_t read = helper.read;
      size_t write = helper.write;
      read_write_counter &counter = in_use_map[address];
      counter.tasks.push_back(task);
      counter.reads.push_back(read);
      counter.writes.push_back(write);
      counter.total_read += read;
      counter.total_write += write;
    }

    // add task to the ready queue
    //{
    //  std::lock_guard<std::mutex> queue_lock(queue_mutex);
    //  ready_queue.push(task);
    //}
  }
}

void TD_Dependency_Manager::add_task(td_task_t *task) {
  process_deps(task);
}

void TD_Dependency_Manager::delete_task(td_task_t *task) {
  std::vector<filtered_helper> filtered;
  filter_addresses(task->KernelArgs, filtered);

  // clear the assigned addresses
  {
    std::lock_guard<std::mutex> map_lock(map_mutex);
    for (filtered_helper &helper : filtered) {
      void *address = helper.address;
      size_t read = helper.read;
      size_t write = helper.write;

      auto map_iter = in_use_map.find(address);
      int index = std::find(map_iter->second.tasks.begin(), map_iter->second.tasks.end(), task) - map_iter->second.tasks.begin();
      map_iter->second.tasks.erase(map_iter->second.tasks.begin() + index);
      map_iter->second.reads.erase(map_iter->second.reads.begin() + index);
      map_iter->second.writes.erase(map_iter->second.writes.begin() + index);
      map_iter->second.total_read -= read;
      map_iter->second.total_write -= write;

    }
    DP("Dependency: Task (%ld%ld) is finished and dependencies are cleared\n", task->uid.rank, task->uid.id);
  }

  for (td_task_t *successor : task->successors) {
    successor->n_predecessors--;

    if (successor->n_predecessors == 0) {
      // Successor task has no other predecessors
      // -> Check if requested addresses are available
      process_deps(successor);
    }
  }
}
