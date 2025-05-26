#ifndef _TARGETDART_DEPENDENCY_MANAGER_H
#define _TARGETDART_DEPENDENCY_MANAGER_H

#include <map>
#include <mutex>
#include <queue>
#include <unistd.h>
#include <vector>

#include "PluginManager.h"
#include "task.h"


class TD_Dependency_Manager {
private:
  //structure that stores by which task and how an address is used
  struct read_write_counter {
    std::vector<td_task_t *>  tasks;
    std::vector<size_t> reads;
    std::vector<size_t> writes; 
    size_t total_read;
    size_t total_write;
  };
  struct filtered_helper {
    void *address;
    size_t read;
    size_t write;
  };
  // Stores the visied addresses and how it is currently used
  std::map<void *, read_write_counter> in_use_map;
  // needed for synchronizing the map
  std::mutex map_mutex;
  // adds a new task to the dependency manager to manager
  void add_task_dependencies(td_task_t *task, std::vector<filtered_helper> const &filtered);
  void filter_addresses(KernelArgsTy const *KernelArgs, std::vector<filtered_helper> &filtered) const;
public:
  TD_Dependency_Manager() = default;
  // removes a task which was handeled by the dependency manager
  void clear_task_dependencies(td_task_t *task);
  // computes predecessors of task based on map dependencies
  // returns if task is runnable
  [[nodiscard]]
  bool process_deps(td_task_t *task);
};

#endif // _TARGETDART_DEPENDENCY_MANAGER_H
