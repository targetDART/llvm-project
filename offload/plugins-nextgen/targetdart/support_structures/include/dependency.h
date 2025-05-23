#ifndef _TARGETDART_DEPENDENCY_MANAGER_H
#define _TARGETDART_DEPENDENCY_MANAGER_H

#include <map>
#include <mutex>
#include <queue>
#include <unistd.h>
#include <vector>

#include "PluginManager.h"
#include "task.h"

//structure that stores by which task and how an address is used
struct read_write_counter {
  std::vector<td_task_t *>  tasks;
  std::vector<size_t> reads;
  std::vector<size_t> writes; 
  size_t total_read;
  size_t total_write;
};

class TD_Dependency_Manager {
private:
  // Stores the visied addresses and how it is currently used
  std::map<void *, read_write_counter> in_use_map;
  // needed for synchronizing the map
  std::mutex map_mutex;
  // computes predecessors of task based on map dependencies
  void process_deps(td_task_t *task);
public:
  TD_Dependency_Manager() = default;
  // adds a new task to the dependency manager to manager
  void add_task(td_task_t *task);
  // removes a task which was handeled by the dependency manager
  void delete_task(td_task_t *task);
};

#endif // _TARGETDART_DEPENDENCY_MANAGER_H
