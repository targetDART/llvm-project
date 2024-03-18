#include "omp.h"
#include "omptarget.h"
#include "device.h"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <fcntl.h>
#include "unistd.h"
#include "mpi.h"
#include <link.h>
#include <mutex>
#include <pthread.h>
#include <queue>
#include <dlfcn.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include "targetDART/TargetDART.h"
#include "targetDART/TD_communication.h"
#include "targetDART/TD_common.h"
#include "targetDART/TD_cost_estimation.h"
#include "targetDART/TD_scheduling.h"
#include "targetDART/TD_comm_thread.h"
#include "../private.h"


// Mixed endianness is not supported

static bool __td_initialized = false;

static bool __td_did_initialize_mpi = false;

// stores all tasks that are migrated or replicated to simplify receiving results.
std::unordered_map<long long, td_task_t*> td_remote_task_map;

// stores how many tasks have been generated in total
std::atomic<long> __td_tasks_generated = {0};

// communicator for remote task requests
MPI_Comm targetdart_comm;

int td_comm_size;
int td_comm_rank;

MPI_Datatype TD_Kernel_Args;
MPI_Datatype TD_MPI_Task;

// array that holds image base addresses
std::vector<intptr_t> _image_base_addresses;
TD_Conditional_Map* conditional_map;

//removes spaces from text
template<typename T>
T remove_space(T beg, T end)
{
    T dest = beg;
    for (T itr = beg;itr != end; ++itr)
        if (!isspace(*itr))
            *(dest++) = *itr;
    return dest;
}

// splits a String by "," or ";"
std::vector<std::string> split(std::string base) {
	char delimiter = ',';
	std::vector<std::string> res;

	if (base.find(';') != std::string::npos) {
		delimiter = ';';
	}

	bool mode = true;

	long long start = 0;
	long long length = 0;
	for ( int i = 0; i < base.size(); i++) {
		if (base[i] == '\'' || base[i] =='"') {
			//toggle mode
			mode = mode != false;
			length++;
			continue;
		}
		if (mode && base[i] == delimiter) {
			if (length > 0) {
				std::string block = base.substr(start, length);
				block.erase(remove_space(block.begin(), block.end()), block.end());
				if (block != "") {
					res.push_back(block);
				}
			}
			start = i + 1;
			length = 0;
			continue;
		}
		length++;
	}
	if (length != 0) {
		std::string block = base.substr(start, length);
		res.push_back(block);
	}
	return res;
}

/**
* Reads the environment variable TD_MANAGEMENT
*/
std::vector<int>* __td_get_thread_placement_from_env() {
  std::vector<int> *placements = new std::vector<int>(omp_get_num_devices() + 2, -1);

  if (std::getenv("TD_MANAGEMENT") == NULL) {
    return placements;
  }

  std::string management = std::getenv("TD_MANAGEMENT");

  std::vector<std::string> assignments = split(management);

  placements->at(0) = std::stoi(assignments.at(0));
  DB_TD("Scheduling thread assiged to core %d from env", (*placements)[0]);

  for (int i = 1; i < std::min( assignments.size(), placements->size()); i++) {
    placements->at(i) = std::stoi(assignments.at(i));
    DB_TD("Execution thread %d assiged to core %d from env", i - 1, (*placements)[i]);
  }
  return placements;
}


//Adds a task to the TargetDART runtime
int td_add_task( ident_t *Loc, int32_t NumTeams,
                        int32_t ThreadLimit, void *HostPtr,
                        KernelArgsTy *KernelArgs, int64_t *DeviceId) 
{
  DB_TD("create new task from HostPtr (0x%020x)", HostPtr);
  // create internal task data structure
  td_task_t *task = (td_task_t*) std::malloc(sizeof(td_task_t));
  task->host_base_ptr = apply_image_base_address((intptr_t) HostPtr, false);
  task->KernelArgs = KernelArgs;
  task->Loc = Loc;
  task->num_teams = NumTeams;
  task->thread_limit = ThreadLimit;
  task->local_proc = td_comm_rank;
  task->uid = __td_tasks_generated.fetch_add(1);
  DB_TD("add task (%d%d) to queue from HostPtr (0x%020x)", task->local_proc, task->uid, HostPtr);

  DB_TD("add task (%d%d) to queue from BasePtr (0x%020x)", task->local_proc, task->uid, task->host_base_ptr);

  DB_TD("Current hardware thread %d", syscall(__NR_gettid));

  td_conditional_wrapper_t* cond_var = conditional_map->add_conditional(task->uid);

  
  //start yielding lock here to avoid rare deadlocks in which the task is finished, before td_yield() is called 
  //std::unique_lock<std::mutex> lock(cond_var->thread_mutex);

  //initial assignment
  if (*DeviceId >= SPECIFIC_DEVICE_RANGE_START) {
    task->affinity = TD_FIXED;
    td_add_to_load_local(task, (*DeviceId - DEVICE_BASE - TD_NUM_AFFINITIES));
  } else  {
    task->affinity = (td_device_affinity) (*DeviceId - DEVICE_BASE);
    td_add_to_load_local(task);
  }

  /*
  Number of hidden_helper_threads is defined by __kmp_hidden_helper_threads_num in kmp_runtime.cpp line 9142
  Default is currently 8
  */

  /*if (targetdart_comm_rank == 0) {
    td_send_task(1, task);
  } else {
    td_receive_task(0, task);
  }*/


  //yields until the current task has finished.
  td_yield(task);
  // make sure the current thread sleeps here until the task is executed.
  // Run a dedicated device thread which consumes the task for its assigned device. 
  //(think about multiple threads per device, at least for GPUs to overlap com/comp)
  // ensure that the local thread cannot progress until the task is finished and the Data is available again 
  // Ideas: yield + ping, Barrier, Mutex, busy-waiting
  // requires additional parameters in task

  return task->return_code;
}

// initializes the targetDART lib
/*
main_ptr needs to be the same function in the same binary in all processes.
Preferably main().
If MPI is used in the user code, MPI must be initialized before TargetDART.
*/
int initTargetDART(void* main_ptr) {

  if (__td_initialized) {
    return TARGETDART_SUCCESS;
  }

  // check whether MPI is initialized, otherwise do so
  int mpi_initialized, err;
  mpi_initialized = 0;
  int provided;
  err = MPI_Initialized(&mpi_initialized);
  if(!mpi_initialized) {
      // MPI_Init(NULL, NULL);
      MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
      __td_did_initialize_mpi = true;
      DB_TD("Internal MPI initialization");
  }
  MPI_Query_thread(&provided);
  if(provided != MPI_THREAD_MULTIPLE) {
    //TODO: Fehler meldung
  }

  //TODO: add redirect to split mpioutput

  //decare KernelArgs,task as MPI Type
  declare_KernelArgs_type();
  declare_task_type();

  // create separate communicator for targetdart
  err = MPI_Comm_dup(MPI_COMM_WORLD, &targetdart_comm);
  MPI_Comm_size(targetdart_comm, &td_comm_size);
  MPI_Comm_rank(targetdart_comm, &td_comm_rank);
  DB_TD("MPI environment setup finished");

  //Initialize the data structures for scheduling
  td_init_task_queues();
  std::unordered_map<intptr_t,std::vector<double>> td_cost;
  // Initialize the map of remote and replicated tasks
  td_remote_task_map = std::unordered_map<long long, td_task_t*>();
  conditional_map = new TD_Conditional_Map();


  // define the base address of the current process
  get_base_address(main_ptr);

  // initial placements
  // TODO: Implement callbacks for compile time parameters or Environment variables
  std::vector<int> *placements = __td_get_thread_placement_from_env();
  td_init_threads(placements);
  //free(placements);

  // Init devices during installation
  for (long i = 0; i < omp_get_num_devices(); i++) {
    if (checkDeviceAndCtors(i, nullptr)) {
      DB_TD("Not offloading to device %" PRId64 "\n", i);
      return TARGETDART_FAILURE;
    }
  }

  DB_TD("Initialization completed");
  return TARGETDART_SUCCESS;
}

/* finalizes targetDART lib, iff TD initilized MPI it also needs to finalize it.
 * Should be one of the last functions in your program.
*/
int finalizeTargetDART() {
  //synchronize all threads and processes
  td_finalize_threads();

  delete conditional_map;

  //finalize MPI
  if (__td_did_initialize_mpi) {
    MPI_Finalize();
    DB_TD("local MPI finalized");
  }

    DB_TD("targetDART finalized");
  return TARGETDART_SUCCESS;
}

/*
 * Function set_image_base_address
 * Sets base address of particular image index.
 * This is necessary to determine the entry point for functions that represent a target construct
 */
int32_t set_image_base_address(int idx_image, intptr_t base_address) {
    if(_image_base_addresses.size() < idx_image+1) {
        _image_base_addresses.resize(idx_image+1);
    }
    // set base address for image (last device wins)
    DB_TD("set_image_base_address (enter) Setting base_address: " DPxMOD " for img: %d", DPxPTR((void*)base_address), idx_image);
    _image_base_addresses[idx_image] = base_address;
    return TARGETDART_SUCCESS;
}

/*
 * Function apply_image_base_address
 * Adds the base address to the address if iBaseAddress == true
 * Else it creates a base address
 */
intptr_t apply_image_base_address(intptr_t base_address, bool isBaseAddress) {
    if (isBaseAddress) {
      return base_address + _image_base_addresses[99];
    }
    return base_address - _image_base_addresses[99];
}

/**
* Function get_base_address
* Generates the base address for the current process
* 
* Works only for identical BINARIES
*/
tdrc get_base_address(void * main_ptr) {
  Dl_info info;
  int rc;
  link_map * map = (link_map *)malloc(1000*sizeof(link_map));
  void *start_ptr = (void*)map;
  // struct link_map map;
  rc = dladdr1(main_ptr, &info, (void**)&map, RTLD_DL_LINKMAP);
  // Store base pointer for lokal process
  // TODO: is the image base address necessary
  set_image_base_address(99, (intptr_t)info.dli_fbase);    
  // TODO: keep it simply for now and assume that target function is in main binary
  // If it is necessary to apply different behavior each loaded library has to be covered and analyzed
  //free(start_ptr);
  return TARGETDART_SUCCESS;
}