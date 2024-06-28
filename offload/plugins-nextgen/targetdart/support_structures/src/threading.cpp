#include "../include/threading.h"
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <string>
#include <iostream>

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

// splits a String by \n"," or ";"
std::vector<std::string> split(std::string base) {
	char delimiter = ',';
	std::vector<std::string> res;

	if (base.find(';') != std::string::npos) {
		delimiter = ';';
	}

	bool mode = true;

	long long start = 0;
	long long length = 0;
	for ( size_t i = 0; i < base.size(); i++) {
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
tdrc TD_Thread_Manager::get_thread_placement_from_env(std::vector<int> *placements) {
    if (std::getenv("TD_MANAGEMENT") == NULL) {
        for (size_t i = 0; i < placements->size(); i++) {
            placements->at(i) = i;
        }
        DP("Management threads assigned cores 0-%zu, use OMP_PLACES=%zu:num_threads\n", placements->size()-1, placements->size());
        return TARGETDART_FAILURE;
    }

    std::string management = std::getenv("TD_MANAGEMENT\n");

    std::vector<std::string> assignments = split(management);

    placements->at(0) = std::stoi(assignments.at(0));
    DP("Scheduling thread assiged to core %d from env\n", (*placements)[0]);

    for (size_t i = 1; i < std::min( assignments.size(), placements->size()); i++) {
        placements->at(i) = std::stoi(assignments.at(i));
        DP("Execution thread %zu assiged to core %d from env\n", i - 1, (*placements)[i]);
    }
    return TARGETDART_SUCCESS;
}

// pins 
void __pin_and_workload(std::thread* thread, int core, std::function<void(int)> *work, int deviceID) {
    if (core != -1) {
    
        cpu_set_t cpuset;// = CPU_ALLOC(N);

        int s;

        CPU_ZERO(&cpuset);
        CPU_SET(core, &cpuset);

        // pin current thread to core
        // WARNING: Only works on Unix systems
        s = pthread_setaffinity_np(thread->native_handle(), sizeof(cpu_set_t), &cpuset);
        if (s != 0) 
            handle_error_en(s, "Failed to initialize thread\n");
    }

    //Do work
    (*work)(deviceID);
}


/*
* initializes the scheduling and executor threads.
* The threads are pinned to the cores defined in the parameters.
* the number of exec placements must be equal to the omp_get_num_devices() + 1.
*/
tdrc TD_Thread_Manager::init_threads(std::vector<int> *assignments) {

    DP("Creating %zu Threads\n", assignments->size());

    scheduler_th = std::thread(__pin_and_workload, &scheduler_th, (*assignments)[0], &schedule_thread_loop, -1);

    executor_th.resize(physical_device_count + 1);

    //initialize all offloading threads
    for (int i = 0; i <= physical_device_count; i++) {
        executor_th.at(i) = std::thread(__pin_and_workload, &executor_th.at(i), (*assignments)[i+1], &exec_thread_loop, i);
    }

    DP("spawned management threads\n");
    return TARGETDART_SUCCESS;
}

TD_Thread_Manager::TD_Thread_Manager(int32_t device_count, TD_Communicator *comm, TD_Scheduling_Manager *sched) {
    physical_device_count = device_count;

    schedule_man = sched;
    comm_man = comm;

    schedule_thread_loop = [&] (int deviceID) {
        int iter = 0;
        DP("Starting scheduler thread\n");
        while (comm_man->test_finalization(!schedule_man->is_empty() || !is_finalizing) && comm_man->size > 1) {
            if (iter == 800000 || schedule_man->do_repartition()) {
                iter = 0;
                // TODO: this differentiation kann lead to a Deadlock
                // TODO: restructure multi-schedule approaches
                //td_global_reschedule(TD_ANY);
                schedule_man->reset_repatition();
                DP("ping\n");
            }
            iter++;        
            schedule_man->iterative_schedule(ANY);
            td_uid_t uid;
            if (comm_man->test_and_receive_results(&uid) == TARGETDART_SUCCESS) {
                schedule_man->notify_task_completion(uid, false);
            }
        }

        scheduler_done.store(true);
        DP("Scheduling thread finished\n");  
    };

    exec_thread_loop = [&] (int deviceID) {
        DP("Starting executor thread for device %d\n", deviceID);
        int iter = 0;
        int phys_device_id = deviceID;
        if (deviceID == physical_device_count) {
            phys_device_id = -1;
        }
        while (!scheduler_done.load() || !is_finalizing || !schedule_man->is_empty()) {
            td_task_t *task;
            iter++;
            if (iter == 8000000) {
                iter = 0;
                DP("ping from executor of device %d\n", deviceID);
            }
            if (schedule_man->get_task(deviceID, &task) == TARGETDART_SUCCESS) {
                DP("start execution of task (%ld%ld)\n", task->uid.rank, task->uid.id);
                //execute the task on your own device
                int return_code = invoke_task(task, phys_device_id);
                task->return_code = return_code;
                /* if (return_code == TARGETDART_FAILURE) {         
                    //handle_error_en(-1, "Task execution failed.\n");
                    //exit(-1);
                } */
                //finalize after the task finished
                if (task->uid.rank != comm_man->rank) {
                    comm_man->send_task_result(task);
                    schedule_man->notify_task_completion(task->uid, false);
                    //free(task);
                    DP("finished remote execution of task (%ld%ld)\n", task->uid.rank, task->uid.id);
                } else {
                    schedule_man->notify_task_completion(task->uid, false);
                    //free(task);
                    DP("finished local execution of task (%ld%ld)\n", task->uid.rank, task->uid.id);
                }
            } 
        }    

        DP("executor thread for device %d finished\n", deviceID);
    };

    // Physical devices aka GPUs + CPU + Scheduling
    std::vector<int> placements(physical_device_count + 2);

    get_thread_placement_from_env(&placements);

    init_threads(&placements);
}


TD_Thread_Manager::~TD_Thread_Manager() {
    //TODO: sync and clean up
}