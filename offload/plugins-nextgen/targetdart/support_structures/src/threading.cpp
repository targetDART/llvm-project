#include "../include/threading.h"
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <omp.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <utility>
#include <vector>

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
tdrc TD_Thread_Manager::get_thread_placement_from_env(std::vector<int> &placements) {

    // Set the number of cores that TD should use
    // Prioritize env variable
    size_t nprocs = 0;
    if (auto env_nprocs = std::getenv("TD_EXECUTOR_NPROCS")) {
        nprocs = std::max(std::atoi(env_nprocs), 1);
        DP("executor nprocs assigned to %zu from env\n", nprocs);
    } else {
        cpu_set_t set;
        CPU_ZERO(&set);
        sched_getaffinity(0, sizeof(set), &set);
        // All available processors - 2 used for scheduler/receiver threads
        nprocs = CPU_COUNT(&set) - 2;
        DP("executor nprocs assigned to %zu\n", nprocs);
    }

    if (std::getenv("TD_MANAGEMENT") == NULL) {
        // scheduler
        placements[0] = 0;
        // receiver
        placements[1] = 1;
        // executors
        for (size_t i = 2; i < placements.size(); i++) {
            placements.at(i) = (i-2) % nprocs + 2;
            DP("Executor thread assigned core %d\n", placements[i]);
        }
        DP("Management threads assigned cores 0-%zu\n", std::min(placements.size()-1, nprocs));
        DP("For a parallel CPU execution use OMP_NUM_TEAMS with a value as high as possible.\n");
        return TARGETDART_SUCCESS;
    }

    std::string management = std::getenv("TD_MANAGEMENT");

    std::vector<std::string> assignments = split(management);

    placements.at(0) = std::stoi(assignments.at(0));
    DP("Scheduling thread assiged to core %d from env\n", placements[0]);
    placements.at(1) = std::stoi(assignments.at(1));
    DP("Receiver thread assiged to core %d from env\n", placements[1]);

    for (size_t i = 2; i < std::min( assignments.size(), placements.size()); i++) {
        placements.at(i) = std::stoi(assignments.at(i));
        DP("Execution thread %zu assiged to core %d from env\n", i - 1, placements[i]);
    }
    return TARGETDART_SUCCESS;
}

// pins 
template<typename... Args>
void __pin_and_workload(std::thread* thread, int core, std::function<void(Args...)> *work, Args... vargs) {
    if (core != -1) {
        int s;
        cpu_set_t cpuset;// = CPU_ALLOC(N);
        cpu_set_t old_cpuset;
        std::vector<int> possible_cores;

        CPU_ZERO(&cpuset);
        CPU_ZERO(&old_cpuset);
        sched_getaffinity(getpid(), sizeof(cpu_set_t), &old_cpuset);

        for (int i = 0; i < sysconf( _SC_NPROCESSORS_ONLN ); i++) {
            if (CPU_ISSET(i, &old_cpuset)) {
                possible_cores.push_back(i);
            }
        }

        int assignment = possible_cores.at(core);

        CPU_SET(assignment, &cpuset);

        // pin current thread to core
        // WARNING: Only works on Unix systems
        s = pthread_setaffinity_np(thread->native_handle(), sizeof(cpu_set_t), &cpuset);
        if (s != 0) 
            handle_error_en(s, "Failed to initialize thread\n");
    }

    //Do work
    (*work)(vargs...);
}


/*
* initializes the scheduling and executor threads.
* The threads are pinned to the cores defined in the parameters.
* the number of exec placements must be equal to the omp_get_num_devices() + 1.
*/
tdrc TD_Thread_Manager::init_threads(std::vector<int> const &assignments) {

    //DP("Creating %zu Threads\n", assignments->size());

    scheduler_th = std::thread(__pin_and_workload<int>, &scheduler_th, assignments[0], &schedule_thread_loop, -1);

    receiver_th = std::thread(__pin_and_workload<int>, &receiver_th, assignments[1], &receiver_thread_loop, -1);

    executor_th.resize(executors_per_device * (physical_device_count) + 1);
    DP("Creating %zu Threads\n", executor_th.size() + 2);

    //initialize all device offloading threads
    for (int i = 0; i < (int)executor_th.size() - 1; i++) {
        int device = i / executors_per_device;
        executor_th.at(i) = std::thread(__pin_and_workload<int, int>, &executor_th.at(i), assignments[i+2], &exec_thread_loop, device, i);
    }

    // initalize cpu executor
    int idx = executor_th.size() - 1;
    int cpu_device = idx / executors_per_device;
    executor_th.at(idx) = std::thread(__pin_and_workload<int, int>, &executor_th.at(idx), assignments[executor_th.size() - 1], &exec_thread_loop, cpu_device, idx);

    DP("spawned management threads\n");
    return TARGETDART_SUCCESS;
}

TD_Thread_Manager::TD_Thread_Manager(int32_t device_count, TD_Communicator *comm, TD_Scheduling_Manager *sched) {
    physical_device_count = device_count;

    schedule_man = sched;
    comm_man = comm;

    if (auto env_executors_per_device = std::getenv("TD_EXECUTORS_PER_DEVICE")) {
        executors_per_device = std::max(std::atoi(env_executors_per_device), 1);
    }
    DP("Starting %d executors per device\n", executors_per_device);

    schedule_thread_loop = [&] (int deviceID) {
        TRACE_START("sched_loop\n");
        int iter = 0;
        DP("Starting scheduler thread\n");
        while (comm_man->test_finalization(!schedule_man->is_empty() || !is_finalizing) && comm_man->size > 1) {
            if (iter == 80000 || schedule_man->do_repartition()) {
                iter = 0;
                // TODO: this differentiation kann lead to a Deadlock
                // TODO: restructure multi-schedule approaches
                //schedule_man->global_reschedule(CPU);
                //schedule_man->global_reschedule(GPU);
                //schedule_man->global_reschedule(ANY);
                schedule_man->reset_repatition();
                DP("ping\n");
            }
            iter++;        
            schedule_man->iterative_schedule(CPU);
            schedule_man->iterative_schedule(GPU);
            schedule_man->iterative_schedule(ANY);
            std::this_thread::sleep_for(std::chrono::microseconds(5));
            /*td_uid_t uid;
            if (comm_man->test_and_receive_results(&uid) == TARGETDART_SUCCESS) {
                schedule_man->notify_task_completion(uid, false);
            }*/
        }

        scheduler_done.store(true);
        TRACE_END("sched_loop\n");
        DP("Scheduling thread finished\n");  
    };

    receiver_thread_loop = [&] (int deviceID) {
        TRACE_START("recv_loop\n");
        DP("Starting Receiver thread\n");
        while ((!scheduler_done.load() || !schedule_man->is_empty()) && comm_man->size > 1) {
            td_uid_t uid;
            if (comm_man->test_and_receive_results(&uid) == TARGETDART_SUCCESS) {
                schedule_man->notify_task_completion(uid, false);
            }

            td_task_t *task = new td_task_t;
            bool keep = false;
            if (comm_man->test_and_receive_task(task) == TARGETDART_SUCCESS) {
                schedule_man->add_remote_task(task, task->affinity);
                keep = true;
            }

            if (!keep) {
                delete task;
            }
            //std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        TRACE_END("recv_loop\n");
    };

    exec_thread_loop = [&] (int deviceID, int executorID) {
        TRACE_START("exec_loop\n");
        int iter = 0;
        int phys_device_id = deviceID;
        if (deviceID == physical_device_count) {
            phys_device_id = schedule_man->total_device_count();
        }
        DP("Starting executor thread %d for device %d\n", executorID, phys_device_id);
        while (!scheduler_done.load() || !is_finalizing || !schedule_man->is_empty()) {
            td_task_t *task;
            iter++;
            if (iter == 8000000) {
                iter = 0;
                DP("ping from executor of device %d\n", deviceID);
                DP("remaining load %ld\n", schedule_man->get_active_tasks());
            }
            if (schedule_man->get_task(deviceID, &task) == TARGETDART_SUCCESS) {
                DP("start execution of task (%ld%ld) on device %d, executor %d\n", task->uid.rank, task->uid.id, phys_device_id, executorID);
                //execute the task on your own device
                int return_code = schedule_man->invoke_task(task, phys_device_id);
                task->return_code = return_code;
                /* if (return_code == TARGETDART_FAILURE) {         
                    //handle_error_en(-1, "Task execution failed.\n");
                    //exit(-1);
                } */
                //finalize after the task finished
                if (task->uid.rank != comm_man->rank) {
                    comm_man->send_task_result(task);
                    schedule_man->notify_task_completion(task->uid, false);
                    DP("finished remote execution of task (%ld%ld)\n", task->uid.rank, task->uid.id);
                    delete_task(task, false);
                } else {
                    schedule_man->notify_task_completion(task->uid, false);
                    DP("finished local execution of task (%ld%ld)\n", task->uid.rank, task->uid.id);
                    delete_task(task, true);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }    
        TRACE_END("exec_loop\n");
        DP("executor thread for device %d finished\n", deviceID);
    };

    // Physical devices aka executors * GPUs + CPU + Scheduling + Receiver
    std::vector<int> placements(executors_per_device * (physical_device_count) + 3);

    get_thread_placement_from_env(placements);

    init_threads(placements);
}


TD_Thread_Manager::~TD_Thread_Manager() {

    DP("begin finalization of targetDART, wait for remaining work\n");
    is_finalizing = true;  

    scheduler_th.join();

    receiver_th.join();

    DP("Synchronized threads start joining %zu management threads\n", executor_th.size());
    for (size_t i = 0; i < executor_th.size(); i++) { 
        DP("Joining executor: %zu\n", i);
        executor_th.at(i).join();
        DP("Joined executor: %zu\n", i);
    }

    DP("finalized management threads\n");
}
