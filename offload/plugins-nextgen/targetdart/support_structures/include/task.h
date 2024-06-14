#ifndef _TARGETDART_TASK_H
#define _TARGETDART_TASK_H


#include <cstddef>
#include <cstdint>
#include <string>
#include "omptarget.h"

#include "Shared/Environment.h"

// TODO: redefine affinity to enable combined devices
enum td_device_affinity {TD_CPU};

class td_uid_t{
private:
    size_t      id;
    size_t      rank;
public:
    td_uid_t();
    ~td_uid_t();

    size_t get_rank();
    size_t get_id();
    std::string toString();

    bool operator == (const td_uid_t &task) {
        if (id == task.id && rank == task.rank)
            return true;
        return false;
    }
};

typedef struct td_task_t{
    intptr_t            host_base_ptr;
    KernelArgsTy*       KernelArgs;
    int32_t             num_teams;
    int32_t             thread_limit;
    ident_t*            Loc;
    td_device_affinity  affinity;
    td_uid_t            uid;
    int                 return_code;
} td_task_t;



#endif //_TARGETDART_TASK_H