#include "TD_CrashHandling.h"

#include <signal.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>
#include <execinfo.h>
#include <vector>
#include <mutex>
#include "TD_common.h"

#ifndef _EXTERN_C_
#ifdef __cplusplus
#define _EXTERN_C_ extern "C"
#else /* __cplusplus */
#define _EXTERN_C_
#endif /* __cplusplus */
#endif /* _EXTERN_C_ */

typedef void (*sighandler_t)(int);

static void set_signalhandlers(sighandler_t handler)
{
    signal(SIGSEGV, handler);
    signal(SIGINT, handler);
    signal(SIGHUP, handler);
    signal(SIGABRT, handler);
    signal(SIGTERM, handler);
    signal(SIGUSR2, handler);
    signal(SIGQUIT, handler);
    signal(SIGALRM, handler);
}

__attribute__((destructor)) static void disable_signalhandlers() { set_signalhandlers(SIG_DFL); }

#define CALLSTACK_SIZE 20

static void print_stack(void)
{
    int nptrs;
    void* buf[CALLSTACK_SIZE + 1];

    nptrs = backtrace(buf, CALLSTACK_SIZE);

    backtrace_symbols_fd(buf, nptrs, STDOUT_FILENO);
}

/**
 * Handler that is called on signals.
 *
 * @param signum the signal number
 */
void mySignalHandler(int signum)
{
    disable_signalhandlers();
    printf(
        "rank %i (of %i), pid %i caught signal nr %i\n",
        td_comm_rank,
        td_comm_size,
        getpid(),
        signum);
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (signum == SIGINT || signum == SIGKILL) {
        print_stack();
        if (!static_cast<bool>(finalized)) {
            MPI_Abort(MPI_COMM_WORLD, signum + 128);
        } else {
            _exit(signum + 128);
        }
    }
    if (signum == SIGTERM || signum == SIGUSR2) {
            print_stack();
            fflush(stdout);
            sleep(1);
        if (!static_cast<bool>(finalized)) {
            MPI_Abort(MPI_COMM_WORLD, signum + 128);
        } else {
            _exit(signum + 128);
        }
    }
    print_stack();

    _exit(1);
}

void initHandler(void) {

    static auto inited = false;
    if (inited) {
        return;
    }
    inited = true;

    set_signalhandlers(mySignalHandler);

}