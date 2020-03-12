//
// Created by muyezhu on 3/27/18.
//
#ifndef MCP3D_MACROS_INTERNAL
#error mcp3d_macros_mpi.hpp must be included through mcp3d_macros.hpp
#endif

#if MCP3D_MPI_BUILD

#ifndef MCP3D_MCP3D_MACROS_MPI_HPP
#define MCP3D_MCP3D_MACROS_MPI_HPP

#include <mpi.h>

inline void __FinalizeMPI__()
{
    int __mpi__;
    MPI_Initialized(&__mpi__);
    if (__mpi__)
        MPI_Finalize();
}

/* prepend exception throwing with MPI environment termination
 * otherwise under parallel execution mpirun / mpiexec may not terminate
 * even when individual process has terminated */
#define MCP3D_RUNTIME_ERROR_FINALIZE(arg) {                                                \
    __FinalizeMPI__();                                                            \
    throw ::mcp3d::MCPRuntimeError(arg, __FILE__, __LINE__, __func__);            \
}
#define MCP3D_OS_ERROR_FINALIZE(arg) {                                                     \
    __FinalizeMPI__();                                                            \
    throw ::mcp3d::MCPOSError(arg, __FILE__, __LINE__, __func__);                 \
}

#define MCP3D_TEST_ERROR_FINALIZE(arg) {                                                   \
    __FinalizeMPI__();                                                            \
    throw ::mcp3d::MCPTestError(arg, __FILE__, __LINE__, __func__);               \
}

#define MCP3D_INVALID_ARGUMENT_FINALIZE(arg) {                                             \
    __FinalizeMPI__();                                                            \
    throw ::mcp3d::MCPInvalidArgument(arg, __FILE__, __LINE__, __func__);         \
}
#define MCP3D_DOMAIN_ERROR_FINALIZE(arg) {                                                 \
    __FinalizeMPI__();                                                            \
    throw ::mcp3d::MCPDomainError(arg, __FILE__, __LINE__, __func__);             \
}

#define MCP3D_OUT_OF_RANGE_FINALIZE(arg) {                                                 \
    __FinalizeMPI__();                                                            \
    throw ::mcp3d::MCPOutOfRangeError(arg, __FILE__, __LINE__, __func__);         \
}

#define MCP3D_BAD_ALLOC_FINALIZE(arg) {                                                    \
    __FinalizeMPI__();                                                            \
    throw ::mcp3d::MCPBadAlloc(arg, __FILE__, __LINE__, __func__);                \
}

#define MCP3D_ASSERT_FINALIZE(expr) {                                                      \
    __FinalizeMPI__();                                                            \
    if (expr);                                                                    \
    else throw ::mcp3d::MCPAssertionError(#expr, __FILE__, __LINE__, __func__);   \
}

#define MCP3D_IMAGE_FORMAT_MISMATCH_FINALIZE(arg) {                                                        \
    __FinalizeMPI__();                                                                            \
    throw ::mcp3d::MCPImageFormatMismatchError(arg, __FILE__, __LINE__, __func__);                \
}

#define MCP3D_IMAGE_FORMAT_UNSUPPORTED_FINALIZE(arg) {                                                     \
    __FinalizeMPI__();                                                                            \
    throw ::mcp3d::MCPImageFormatUnsupportedError(arg, __FILE__, __LINE__, __func__);             \
}

/* convenient defines for timing. saves some messy repetitive lines
 * id: multiple sets of timers are sometimes needed, in which case each
 * time point pairs should have a unique id. potential use of this is a timer
 * in a larger block with another timer in a nested block */
// internal, do not use outside of mcp3d_def.hpp
#define __DEC_TIMEPOINT std::chrono::high_resolution_clock::time_point
#define __TIMESTART(id) __mcp3d_macro_clock_start_ ## id
#define __TIMEEND(id) __mcp3d_macro_clock_end_ ## id
#define __WORLD_BARRIER MPI_Barrier(MPI_COMM_WORLD);
#define __COMM_BARRIER(comm) MPI_Barrier((comm));

/* intended as public "API"
 * {
 *     // timers have block scope
 *     // decalare 2 time_point objects: __mcp3d_macro_clock_start_${id}
 *     //                                __mcp3d_macro_clock_end_${id}
 *     INIT_TIMER(id)
 *     ...
 *     TIC(id) or (TIC_WORLD(id) / TIC_COMM(id) in parallel environment)
 *     {
 *         code to be timed
 *     }
 *     TOC(id) or (TOC_WORLD(id) / TOC_COMM(id) in parallel environment)
 *     // if want to extract elapse value
 *     double elapse = ELAPSE
 *     // print: msg: ELAPSE seconds
 *     (MPI)_REPORT_TIME_TO_COMPLETION((rank), msg, id)
 * }
 */
#define INIT_TIMER(id) __DEC_TIMEPOINT __TIMESTART(id), __TIMEEND(id);
#define TIC(id) __TIMESTART(id) = std::chrono::high_resolution_clock::now();
#define TOC(id) __TIMEEND(id) = std::chrono::high_resolution_clock::now();
#define TIC_WORLD(id) __WORLD_BARRIER TIC(id)
#define TOC_WORLD(id) __WORLD_BARRIER TOC(id)
#define TIC_COMM(comm, id) __COMM_BARRIER(comm) TIC(id)
#define TOC_COMM(comm, id) __COMM_BARRIER(comm) TOC(id)
#define ELAPSE(id) std::chrono::duration_cast<std::chrono::duration<double>>(__TIMEEND(id) - __TIMESTART(id)).count()
inline void PrintElapse(const std::string &step, double elapse)
{
    std::cout << step << ": " << elapse << " seconds" << std::endl;
}
#define REPORT_TIME_TO_COMPLETION(msg, id) PrintElapse(msg, ELAPSE(id));
#define MPI_REPORT_TIME_TO_COMPLETION(r, msg, id) if (r == 0) PrintElapse(msg, ELAPSE(id));

#define CHECK_PARALLEL_MODEL    {                                      \
    int __mpi__;                                                       \
    MPI_Initialized(&__mpi__);                                         \
    int __n_threads__ = ::omp_get_num_threads();                       \
    if (__mpi__ && __n_threads__ > 1)                                  \
        MCP3D_RUNTIME_ERROR("can not use both mpi and openmp");        \
}

#define RANK0_CALL(function, ...)  {                                   \
    CHECK_PARALLEL_MODEL                                               \
    int __mpi__;                                                           \
    MPI_Initialized(&__mpi__);                                             \
    if (__mpi__)                                                           \
    {                                                                  \
        int __rank__;                                                      \
        MPI_Comm_rank(MPI_COMM_WORLD, &__rank__);                          \
        if (__rank__ == 0)                                                 \
            function(__VA_ARGS__);                                     \
    }                                                                  \
    if (omp_get_thread_num() == 0)                                     \
        function(__VA_ARGS__);                                         \
}

inline void __omp_barrier__()
{
#pragma omp barrier
}

#define RANK0_CALL_SYNC(function, ...)  {                              \
    CHECK_PARALLEL_MODEL                                               \
    int __mpi__;                                                           \
    MPI_Initialized(&__mpi__);                                             \
    if (__mpi__)                                                           \
    {                                                                  \
        int __rank__;                                                      \
        MPI_Comm_rank(MPI_COMM_WORLD, &__rank__);                          \
        if (__rank__ == 0)                                                 \
            function(__VA_ARGS__);                                     \
        MPI_Barrier(MPI_COMM_WORLD);                                   \
    }                                                                  \
    else if (omp_get_num_threads() == 1)                                    \
        function(__VA_ARGS__);                                         \
    else                                                               \
    {                                                                  \
        __omp_barrier__();                                             \
        if (omp_get_thread_num() == 0)                                 \
            function(__VA_ARGS__);                                     \
        __omp_barrier__();                                             \
    }                                                                  \
}

#define RANK0_COMM_CALL_SYNC(function, comm, ...)  {                              \
    CHECK_PARALLEL_MODEL                                               \
    int __mpi__;                                                           \
    MPI_Initialized(&__mpi__);                                             \
    if (__mpi__)                                                           \
    {                                                                  \
        int __rank__;                                                      \
        MPI_Comm_rank(comm, &__rank__);                          \
        if (__rank__ == 0)                                                 \
            function(__VA_ARGS__);                                     \
        MPI_Barrier(comm);                                   \
    }                                                                  \
    else if (omp_get_num_threads() == 1)                                    \
        function(__VA_ARGS__);                                         \
    else                                                               \
    {                                                                  \
        __omp_barrier__();                                             \
        if (omp_get_thread_num() == 0)                                 \
            function(__VA_ARGS__);                                     \
        __omp_barrier__();                                             \
    }                                                                  \
}


#define CALL_SYNC(function, ...)  {                                    \
    CHECK_PARALLEL_MODEL                                               \
    int mpi;                                                           \
    MPI_Initialized(&mpi);                                             \
    if (mpi == 1)                                                      \
    {                                                                  \
        MPI_Barrier(MPI_COMM_WORLD);                                   \
        function(__VA_ARGS__);                                         \
        MPI_Barrier(MPI_COMM_WORLD);                                   \
    }                                                                  \
    else                                                               \
    {                                                                  \
        __omp_barrier__();                                             \
        function(__VA_ARGS__);                                         \
        __omp_barrier__();                                             \
    }                                                                  \
}

#define CALL_IN_MPI_BUILD(function, ...) {                             \
    function(__VA_ARGS__);                                             \
}

#define SERIALIZE_OPENCV_MPI {                                         \
    int mpi_cv_serial_;                                                \
    MPI_Initialized(&mpi_cv_serial_);                                  \
    if (mpi_cv_serial_ == 1)                                           \
        cv::setNumThreads(0);                                          \
}


#endif //MCP3D_MCP3D_MACROS_MPI_HPP

#endif //LONICLUSTER
