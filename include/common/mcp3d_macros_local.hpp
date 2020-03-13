//
// Created by muyezhu on 9/19/17.
//
#ifndef MCP3D_MACROS_INTERNAL
#error mcp3d_macros_local.happ must be included through mcp3d_macros.hpp
#endif

#if !MCP3D_MPI_BUILD
    #ifndef MCP3D_MCP3D_MACROS_LOCAL_HPP
        #define MCP3D_MCP3D_MACROS_LOCAL_HPP

        /* convenient defines for timing. saves some messy repetitive lines
         * id: multiple sets of timers are sometimes needed, in which case each
         * time point pairs should have a unique id. potential use of this is a timer
         * in a larger block with another timer in a nested block */
        // internal, do not use outside of mcp3d_def.hpp
        #define __DEC_TIMEPOINT std::chrono::high_resolution_clock::time_point
        #define __TIMESTART(id) __mcp3d_macro_clock_start_ ## id
        #define __TIMEEND(id) __mcp3d_macro_clock_end_ ## id

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
        #define ELAPSE(id) std::chrono::duration_cast<std::chrono::duration<double>>(__TIMEEND(id) - __TIMESTART(id)).count()
        inline void PrintElapse(const std::string &step, double elapse)
        {
            std::cout << step << ": " << elapse << " seconds" << std::endl;
        }
        #define REPORT_TIME_TO_COMPLETION(msg, id) PrintElapse(msg, ELAPSE(id));

        #define RANK0_CALL(function, ...)  {                                   \
            if (omp_get_thread_num() == 0)                                     \
                function(__VA_ARGS__);                                         \
        }

        inline void __omp_barrier__()
        {
        #pragma omp barrier
        }

        #define CHECK_PARALLEL_MODEL

        #define RANK0_CALL_SYNC(function, ...)  {                              \
            if (omp_get_num_threads() == 1)                                    \
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
            __omp_barrier__();                                             \
            function(__VA_ARGS__);                                         \
            __omp_barrier__();                                             \
        }

        inline void ErrorCallingMPIInNoMPIBuild()
        {
            MCP3D_RUNTIME_ERROR("function requiring MPI called in build with no MPI support")
        }

        #define CALL_IN_MPI_BUILD(function, ...) ErrorCallingMPIInNoMPIBuild();

        #define SERIALIZE_OPENCV_MPI

    #endif //MCP3D_MCP3D_MACROS_LOCAL_HPP

#endif //MPIPARALLEL
