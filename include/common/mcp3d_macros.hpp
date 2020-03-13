//
// Created by muyezhu on 3/27/18.
//

#ifndef MCP3D_MCP3D_MACROS_HPP
#define MCP3D_MCP3D_MACROS_HPP

#ifdef SUPPORT_MPI
    #define MCP3D_MPI_BUILD 1
#else
    #define MCP3D_MPI_BUILD 0
#endif

#ifdef COMPILE_DLL
    #define IS_DLL 1
#else
    #define IS_DLL 0
#endif

#ifndef EIGEN_USE_THREADS
    #if !MCP3D_MPI_BUILD
        #define EIGEN_USE_THREADS
    #endif
#endif

#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#endif

#ifndef EIGEN_USE_LAPACKE
#define EIGEN_USE_LAPACKE
#endif

#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif

#ifndef EIGEN_UNALIGNED_VECTORIZE
#define EIGEN_UNALIGNED_VECTORIZE 1
#endif

#ifndef EIGEN_UNROLLING_LIMIT
#define EIGEN_UNROLLING_LIMIT 100
#endif

#if BOOST_OS_WINDOWS && IS_DLL
    #define MCP3D_EXPORT __declspec(dllexport)
#elif BOOST_OS_WINDOWS
#define MCP3D_EXPORT __declspec(dllimport)
#elif defined __GNUC__ && __GNUC__ >= 4
    #define MCP3D_EXPORT __attribute__ ((visibility ("default")))
#else
    #define MCP3D_EXPORT
#endif

#include <type_traits>
#include <chrono>
#include <omp.h>
#include <boost/predef.h>
#include "common/mcp3d_exceptions.hpp"

static_assert(std::is_nothrow_copy_constructible<mcp3d::MCPRuntimeError>::value,
              "MCPRuntimeError is not nothrow copy constructible");

static_assert(std::is_nothrow_copy_constructible<mcp3d::MCPInvalidArgument>::value,
              "MCPInvalidArgument is not nothrow copy constructible");

static_assert(std::is_nothrow_copy_constructible<mcp3d::MCPTestError>::value,
              "MCPTestError is not nothrow copy constructible");

static_assert(std::is_nothrow_copy_constructible<mcp3d::MCPDomainError>::value,
              "MCPDomainError is not nothrow copy constructible");

static_assert(std::is_nothrow_copy_constructible<mcp3d::MCPRuntimeError>::value,
              "MCPRuntimeError is not nothrow copy constructible");

static_assert(std::is_nothrow_copy_constructible<mcp3d::MCPBadAlloc>::value,
              "MCPBadAlloc is not nothrow copy constructible");

static_assert(std::is_nothrow_copy_constructible<mcp3d::MCPAssertionError>::value,
              "MCPAssertionError is not nothrow copy constructible");

static_assert(std::is_nothrow_copy_constructible<mcp3d::MCPImageFormatMismatchError>::value,
              "MCPImageFormatMismatchError is not nothrow copy constructible");

static_assert(std::is_nothrow_copy_constructible<mcp3d::MCPImageFormatUnsupportedError>::value,
              "MCPImageFormatUnsupportedError is not nothrow copy constructible");

#define MCP3D_MESSAGE(arg) {                                                                              \
    ::std::cout << __FILE__ << ":" << __LINE__ << ": " << arg << ::std::endl;      \
}

#define MCP3D_RUNTIME_ERROR(arg) {                                                \
    throw ::mcp3d::MCPRuntimeError(arg, __FILE__, __LINE__, __func__);            \
}

#define MCP3D_OS_ERROR(arg) {                                                     \
    throw ::mcp3d::MCPOSError(arg, __FILE__, __LINE__, __func__);                 \
}

#define MCP3D_INVALID_ARGUMENT(arg) {                                             \
    throw ::mcp3d::MCPInvalidArgument(arg, __FILE__, __LINE__, __func__);         \
}

#define MCP3D_DOMAIN_ERROR(arg) {                                                 \
    throw ::mcp3d::MCPDomainError(arg, __FILE__, __LINE__, __func__);             \
}

#define MCP3D_OUT_OF_RANGE(arg) {                                                 \
    throw ::mcp3d::MCPOutOfRangeError(arg, __FILE__, __LINE__, __func__);         \
}

#define MCP3D_BAD_ALLOC(arg) {                                                    \
    throw ::mcp3d::MCPBadAlloc(arg, __FILE__, __LINE__, __func__);                \
}

#define MCP3D_ASSERT(expr) {                                                      \
    if (expr);                                                                    \
    else throw ::mcp3d::MCPAssertionError(#expr, __FILE__, __LINE__, __func__);   \
}

#define MCP3D_IMAGE_FORMAT_UNSUPPORTED(arg) {                                                     \
    throw ::mcp3d::MCPImageFormatUnsupportedError(arg, __FILE__, __LINE__, __func__);             \
}

#define MCP3D_PRINT_NESTED_EXCEPTION {                                                                  \
    ::mcp3d::PrintNestedException(::std::current_exception(), __FILE__, __LINE__, __func__);      \
}

/* Intended to be used within a catch block. Will throw as nested exception,
 * used by mcp3d::PrintNested to print an exception trace
 */
#define MCP3D_RETHROW(e) {                                      \
    ::mcp3d::ReThrow(e, __FILE__, __LINE__, __func__);          \
}

#define MCP3D_TRY(statement) {                                    \
    try                                                           \
    {                                                             \
        statement                                                 \
    }                                                             \
    catch (...)                                                   \
    {                                                             \
        MCP3D_RETHROW(::std::current_exception())                 \
    }                                                             \
}

#define MCP3D_MACROS_INTERNAL

#include "mcp3d_macros_local.hpp"
#include "mcp3d_macros_mpi.hpp"

#undef MCP3D_MACROS_INTERNAL

#endif //MCP3D_MCP3D_MACROS_HPP
