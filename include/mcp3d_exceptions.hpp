//
// Created by muyezhu on 5/14/17.
//

#ifndef MCP3D_MCP3D_EXCEPTIONS_HPP
#define MCP3D_MCP3D_EXCEPTIONS_HPP

#include <stdexcept>
#include <exception>
#include <iostream>
#include <string>
#include <system_error>
#include <mutex>

namespace mcp3d
{
/*
 * see link for rational to use private exception member
 * https://wiki.sei.cmu.edu/confluence/display/cplusplus/ERR60-CPP.+Exception+objects+must+be+nothrow+copy+constructible
 */
class MCPRuntimeError: public std::exception
{
public:
    MCPRuntimeError(const std::string& arg, const char* file,
                        int line, const char* func):
            e("MCPRuntimeError at " + std::string(file) +
              " line " + std::to_string(line) +
              ", function " + std::string(func) + ":\n" + arg) {}
    ~MCPRuntimeError() noexcept override = default;
    const char *what() const noexcept override {return e.what();};
private:
    std::runtime_error e;
};

class MCPOSError: public std::exception
{
public:
    MCPOSError(const std::string& arg, const char* file,
                    int line, const char* func):
            e("MCPOSError at " + std::string(file) +
              " line " + std::to_string(line) +
              ", function " + std::string(func) + ":\n" + arg) {}
    ~MCPOSError() noexcept override = default;
    const char *what() const noexcept override {return e.what();};
private:
    std::runtime_error e;
};

class MCPAssertionError: public std::exception
{
public:
    MCPAssertionError(const std::string& arg, const char* file,
                          int line, const char* func):
            e("MCPAssertionError at " + std::string(file) +
              " line " + std::to_string(line) +
              ", function " + std::string(func) + ":\n" +
              "assertion: " + arg + " failed\n") {}
    ~MCPAssertionError() noexcept override = default;
    const char *what() const noexcept override {return e.what();};
private:
    std::runtime_error e;
};

class MCPTestError: public std::exception
{
public:
    MCPTestError(const std::string& arg, const char* file,
                     int line, const char* func):
              e("MCPTestError at " + std::string(file) +
                " line " + std::to_string(line) +
                ", function " + std::string(func) + ":\n" + "test failed: "
                + arg + "\n") {}
    ~MCPTestError() noexcept override = default;
    const char *what() const noexcept override {return e.what();};
private:
    std::runtime_error e;
};

class MCPDomainError: public std::exception
{
public:
    MCPDomainError(const std::string& arg, const char* file,
                       int line, const char* func):
              e("MCPDomainError at " + std::string(file) +
                " line " + std::to_string(line) +
                ", function " + std::string(func) + ":\n" + arg) {}

    ~MCPDomainError() noexcept override = default;
    const char *what() const noexcept override {return e.what();};
private:
    std::domain_error e;
};

class MCPInvalidArgument: public std::exception
{
public:
    MCPInvalidArgument(const std::string& arg, const char* file,
                           int line, const char* func):
              e("MCPInvalidArgument at " + std::string(file) +
                " line " + std::to_string(line) +
                ", function " + std::string(func) + ":\n" + arg) {}
    ~MCPInvalidArgument() noexcept override = default;
    const char *what() const noexcept override {return e.what();};
private:
    std::invalid_argument e;
};

class MCPOutOfRangeError: public std::exception
{
public:
    MCPOutOfRangeError(const std::string& arg, const char* file,
                       int line, const char* func):
              e("MCPOutOfRangeError at " + std::string(file) +
                " line " + std::to_string(line) +
                ", function " + std::string(func) + ":\n" + arg) {}
    ~MCPOutOfRangeError() noexcept override = default;
    const char *what() const noexcept override {return e.what();};
private:
    std::out_of_range e;
};

class MCPBadAlloc: public std::bad_alloc
{
public:
    MCPBadAlloc(const std::string &arg, const char *file,
                int line, const char *func) :
            e("MCPBadAlloc at " + std::string(file) +
              " line " + std::to_string(line) +
              ", function " + std::string(func) + ":\n" + arg) {}

    ~MCPBadAlloc() noexcept override = default;

    const char *what() const noexcept override { return e.what(); };
private:
    std::runtime_error e;
};

class MCPImageFormatMismatchError: public std::exception
{
public:
    MCPImageFormatMismatchError(const std::string& arg, const char* file,
                                int line, const char* func):
            e("MCPImageFormatMismatchError at " + std::string(file) +
              " line " + std::to_string(line) +
              ", function " + std::string(func) + ":\n" + arg) {}
    ~MCPImageFormatMismatchError() noexcept override = default;
    const char *what() const noexcept override {return e.what();};
private:
    std::runtime_error e;
};

class MCPImageFormatUnsupportedError: public std::exception
{
public:
    MCPImageFormatUnsupportedError(const std::string& arg, const char* file,
                                   int line, const char* func):
            e("MCPImageFormatUnsupportedError at " + std::string(file) +
              " line " + std::to_string(line) +
              ", function " + std::string(func) + ":\n" + arg) {}
    ~MCPImageFormatUnsupportedError() noexcept override = default;
    const char *what() const noexcept override {return e.what();};
private:
    std::runtime_error e;
};

class MultiThreadExceptions
{
public:
    MultiThreadExceptions(): e_ptr_(nullptr) {}

    void CaptureException();

    bool HasCapturedException() const  { return e_ptr_ != nullptr; }

    template <typename Function, typename... Parameters>
    void RunAndCaptureException(Function f, Parameters... params);

    std::exception_ptr e_ptr()  { return e_ptr_; }

private:
    std::exception_ptr e_ptr_;
    std::mutex lock_;
};

void PrintNested(const std::exception &e,
                 std::ostream &out = std::cout, int level = 0);

void PrintNestedException(const std::exception_ptr &eptr,
                          const char *file, int line,
                          const char *function,
                          std::ostream &out = std::cout);

void ReThrow(const std::exception_ptr& eptr, const char *file,
             int line, const char *function);

}

template <typename Function, typename... Parameters>
void mcp3d::MultiThreadExceptions::RunAndCaptureException(Function f, Parameters... params)
{
    try
    {
        f(params...);
    }
    catch (...)
    {
        CaptureException();
    }
}

#endif //MCP3D_MCP3D_EXCEPTIONS_HPP
