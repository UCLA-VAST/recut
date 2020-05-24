//
// Created by mzhu on 3/13/17.
//

#ifndef MCP3D_MCP3D_UTILITY_HPP
#define MCP3D_MCP3D_UTILITY_HPP

#include "mcp3d_macros.hpp"
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>

namespace mcp3d {
// given a directory with an underlying tree of files and sub directories,
// a DFSDirectoryWalker list contents in a subdirectory somewhere in the tree
// structure by TerminalFiles() API when:
// (1) no deeper level contents from any items in the subdirectory exist or
// (2) if such contents exists, they have been list earlier by the walker
// path to the subdirectory is provided to caller by TerminalDirPath() API
// does not do read_symlink, path to the symlink will be returned. caller can
// resolve symlink itself
class DFSDirectoryWalker {
#if BOOST_OS_LINUX
public:
  DFSDirectoryWalker() = delete;
  // sort_alpha is optional argument to mcp3d::ListDir. mostly included here
  // for easy testing
  explicit DFSDirectoryWalker(const std::string &root_dir,
                              bool sort_alpha = false);
  // terminal directory:
  // (1) empty or (2) contain only files and no other directory or
  // (3) has subdirectories but all of them already expanded
  // terminals are reached in DFS order, with tree structure based on
  // tree_stack_ tree nodes are only expandable at directories, which are always
  // placed at the front of each tree depth. if a tree level has all its
  // terminal directories visited, the next ExpandNextTerminal() place all files
  // as well as directories at same depth in files_in_terminal_dir. and view the
  // parent dir of current level as terminal directory
  void ExpandNextTerminal();
  bool Exausted() { return ConstructTerminalDirPath().empty(); }
  std::string &TerminalFiles() { return files_in_terminal_dir_; }
  // path to terminal directory
  std::string TerminalDirPath() { return ConstructTerminalDirPath(); }
  // return number of sub directories under TerminalDirPath()
  int TerminalDirCounts();
  int TerminalEntryCounts();

private:
  // ith entry at level, starting from left
  std::string EntryAtTreePosition(int level, int i = 0);
  std::string ConstructTerminalDirPath();
  // number of dir/file in the string dir1\ndir2\nfile1
  int CountEntries(const std::string &contents);
  bool sort_alpha_;
  std::string tree_root_;
  // first in first expanded, directories are placed at front of each vector
  // any vector<string> element must be expandable (aka contain at least one
  // directory)
  std::vector<std::string> tree_stack_;
  // true false mark if dirs have been expanded. does not record files
  std::vector<int> tree_path_to_frontier_dir_, terminal_dir_counts_,
      terminal_entry_counts_;
  std::string files_in_terminal_dir_;
};
#endif

/// filter regular files in given directory
/// \param directory: input directory
/// \param files: output vector, contain full file paths
/// \param full_path: if to return full path or basename
/// \param sort_files: if the file paths will be sorted
/// \param incl: file path must contain strings in incl
/// \param n: number of file paths to place in files vector.
///           if -1 or greater than total number of files in directory,
///           return all file paths
/// \param excl: exclude file path containing strings in excl
MCP3D_EXPORT
std::vector<std::string>
FilesInDir(const std::string &directory, bool full_path = true,
           bool sort_files = true,
           const std::vector<std::string> &incl = std::vector<std::string>(),
           int n = -1,
           const std::vector<std::string> &excl = std::vector<std::string>());

MCP3D_EXPORT
std::vector<std::string> FilesEndsWithInDir(const std::string &directory,
                                            const std::string &end_str,
                                            bool full_path = true,
                                            bool sort_files = true, int n = -1);

MCP3D_EXPORT
std::vector<std::string>
DirsInDir(const std::string &directory_, bool full_path = true,
          bool sort_files = true,
          const std::vector<std::string> &incl = std::vector<std::string>(),
          int n = -1,
          const std::vector<std::string> &excl = std::vector<std::string>());

bool IsFile(const std::string &file_path);

bool IsFile(const filesystem::path &file_path);

bool IsDir(const std::string &dir_path);

bool IsEmptyDir(const std::string &dir_path);

filesystem::path ResolveSymlink(const std::string &path_);

void AllFilesExist(const std::vector<std::string> &file_paths);

char Separator();

/// return parent directory of file_path.
/// refer here:
/// http://www.boost.org/doc/libs/1_53_0/libs/filesystem/doc/reference.html#Path-decomposition-table
/// trailing / will be removed before calling boost path decomposition
std::string ParentDir(const std::string &file_path);

/// return the first directory component in a path. '/' is considered global
/// file system root directory. if a file path starts with '/', '/' will be
/// returned. for example, /foo -> /. otherwise expected output as following
/// foo/ -> foo
/// // -> /
/// ./foo/bar -> .
std::string FirstDir(const std::string &file_path);

/// trailing / will be removed before calling boost path decomposition
std::string Basename(const std::string &path_);

std::vector<std::string> Basenames(const std::vector<std::string> &paths);

/// return the pair: file parent dir and basename
std::pair<std::string, std::string> SplitBaseName(const std::string &file_path);

/// return the pair: root dir and remainder of path
std::pair<std::string, std::string> SplitFirstDir(const std::string &file_path);

/// return file name with extension romoved: a.txt -> a
std::string RemoveFileExt(const std::string &file_name);

/// return file extension: a.txt -> .txt
std::string FileExt(const std::string &file_name);

/// create directory dir_path and all necessary parent directories
/// does not throw error if directory already exist.
void MakeDirectories(const std::string &dir_path);

/// remove path if exists. does not throw error if path does not exist
/// path_ can be directory or file
void RemovePath(const std::string &path_);

/// list all directory items, excluding . and ..
/// optionally sort items alphabetically, note that hidden files such as .git
/// will be sorted as if its name is git. if list_directories_first is true,
/// directories will appear before any files. this will force sort_alpha to true
/// if reverse is true, its executed after any other options.
/// list_directories_firs = true and reverse = false will return directories
/// last
#if BOOST_OS_LINUX
std::vector<std::string> ListDir(const std::string &dir_path,
                                 bool sort_alpha = false,
                                 bool list_directories_first = true,
                                 bool revert = false);

/// return listdir output in string format, each token separated by '\n'
/// no revert option
std::string ListDirString(const std::string &dir_path, bool sort_alpha = false,
                          bool list_directories_first = true);

void AsyncRemovePath(const std::string &path_,
                     const std::string &mode = std::string());
#endif

/// remove trailing / at end of path_, e.g. /foo/bar// -> /foo/bar
/// except if path_ == '/', then path_ is returned as is
std::string RemoveTrailingSeparator(const std::string &path_);

/// remove leading / at front of path_, e.g. ///foo/bar -> foo/bar
/// note if path_ consists of only '/', then empty string is returned
std::string RemoveLeadingSeparator(const std::string &path_);

/// collapse leading /. e.g. ///foo/bar -> /foo/bar
std::string CollapseLeadingSeparator(const std::string &path_);

/// if path starts with component (this is affirmed when leading '/' is
/// collapsed in both), return the remainder path after leading component is
/// discarded, leading and trailing '/' not kept in output. otherwise return
/// original path unmodified e.g. /foo/bar, /foo -> bar
std::string DiscardLeadingComponent(const std::string &path_,
                                    const std::string &component_);

std::vector<std::string> PathComponents(const std::string &path);

std::pair<bool, std::string>
NearestCommonDir(const std::vector<std::string> &paths);

std::string JoinPath(const std::vector<std::string> &components_);

template <typename... Ts> std::string JoinPath(const Ts &... args);

template <typename T>
std::string JoinVector(const std::vector<T> &v,
                       const std::string &d = std::string(),
                       bool brackets = false);

template <>
std::string JoinVector<std::string>(const std::vector<std::string> &v,
                                    const std::string &d, bool brackets);

template <typename T>
std::string JoinArray(const std::unique_ptr<T[]> &v, int n,
                      const std::string &d = std::string(),
                      bool brackets = true);

/// make a tczyx 5D vector from dims_
/// if dims_ has less than 5 dimensions, 1s will be inserted at
/// start of output vector
template <typename T> std::vector<int> To5D(const std::vector<T> &dims_);

template <typename Container, typename T = typename Container::value_type>
Container Maximum(const Container &container, T val);

template <typename Container>
Container Maximum(const Container &container0, const Container &container1);

template <typename Container>
Container Minimum(const Container &container0, const Container &container1);

template <typename T, typename Container>
T ReduceProdSeq(const Container &container);

template <typename T, typename Container>
T ReduceSumSeq(const Container &container);

template <typename T1, typename T2, typename T3>
std::vector<T1> AddSeq(const std::vector<T2> &lhs, const std::vector<T3> &rhs);

template <typename T>
std::vector<T> operator+(const std::vector<T> &lhs, const std::vector<T> &rhs);

template <typename T>
std::vector<T> operator+(const std::vector<T> &lhs, const T rhs);

template <typename T1, typename T2, typename T3>
std::vector<T1> SubtractSeq(const std::vector<T2> &lhs,
                            const std::vector<T3> &rhs);

template <typename T>
std::vector<T> operator-(const std::vector<T> &lhs, const std::vector<T> &rhs);

template <typename T>
std::vector<T> operator-(const std::vector<T> &lhs, const T rhs);

template <typename T1, typename T2, typename T3>
std::vector<T1> MultiplySeq(const std::vector<T2> &lhs,
                            const std::vector<T3> &rhs);

template <typename T>
std::vector<T> operator*(const std::vector<T> &lhs, const std::vector<T> &rhs);

template <typename T>
std::vector<T> operator*(const std::vector<T> &lhs, const T rhs);

template <typename T1, typename T2, typename T3>
std::vector<T1> DivideSeq(const std::vector<T2> &numerator,
                          const std::vector<T3> &denominator);

template <typename T>
std::vector<T> operator/(const std::vector<T> &lhs, const T rhs);

template <typename Container> bool AllPositive(const Container &container);

template <typename Container> bool AllNonNegative(const Container &container);

template <typename Container> bool AllNonZero(const Container &container);

template <typename Container> bool AllZeros(const Container &container);

template <typename Container> bool AnyPositive(const Container &container);

template <typename Container> bool AnyNegative(const Container &container);

template <typename Container> bool AnyZeros(const Container &container);

/// true of element wise container0 > container1 for every element
template <typename Container0, typename Container1>
bool AllGreater(Container0 container0, Container1 container1);

/// used to help convert number string sequence to a sortable format
/// for example, if a sequence has max value 192, the number 12 will
/// be returned as "012"
std::string PadNumStr(int64_t n, int64_t max_n);

/// rename the file in file_list in a way that file name's string sort
/// result is equal to sorting the number contained in file name.
/// e.g. rename A3.tif, A13.tif to A03.tif, A13.tif.
/// the file_list vector element will be updated if
/// prenumber and postnumber are both found, and rename is successful.
/// the rename is not only done in the file_list vector,
/// but also the actual file on disk is renamed
/// \param prenumber: string preceeding number in the file name,
/// "A" in above example. parent dir will be appended internally if needed
/// \param postnumber: string following number in the file name,
/// ".tif" in above example
/// \param sort_files: if or not to sort files
void RenameFilesToSortable(std::vector<std::string> &file_list,
                           const std::string &prenumber = std::string(),
                           const std::string &postnumber = std::string(),
                           bool sort_files = true);

/// note non whitespace single character delimiter is treated differently,
/// in that every delimiter is viewed to be preceded by a token, the token
/// can be empty. whitespace and other delimiters only return non empty tokens
/// " ab c " -> "ab", "c"
/// ",ab,c," -> "", "ab", "c"
std::vector<std::string> SplitString(const std::string &in_string,
                                     const std::string &delim = std::string());

std::string StringLower(const std::string &input);

std::string &StringLower(std::string &input);

std::string &StringLower(std::string &&input);

std::string StripLeading(const std::string &input);

std::string StripTrailing(const std::string &input);

std::string Strip(const std::string &input);

#if BOOST_OS_LINUX
/// return process's current RAM usage in MB
/// read here for more info: http://ewx.livejournal.com/579283.html
/// retrieved from /proc/pid/status, VmSize
double ProcCurrRAM(const std::string &unit = "GB");

/// return process's peak RAM usage in MB
/// retrieved from /proc/pid/status, VmSize
double ProcPeakRAM(const std::string &unit = "GB");
#endif

double MemorySize(double nbytes, const std::string &unit);

/// return random alphabetical string (A-Z) of length l
std::string RandChars(int l = 10);

std::string HostName();

inline bool MPIInitialized() {
#if !MCP3D_MPI_BUILD
  return false;
#else
    int mpi;
    MPI_Initialized(&mpi);
    return mpi == 1;
#endif
}

bool HostOnCluster();

int DefaultNumThreads();

std::string SysCmdResult(const char *const cmd, const std::string &mode = "");

inline std::string SysCmdResult(const std::string &cmd,
                                const std::string &mode = "") {
  return SysCmdResult(cmd.c_str());
}

template <typename T = long> T IntPow(int base, int exp);

int ParseInt(const std::string &int_str);

double ParseDouble(const std::string &double_str);

} // namespace mcp3d

template <typename... Ts> std::string mcp3d::JoinPath(const Ts &... args) {
  std::vector<std::string> components{args...};
  return JoinPath(components);
}

template <typename T>
std::string mcp3d::JoinVector(const std::vector<T> &v, const std::string &d,
                              bool brackets) {
  std::string delim(d), output = brackets ? "[" : "";
  if (delim.empty())
    delim = " ";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i == 0)
      output.append(std::to_string(v[i]));
    else {
      output.append(delim);
      output.append(std::to_string(v[i]));
    }
  }
  if (brackets)
    output.append("]");
  return output;
}

template <typename T>
std::string mcp3d::JoinArray(const std::unique_ptr<T[]> &v, int n,
                             const std::string &d, bool brackets) {
  std::string delim(d), output = brackets ? "[" : "";
  if (delim.empty())
    delim = " ";
  for (int i = 0; i < n; ++i) {
    if (i == 0)
      output.append(std::to_string(v[i]));
    else {
      output.append(delim);
      output.append(std::to_string(v[i]));
    }
  }
  if (brackets)
    output.append("]");
  return output;
}

template <typename T>
std::vector<int> mcp3d::To5D(const std::vector<T> &dims_) {
  static_assert(std::is_integral<T>(), "must have integer element type");
  MCP3D_ASSERT(dims_.size() <= 5)
  MCP3D_ASSERT(mcp3d::AllPositive(dims_))
  std::vector<int> dims(dims_);
  while (dims.size() < 5)
    dims.insert(dims.begin(), 1);
  return dims;
}

template <typename Container, typename T>
Container mcp3d::Maximum(const Container &container, T val) {
  if (sizeof(typename Container::value_type) < sizeof(T))
    MCP3D_MESSAGE("warning: possible narrowing conversion")
  Container result = container;
  for (auto &v : result)
    if (v < val)
      v = val;
  return result;
}

template <typename Container>
Container mcp3d::Minimum(const Container &container0,
                         const Container &container1) {
  using element_type = typename Container::value_type;
  static_assert(std::is_arithmetic<element_type>(),
                "container elements must be arithmetic type");
  MCP3D_ASSERT(container0.size() == container1.size())
  Container result;
  for (size_t i = 0; i < container0.size(); ++i) {
    if (container0[i] <= container1[i])
      result.push_back(container0[i]);
    else
      result.push_back(container1[i]);
  }
  return result;
}

template <typename T, typename Container>
T mcp3d::ReduceProdSeq(const Container &container) {
  MCP3D_ASSERT(!container.empty())
  using element_type = typename Container::value_type;
  static_assert(std::is_arithmetic<element_type>(),
                "container elements must be arithmetic type");
  if (sizeof(T) < sizeof(element_type) ||
      (std::is_floating_point<element_type>() &&
       std::is_integral<element_type>()))
    std::cout << "warning: output type " << typeid(T).name()
              << " is narrower than input type " << typeid(element_type).name()
              << std::endl;
  T acc(1);
  for (const auto &element : container)
    acc *= static_cast<T>(element);
  return acc;
}

template <typename T, typename Container>
T mcp3d::ReduceSumSeq(const Container &container) {
  MCP3D_ASSERT(!container.empty())
  using element_type = typename Container::value_type;
  static_assert(std::is_arithmetic<element_type>(),
                "container elements must be arithmetic type");
  if (sizeof(T) < sizeof(element_type) ||
      (std::is_floating_point<element_type>() &&
       std::is_integral<element_type>()))
    std::cout << "warning: output type " << typeid(T).name()
              << " is narrower than input type " << typeid(element_type).name()
              << std::endl;
  T acc(1);
  for (const auto &element : container)
    acc += static_cast<T>(element);
  return acc;
}

template <typename T1, typename T2, typename T3>
std::vector<T1> mcp3d::AddSeq(const std::vector<T2> &lhs,
                              const std::vector<T3> &rhs) {
  static_assert(std::is_arithmetic<T1>() && std::is_arithmetic<T2>() &&
                    std::is_arithmetic<T3>(),
                "container elements must be arithmetic type");
  MCP3D_ASSERT(lhs.size() == rhs.size())
  std::vector<T1> output;
  for (size_t i = 0; i < lhs.size(); ++i)
    output.push_back(static_cast<T1>(lhs[i]) + static_cast<T1>(rhs[i]));
  return output;
}

template <typename T>
std::vector<T> mcp3d::operator+(const std::vector<T> &lhs,
                                const std::vector<T> &rhs) {
  static_assert(std::is_arithmetic<T>(),
                "container elements must be arithmetic type");
  MCP3D_ASSERT(lhs.size() == rhs.size())
  std::vector<T> result;
  for (size_t i = 0; i < lhs.size(); ++i)
    result.push_back(lhs[i] + rhs[i]);
  return result;
}

template <typename T>
std::vector<T> mcp3d::operator+(const std::vector<T> &lhs, const T rhs) {
  static_assert(std::is_arithmetic<T>(),
                "container elements must be arithmetic type");
  std::vector<T> result;
  for (size_t i = 0; i < lhs.size(); ++i)
    result.push_back(lhs[i] + rhs);
  return result;
}

template <typename T1, typename T2, typename T3>
std::vector<T1> mcp3d::SubtractSeq(const std::vector<T2> &lhs,
                                   const std::vector<T3> &rhs) {
  static_assert(std::is_arithmetic<T1>() && std::is_arithmetic<T2>() &&
                    std::is_arithmetic<T3>(),
                "container elements must be arithmetic type");
  MCP3D_ASSERT(lhs.size() == rhs.size());
  std::vector<T1> output;
  for (size_t i = 0; i < lhs.size(); ++i)
    output.push_back(static_cast<T1>(lhs[i]) - static_cast<T1>(rhs[i]));
  return output;
}

template <typename T>
std::vector<T> mcp3d::operator-(const std::vector<T> &lhs,
                                const std::vector<T> &rhs) {
  static_assert(std::is_arithmetic<T>(),
                "container elements must be arithmetic type");
  MCP3D_ASSERT(lhs.size() == rhs.size())
  std::vector<T> result;
  for (size_t i = 0; i < lhs.size(); ++i)
    result.push_back(lhs[i] - rhs[i]);
  return result;
}

template <typename T>
std::vector<T> mcp3d::operator-(const std::vector<T> &lhs, const T rhs) {
  static_assert(std::is_arithmetic<T>(),
                "container elements must be arithmetic type");
  std::vector<T> result;
  for (size_t i = 0; i < lhs.size(); ++i)
    result.push_back(lhs[i] - rhs);
  return result;
}

template <typename T1, typename T2, typename T3>
std::vector<T1> mcp3d::MultiplySeq(const std::vector<T2> &lhs,
                                   const std::vector<T3> &rhs) {
  static_assert(std::is_arithmetic<T1>() && std::is_arithmetic<T2>() &&
                    std::is_arithmetic<T3>(),
                "container elements must be arithmetic type");
  MCP3D_ASSERT(lhs.size() == rhs.size());
  std::vector<T1> output;
  for (size_t i = 0; i < lhs.size(); ++i)
    output.push_back(static_cast<T1>(lhs[i]) * static_cast<T1>(rhs[i]));
  return output;
}

template <typename T>
std::vector<T> mcp3d::operator*(const std::vector<T> &lhs,
                                const std::vector<T> &rhs) {
  static_assert(std::is_arithmetic<T>(),
                "container elements must be arithmetic type");
  MCP3D_ASSERT(lhs.size() == rhs.size())
  std::vector<T> result;
  for (size_t i = 0; i < lhs.size(); ++i)
    result.push_back(lhs[i] * rhs[i]);
  return result;
}

template <typename T>
std::vector<T> mcp3d::operator*(const std::vector<T> &lhs, const T rhs) {
  static_assert(std::is_arithmetic<T>(),
                "container elements must be arithmetic type");
  std::vector<T> result;
  for (size_t i = 0; i < lhs.size(); ++i)
    result.push_back(lhs[i] * rhs);
  return result;
}

template <typename T1, typename T2, typename T3>
std::vector<T1> mcp3d::DivideSeq(const std::vector<T2> &numerator,
                                 const std::vector<T3> &denominator) {
  static_assert(std::is_arithmetic<T1>() && std::is_arithmetic<T2>() &&
                    std::is_arithmetic<T3>(),
                "container elements must be arithmetic type");
  MCP3D_ASSERT(numerator.size() == denominator.size())
  if (!AnyZeros<std::vector<T3>>(denominator))
    MCP3D_RUNTIME_ERROR("division by 0")
  std::vector<T1> output;
  for (size_t i = 0; i < numerator.size(); ++i)
    output.push_back(static_cast<T1>(numerator[i]) /
                     static_cast<T1>(denominator[i]));
  return output;
}

template <typename T>
std::vector<T> mcp3d::operator/(const std::vector<T> &lhs, const T rhs) {
  static_assert(std::is_arithmetic<T>(),
                "container elements must be arithmetic type");
  if (rhs == (T)0)
    MCP3D_RUNTIME_ERROR("division by 0")
  std::vector<T> result;
  for (size_t i = 0; i < lhs.size(); ++i)
    result.push_back(lhs[i] / rhs);
  return result;
}

template <typename Container>
bool mcp3d::AllPositive(const Container &container) {
  MCP3D_ASSERT(!container.empty());
  static_assert(std::is_arithmetic<typename Container::value_type>(),
                "container elements must be arithmetic type");
  for (const auto &element : container)
    if (element <= 0)
      return false;
  return true;
}

template <typename Container>
bool mcp3d::AllNonNegative(const Container &container) {
  MCP3D_ASSERT(!container.empty())
  static_assert(std::is_arithmetic<typename Container::value_type>(),
                "container elements must be arithmetic type");
  for (const auto &element : container)
    if (element < 0)
      return false;
  return true;
}

template <typename Container>
bool mcp3d::AllNonZero(const Container &container) {
  MCP3D_ASSERT(!container.empty())
  static_assert(
      std::is_integral<typename Container::value_type>(),
      "container elements must be integer for equality evaluation with zero");
  for (const auto &element : container)
    if (element == 0)
      return false;
  return true;
}

template <typename Container> bool mcp3d::AllZeros(const Container &container) {
  MCP3D_ASSERT(!container.empty())
  static_assert(
      std::is_integral<typename Container::value_type>(),
      "container elements must be integer for equality evaluation with zero");
  for (const auto &element : container)
    if (element != 0)
      return false;
  return true;
}

template <typename Container>
bool mcp3d::AnyPositive(const Container &container) {
  MCP3D_ASSERT(!container.empty())
  static_assert(std::is_arithmetic<typename Container::value_type>(),
                "container elements must be arithmetic type");
  for (const auto &element : container)
    if (element > 0)
      return true;
  return false;
}

template <typename Container>
bool mcp3d::AnyNegative(const Container &container) {
  MCP3D_ASSERT(!container.empty())
  static_assert(std::is_arithmetic<typename Container::value_type>(),
                "container elements must be arithmetic type");
  for (const auto &element : container)
    if (element < 0)
      return true;
  return false;
}

template <typename Container> bool mcp3d::AnyZeros(const Container &container) {
  MCP3D_ASSERT(!container.empty())
  static_assert(
      std::is_integral<typename Container::value_type>(),
      "container elements must be integer for equality evaluation with zero");
  for (const auto &element : container)
    if (element == 0)
      return true;
  return false;
}

template <typename Container0, typename Container1>
bool mcp3d::AllGreater(Container0 container0, Container1 container1) {
  MCP3D_ASSERT(!container0.empty() && !container1.empty())
  static_assert(std::is_arithmetic<typename Container0::value_type>() &&
                    std::is_arithmetic<typename Container1::value_type>(),
                "container elements must be arithmetic type");
  MCP3D_ASSERT(container0.size() == container1.size())
  for (size_t i = 0; i < container0.size(); ++i)
    if (container0[i] <= container1[i])
      return false;
  return true;
}

template <typename T> T mcp3d::IntPow(int base, int exp) {
  if (exp < 0 || base < 0)
    MCP3D_DOMAIN_ERROR("base and exp must both be positive integer")
  if (exp == 0)
    return 1;
  long result = base;
  while (exp > 1) {
    result *= (long)base;
    --exp;
  }
  return (T)result;
}

#if MCP3D_MPI_BUILD

#include <mpi.h>

namespace mcp3d {

void AsyncRemovePathMPI(const std::string &path_,
                        MPI_Comm comm = MPI_COMM_WORLD,
                        const std::string &mode = std::string());

}

#endif // MPIPARALLEL

#endif // MCP3D_MCP3D_UTILITY_HPP
