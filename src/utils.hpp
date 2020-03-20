#include <filesystem>
using std::filesystem::exists;
using std::filesystem::remove;
using std::filesystem::remove_all;
using std::filesystem::directory_iterator;
using std::filesystem::canonical;
using std::filesystem::current_path;
using std::filesystem::full_path;

std::string get_curr() {
  path full_path(current_path());
  return canonical(full_path).string();
}

