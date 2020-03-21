#include <filesystem>
namespace fs = std::filesystem;

using fs::exists;
using fs::remove;
using fs::remove_all;
using fs::directory_iterator;
using fs::canonical;
using fs::current_path;
using fs::path;


std::string get_curr() {
    path full_path(current_path());
    return canonical(full_path).string();
}
