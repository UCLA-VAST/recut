#include <filesystem>
using std::filesystem::exists;
using std::filesystem::remove;
using std::filesystem::remove_all;
using std::filesystem::directory_iterator;
using std::filesystem::canonical;
using std::filesystem::current_path;
using std::filesystem::full_path;

namespace fs = std::filesystem;

string get_curr() {
    path full_path(fs::current_path());
    return fs::canonical(full_path).string();
}
