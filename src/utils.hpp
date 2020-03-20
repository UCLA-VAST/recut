
#include <filesystem>

namespace fs = std::filesystem;

string get_curr() {
    path full_path(fs::current_path());
    return fs::canonical(full_path).string();
}
