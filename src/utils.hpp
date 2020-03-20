
#include <filesystem>

namespace fs = std::filesystem;

string get_curr() {
    path fs::current_path();
    return fs::canonical(path).string();
}
