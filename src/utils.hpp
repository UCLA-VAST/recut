
#include<boost/filesystem.hpp>
using namespace boost::filesystem;

std::string get_curr() {
  path full_path(current_path());
  return canonical(full_path).string();
}

