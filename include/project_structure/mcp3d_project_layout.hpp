//
// Created by muyezhu on 3/1/19.
//

#ifndef MCP3D_MCP3D_PROJECT_LAYOUT_HPP
#define MCP3D_MCP3D_PROJECT_LAYOUT_HPP

#include <unordered_set>
#include <unordered_map>
#include <map>

namespace mcp3d
{

std::string MultiChannelVolumeRootDir(const std::string &pyr_level_dir);

std::vector<std::string> ChannelNames(const std::string& pyr_level_dir);


class MProjectTree
{
public:
    MProjectTree(const std::string& project_name = std::string{});
    void AddVolume();

private:
    bool ProjectNameValid();

    class Volume
    {
    public:
        Volume(const std::string volume_name);
    private:
        std::string volume_name_;
    };

    class Slice
    {
    public:
        Slice(const std::string slice_name);
        void AddVolume(const std::string& volume_name);

    private:
        std::string slice_name_;
        std::unordered_map<std::string, Volume> volumes_;
    };

    class ScopeMagnification
    {
    public:
        ScopeMagnification(const std::string& scope_name = "Dragonfly", int mag = 30);
        void AddMagnification(int magnification = 30);

    private:
        std::string scope_name_;
        int magnification_;
        std::unordered_map<std::string, Slice> slices_;
    };

    class Stage
    {
    public:
        Stage(const std::string& stage_name = "Raw");
        void AddScope(const std::string& stage_name);

    private:
        std::string _stage_name;
        std::map<std::pair<std::string, int>,
                 ScopeMagnification> scope_magnifications_;
    };

    std::string project_name_;
};

}



#endif //MCP3D_MCP3D_PROJECT_LAYOUT_HPP
