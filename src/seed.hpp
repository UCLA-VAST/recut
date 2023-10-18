#pragma once
#include "config.hpp"

struct Seed {
  // coordinate in pixels with respect to the image volume
  // agnostic of voxel size and world-space
  GridCoord coord;
  std::array<double, 3> coord_um;

  // based purely off the length in pixels in 1 cardinal direction
  uint8_t radius;
  // converted radius based off voxel size (iso/anisotropic)
  double radius_um;
  // world-space volume based off voxel size (iso/anisotropic)
  uint64_t volume;

  Seed(GridCoord coord, std::array<double, 3> coord_um, uint8_t radius,
      double radius_um, uint64_t volume)
      : coord(coord), coord_um(coord_um), radius(radius), radius_um(radius_um), volume(volume) {}

  friend std::ostream &operator<<(std::ostream &os, const Seed &s) {
    os << s.coord << ", in um: [" << std::to_string(s.coord_um[0]) << ", " +
              std::to_string(s.coord_um[1]) + ", " +
              std::to_string(s.coord_um[2]) +
              "], radius: " + std::to_string(s.radius) + 
              " radius um: " + std::to_string(s.radius_um) + '\n';
    return os;
  }

};
