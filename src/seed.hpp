#pragma once
#include "config.hpp"

struct Seed {
  GridCoord coord;
  uint8_t radius;
  float radius_um;
  uint64_t volume;

  Seed(GridCoord coord, uint8_t radius, float radius_um, uint64_t volume)
      : coord(coord), radius(radius), radius_um(radius_um), volume(volume) {}
};
