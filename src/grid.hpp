#pragma once

#include "interval.hpp"
#include <memory>
#include <string>
#include <vector>

class Grid {
public:
  Grid() : grid_vertex_pad_size_(0) {}

  // evenly distribute vptrs
  template <typename T>
  Grid(const VID_t grid_vertex_pad_size, const VID_t interval_block_size,
       const VID_t grid_interval_size, T &program, bool mmap_)
      : grid_vertex_pad_size_(0) {

    assertm(
        grid_vertex_pad_size % grid_interval_size == 0,
        "grid_vertex_pad_size must be evenly divisible by grid_interval_size");
    interval_size_ = grid_interval_size;
    interval_block_size_ = interval_block_size;
    grid_vertex_pad_size_ = grid_vertex_pad_size;
    VID_t vid_left = grid_vertex_pad_size;
    VID_t interval_vertex_pad_size =
        grid_vertex_pad_size /
        grid_interval_size; // guaranteed to evenly divide
    intervals_.reserve(grid_interval_size);
    auto default_interval = INTERVAL_BASE;

    for (auto interval_idx = 0; interval_idx < grid_interval_size;
         interval_idx++) {

      intervals_.push_back(nullptr);
      try {
        intervals_.rbegin()->reset(new Interval(
            interval_vertex_pad_size, interval_idx, default_interval, mmap_));
      } catch (...) {
        assertm(false, "Failed to create new Interval");
      }
    }
  }

  inline Interval *GetInterval(const VID_t idx) {
    assertm(idx < interval_size_,
            "Requested interval can not exceed total contained in Grid");
    return intervals_[idx].get();
  }
  inline size_t GetDim() const { return intervals_.size(); }
  inline VID_t GetNVertices() const { return grid_vertex_pad_size_; }
  inline VID_t GetNIntervals() const { return interval_size_; }
  inline VID_t GetNBlocks() const { return interval_block_size_; }
  inline void Release() const {
    for (auto &interval : intervals_) {
      if (interval->IsInMemory())
        interval->Release();
    }
  }

private:
  std::vector<std::shared_ptr<Interval>> intervals_;
  VID_t grid_vertex_pad_size_;
  VID_t interval_size_;
  VID_t interval_block_size_;
};
