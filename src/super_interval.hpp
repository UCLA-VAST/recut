#ifndef SUPER_INTERVAL_H_
#define SUPER_INTERVAL_H_
#include<memory>
#include<string>
#include<vector>
#include"interval.hpp"
class SuperInterval
{
public:
    SuperInterval():nvertices_(0) {}

    // evenly distribute vptrs
    template<typename T>
    SuperInterval(const VID_t nvid, const VID_t nblocks, const VID_t nintervals, 
                T& program, bool mmap_):nvertices_(0)
    {
        nintervals_ = nintervals;
        nblocks_ = nblocks;
        nvertices_ = nvid;
        VID_t vid_left = nvid;
        VID_t interval_size = nvid / nintervals; // guaranteed to evenly divide
        VID_t offset = 0;
        intervals_.reserve(nintervals);
        auto default_interval = INTERVAL_BASE;
        for (auto interval_num=0; interval_num<nintervals; interval_num++) {
            //VID_t iblock, jblock, kblock;
            //iblock= jblock= kblock = 0;
            //program.get_block_subscript(interval_num, iblock, jblock, kblock);

            //offset = program.get_vid(i, j, k);
            offset = interval_num * interval_size;

            intervals_.push_back(nullptr);
            try {
                intervals_.rbegin()->reset(new Interval(offset, interval_size, interval_num, default_interval, mmap_));
            } catch (...) {
                cout << "********* Failed to create new Interval. quit fastmarching_tree()." << endl;
            }

#ifdef DEBUG
            //cout<< "Created interval # "<< interval_num << endl;
            Interval* interval = intervals_[interval_num].get();
            struct VertexAttr* attr_init = interval->GetData(); 
            VID_t offset_check = interval->GetOffset();
            assert(offset == offset_check);

            //cout << "iblock " << iblock << " jblock " << jblock << " kblock " << kblock << " offset " << offset << " interval " << interval << endl;
#endif

        }
    }

    inline Interval* GetInterval(size_t idx) {
      assertm(idx < nintervals_, "Requested interval can not exceed total contained in SuperInterval");
      return intervals_[idx].get();
    }
    inline size_t GetDim() const {return intervals_.size();}
    inline VID_t GetNVertices() const {return nvertices_;}
    inline VID_t GetNIntervals() const {return nintervals_;}
    inline VID_t GetNBlocks() const {return nblocks_;}
    inline void Release() const {
      for (auto& interval : intervals_) {
        if (interval->IsInMemory())
          interval->Release();
      }
    }
private:
    std::vector<std::shared_ptr<Interval> > intervals_;
    VID_t nvertices_;
    VID_t nintervals_;
    VID_t nblocks_;
};
#endif//SUPER_INTERVAL_H_
