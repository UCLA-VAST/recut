#ifndef INTERVAL_H_
#define INTERVAL_H_

#include "vertex_attr.hpp"
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#define HUGE_PAGE_2MB (2 << 20)

// template <class T>
class Interval {
  public:
    /*
     * interval_vertex_pad_size : number of total vertices to load
     * interval_id : save a unique identifier for the interval
     * base : load the interval from a immutable base interval with all vertices
     * set to defaults
     */
    Interval(VID_t interval_vertex_pad_size, int interval_id, std::string base,
        bool mmap_ = false)
      : in_mem_(false), inmem_ptr0_(nullptr), unmap_(false), active_(false),
      base_(base), fn_(base),
      interval_vertex_pad_size_(interval_vertex_pad_size),
      interval_id_(interval_id), mmap_(mmap_) {
        if (interval_vertex_pad_size > MAX_INTERVAL_VERTICES) {
          cout << "Number of total interval vertices too high: " << interval_vertex_pad_size
            << "\ncurrent max at: " << MAX_INTERVAL_VERTICES
            << "\nIncrease MAX_INTERVAL_BASE in src/config.hpp and rerunning interval base "
            "generation in recut_test.hpp:CreateIntervalBase\n";
          exit(1);
        }
        mmap_length_ = sizeof(VertexAttr) * interval_vertex_pad_size_;

#ifdef USE_HUGE_PAGE
        // For huge pages the length needs to be aligned to the nearest hugepages
        // size
        hp_mmap_length_ =
          ((mmap_length_ + HUGE_PAGE_2MB - 1) / HUGE_PAGE_2MB) * HUGE_PAGE_2MB;
#endif
      }

    Interval(Interval &&m)
      : in_mem_(m.in_mem_), inmem_ptr0_(m.inmem_ptr0_), mmap_ptr_(m.mmap_ptr_),
      mmap_length_(m.mmap_length_),
#ifdef USE_HUGE_PAGE
      hp_mmap_length_(m.hp_mmap_length_),
#endif
      unmap_(m.unmap_), interval_vertex_pad_size_(
          m.interval_vertex_pad_size_) //, heap_(nullptr)

      {
        m.unmap_ = false;
      }

    Interval &operator=(Interval &&m) {
      if (unmap_) {
#ifdef USE_HUGE_PAGE
        assert(munmap(mmap_ptr_, hp_mmap_length_) == 0);
#else
        assert(munmap(mmap_ptr_, mmap_length_) == 0);
#endif
      }
      in_mem_ = m.in_mem_;
      inmem_ptr0_ = m.inmem_ptr0_;
      mmap_ptr_ = m.mmap_ptr_;
      mmap_length_ = m.mmap_length_;
#ifdef USE_HUGE_PAGE
      hp_mmap_length_ = m.hp_mmap_length_;
#endif
      unmap_ = m.unmap_;
      interval_vertex_pad_size_ = m.interval_vertex_pad_size_;
      m.unmap_ = false;
      return *this;
    }

    ~Interval() {
      // if its mmap strategy and currently needs to be unmapped
      if (unmap_ && mmap_) {
#ifdef USE_HUGE_PAGE
        assert(munmap(mmap_ptr_, hp_mmap_length_) == 0);
#else
        assert(munmap(mmap_ptr_, mmap_length_) == 0);
#endif
      } else {
        if (in_mem_) {
          free(inmem_ptr0_);
        }
      }

#ifdef FULL_PRINT
      cout << "~Interval base " << base_ << " fn " << fn_ << '\n';
#endif

      // don't delete the original file protect if haven't written yet
      if (fs::exists(fn_) && (base_ != fn_)) {
        fs::remove(fn_);
        assert(!fs::exists(fn_));
#ifdef FULL_PRINT
        cout << "~Interval deleted fn " << fn_ << '\n';
#endif
      }
    }

    // if this interval has been previously saved to disk
    // the SavedToDisk function will mutate the file name to
    // be specific to this interval such that changes persist
    void LoadFromDisk() {
      assert(!in_mem_);
      in_mem_ = true; // count mmap and user-space buffer as in memory
      if (mmap_) {
        int fd;
#ifdef FULL_PRINT
        cout << "mmap() fn: " << fn_ << " for interval: " << interval_id_ << '\n';
#endif
        assert((fd = open(fn_.c_str(), O_RDWR)) != -1);
#ifdef USE_HUGE_PAGE
        assert((mmap_ptr_ = mmap(nullptr, mmap_length_, PROT_READ | PROT_WRITE,
                MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB, -1,
                0)) != MAP_FAILED);
        assert(read(fd, mmap_ptr_, mmap_length_) > 0);
#else
        assert((mmap_ptr_ = mmap(nullptr, mmap_length_, PROT_READ | PROT_WRITE,
                MAP_PRIVATE, fd, 0)) != MAP_FAILED);
#endif
        assert(close(fd) != 1); // mmap increments the files ref counter, munmap
        // will decrement so we can safely close
        unmap_ = true;
      } else {
        inmem_ptr0_ = (VertexAttr *)malloc(mmap_length_);
#ifdef FULL_PRINT
        cout << "LoadFromDisk() fn: " << fn_ << " for interval: " << interval_id_ << '\n';
#endif
        assert(fs::exists(fn_));
        std::ifstream ifile(
            fn_, ios::in | ios::binary); // ifstream is default read only mode
        // open input
        assert(ifile.is_open());
        assert(ifile.good());

        // read only needed Vertexes
        ifile.read((char *)inmem_ptr0_, mmap_length_);

        // close file
        ifile.close();
      }
    }

    void Release() {
      assert(in_mem_);
      in_mem_ = false;
#ifdef FULL_PRINT
      cout << "Release ";
#endif

      if (mmap_ && unmap_) {
#ifdef USE_HUGE_PAGE
        assert(munmap(mmap_ptr_, hp_mmap_length_) == 0);
#else
        assert(munmap(mmap_ptr_, mmap_length_) == 0);
#endif
#ifdef FULL_PRINT
        cout << "unmapped fn: " << fn_ << '\n';
#endif
      } else {
        free(inmem_ptr0_);
#ifdef FULL_PRINT
        cout << "freed from user-space memory fn: " << fn_ << '\n';
#endif
      }
    }

    void print_state (const std::ofstream& stream) {
      std::cout << " good()=" << stream.good();
      std::cout << " eof()=" << stream.eof();
      std::cout << " fail()=" << stream.fail();
      std::cout << " bad()=" << stream.bad() << '\n';
    }

    /* changes the default fn_ (originally base_) to a unique
     * binary file in the binaries directory
     * This new binary file will be deleted upon ~Interval
     */
    void SaveToDisk() {
      assert(in_mem_);
      in_mem_ = false;
      // open output
      // output is now specific to the interval
      // fn_ = get_data_dir() + "/interval" + std::to_string(interval_id_) + ".bin";
      fn_ = "/tmp/interval" + std::to_string(interval_id_) + ".bin";
#ifdef FULL_PRINT
      cout << "SaveToDisk() fn: " << fn_ <<  " for interval: " << interval_id_ <<'\n';
#endif
      std::ofstream ofile(fn_, ios::out | ios::binary); // write ops
      assert(ofile.is_open());
      assert(ofile.good());

      // write struct array to file
      if (mmap_) {
        print_state(ofile);
        msync((char *) mmap_ptr_, mmap_length_, MS_SYNC);
        ofile.write((char *)mmap_ptr_, mmap_length_);
        print_state(ofile);
        assert(unmap_);
#ifdef USE_HUGE_PAGE
        assert(munmap(mmap_ptr_, hp_mmap_length_) == 0);
#else
        assert(munmap(mmap_ptr_, mmap_length_) == 0);
#endif
        unmap_ = false;
      } else {
        ofile.write((char *)inmem_ptr0_, mmap_length_);
      }
      assert(ofile.good());

      // close file
      ofile.close();

      free(inmem_ptr0_);
    }

    // treat mmap'd as in memory
    inline bool IsInMemory() const { return in_mem_; };
    inline VID_t GetNVertices() const { return interval_vertex_pad_size_; }

    inline VertexAttr *GetData() {
      assert(in_mem_);
      if (mmap_) {
        assert(unmap_);
        return (VertexAttr *)mmap_ptr_;
      } else {
        return inmem_ptr0_;
      }
    }

    // IsActive not currently checked in recut
    inline bool IsActive() const { return active_; }
    inline void SetActive(bool active) { active_ = active; }
    inline void set_start_vertex(VID_t t_start_vertex) {
      start_vertex_ = t_start_vertex;
    }
    inline VID_t get_start_vertex() { return start_vertex_; }
    inline std::string GetFn() { return fn_; }
    inline void set_valid_start(bool t_val) { valid_start_ = t_val; };
    inline bool get_valid_start() { return valid_start_; };

  private:
    bool in_mem_; /// Use mmap if in_mem_, use vector otherwise
    bool valid_start_ = false;
    VertexAttr *inmem_ptr0_;
    VID_t start_vertex_;
    void *mmap_ptr_;
    atomic<bool> active_;
    bool mmap_;
    size_t mmap_length_;
#ifdef USE_HUGE_PAGE
    size_t hp_mmap_length_;
#endif
    bool unmap_; /// Whether should call munmap on destructor
    VID_t interval_vertex_pad_size_;
    std::string fn_;
    std::string base_;
    int interval_id_;
};
#endif // INTERVAL_H_
