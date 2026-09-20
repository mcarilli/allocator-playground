// VMM usage based on
// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html
// https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/
//   full code here https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/cuda-vmm/cuvector.cpp#L92

#include <cassert>
#include <cuda.h>
#include <iostream>
#include <nvtx3/nvtx3.hpp>
#include <stack>
#include <unordered_map>
#include <vector>

#include <algorithm>
#include <random>
#include <chrono>

#include "err.h"

using namespace std;

__global__ void touch_va_kernel(uint8_t *start, unsigned sleep_ns=0, bool disable_touch=false) {
  // Optionally spin a bit to stress potential unmapping race conditions.
  __nanosleep(sleep_ns);
  if (!disable_touch) {
    uint8_t *mine = start + threadIdx.x + blockDim.x * blockIdx.x;
    *mine = *mine + 1;
  }
};

struct Addr {
  CUdeviceptr start;
  size_t size;
  Addr(CUdeviceptr start, size_t size) : start{start}, size{size} {}
};

// "endless VA" allocator backed by fixed-size physical address (PA) handles
// A vector holds the allocated handles, and a stack holds any
// already-allocated-but-currently-available handles for remapping to new VA.
// Probably good for graphs!
class EndlessVAAllocator {
  private:
  struct MappedVARange {
      CUdeviceptr start;
      size_t size;
      vector<CUmemGenericAllocationHandle> pa_handles;
      MappedVARange(CUdeviceptr start, size_t size) : start{start}, size{size}, pa_handles{} {}
  };
  CUmemAllocationProp prop;
  CUmemAccessDesc access_desc;
  // I'm a little worried mapped_va_range might get big if we have many large allocations.
  unordered_map<CUdeviceptr, MappedVARange> mapped_va_ranges;
  vector<Addr> addr_sequence;
  vector<CUmemGenericAllocationHandle> created_pa_handles;
  stack<CUmemGenericAllocationHandle> available_pa_handles;
  size_t granularity;
  size_t granularity_recommended;
  size_t granularity_minimum;
  size_t alloc_sequence_id;
  bool dry_run;

  public:
  EndlessVAAllocator()
    : access_desc{}, prop{}, mapped_va_ranges{}, addr_sequence{}, created_pa_handles{},
      available_pa_handles{}, alloc_sequence_id{0}, dry_run{true}
  {
    nvtx3::scoped_range r{"EndlessVAAllocator()"};

    int device;
    CHK_RNT(cudaGetDevice(&device));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;

    access_desc.location = prop.location;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    granularity = 0;
    granularity_recommended = 0;
    granularity_minimum = 0;
    CHK_DRV(cuMemGetAllocationGranularity(&granularity_recommended, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    CHK_DRV(cuMemGetAllocationGranularity(&granularity_minimum, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    cout << "granularity_recommended = " << granularity_recommended << endl;
    cout << "granularity_minimum = " << granularity_minimum << endl;
    granularity = granularity_minimum;

    // TODO (optional): use a single call to cuMemAddressReserve to grab a huge VA range up front,
    // instead of on-the-fly in alloc.
  }

  void reset_alloc_sequence_id() {
    // Assert the non-dry run used all the allocations.
    // This can be relaxed if you only want to replay part of the dry-run sequence.
    assert(alloc_sequence_id == mapped_va_ranges.size());
    assert(alloc_sequence_id == addr_sequence.size());
    alloc_sequence_id = 0;
  }

  void end_dry_run() {
    dry_run = false;
    reset_alloc_sequence_id();
  }

  Addr alloc(size_t blocks) {
    nvtx3::scoped_range r{"alloc"};

    cout << "Allocating " << blocks << " blocks" << endl;

    size_t size = granularity * blocks;

    if (dry_run) {
      CUdeviceptr ptr;

      // No starting address hints, we don't care where this VA is created.
      CHK_DRV(cuMemAddressReserve(&ptr, size, 0ULL, 0U, 0));

      MappedVARange allocation{ptr, size};
      Addr addr{ptr, size};

      size_t size_remaining = size;

      // Reuse and map unused physical blocks from the available-handle stack.
      // Mapping multiple VA ranges to the same PA should "just work" as long as
      // you don't use aliasing VAs in the same kernel:
      // https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html#virtual-aliasing-support
      while (!available_pa_handles.empty() && size_remaining > 0) {
        cout << "Reusing existing handle" << endl;

        CUmemGenericAllocationHandle handle = available_pa_handles.top();
        available_pa_handles.pop();

        CHK_DRV(cuMemMap(ptr, granularity, 0ULL, handle, 0ULL));

        // I think setting access must be done on a per-handle basis.
        CHK_DRV(cuMemSetAccess(ptr, granularity, &access_desc, 1ULL));

        allocation.pa_handles.push_back(handle);
        size_remaining -= granularity;
        ptr += granularity;
      }

      // Create new PA pages and handles if needed.
      while (size_remaining > 0) {
        CUmemGenericAllocationHandle new_handle;

        CHK_DRV(cuMemCreate(&new_handle, granularity, &prop, 0));

        CHK_DRV(cuMemMap(ptr, granularity, 0ULL, new_handle, 0ULL));

        // I think this must be done on a per-handle basis.
        CHK_DRV(cuMemSetAccess(ptr, granularity, &access_desc, 1ULL));

        allocation.pa_handles.push_back(new_handle);
        created_pa_handles.push_back(new_handle);
        size_remaining -= granularity;
        ptr += granularity;
      }

      const bool ok = mapped_va_ranges.insert({allocation.start, allocation}).second;
      assert(ok);

      addr_sequence.push_back(addr);

      alloc_sequence_id++; // during dry runs, it's just for error checking

      return addr;
    }

    // Fast path after dry run
    const Addr next_addr = addr_sequence[alloc_sequence_id++];
    // Ensure we're replaying the dry run safely
    assert(size == next_addr.size);
    return next_addr;
  }

  void unmap_va(MappedVARange &mapped_va_range) {
    nvtx3::scoped_range r{"unmap_va"};

    CHK_DRV(cuMemUnmap(mapped_va_range.start, mapped_va_range.size));

    CHK_DRV(cuMemAddressFree(mapped_va_range.start, mapped_va_range.size)); // cuMemUnmap must be called first
  }

  void free_pa_handles(MappedVARange &mapped_va_range) {
    for (auto handle : mapped_va_range.pa_handles)
      available_pa_handles.push(handle);

    mapped_va_range.pa_handles.clear();
  }

  // For the dry run, we want to retain VA ranges and VA->PA mappings,
  // but also push its PA blocks to the "available PA" stack for
  // potential remapping by later VA allocations.
  void free(Addr addr) {
    nvtx3::scoped_range r{"free"};

    if (dry_run) {
      cout << "Freeing " << addr.size << " bytes at " << addr.start << endl;
      const auto it = mapped_va_ranges.find(addr.start);
      assert(it != mapped_va_ranges.end());
      assert(addr.start == it->second.start);
      free_pa_handles(it->second);
    }
    // If we're not doing a dry run, free() is a no-op.
  }

  ~EndlessVAAllocator() {
    nvtx3::scoped_range r{"~EndlessVAAllocator()"};

    for (auto& [ptr, mapped_va_range] : mapped_va_ranges) {
      // If this VA range was already freed, its pa_handles will be empty.
      if (!mapped_va_range.pa_handles.empty())
        free_pa_handles(mapped_va_range);
      unmap_va(mapped_va_range);
    }

    assert(available_pa_handles.size() == created_pa_handles.size());

    mapped_va_ranges.clear();

    for (int i = 0; i < created_pa_handles.size(); i++)
      CHK_DRV(cuMemRelease(created_pa_handles[i]));
  }
};

struct FuzzAllocation {
  bool allocated;
  size_t blocks;
  size_t id;
  FuzzAllocation(size_t blocks, size_t id) : blocks{blocks}, id{id}, allocated{false} {}
};

int main() {
  // Initialize primary context
  // https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#interoperability-between-runtime-and-driver-apis
  CHK_RNT(cudaSetDevice(0));

  const size_t num_fuzz_allocations = 100;
  const size_t max_fuzz_allocation_blocks = 200;

  // Couple of "ordinary launches" to sanity check on profile if vmm shenanigans incur performance hit
  {
    nvtx3::scoped_range r{"Ordinary malloc and launch sanity check"};
    uint8_t *ptr;
    size_t bytes = max_fuzz_allocation_blocks << 21;
    CHK_RNT(cudaMalloc(&ptr, bytes));
    for (int i = 0; i < 2; i++) {
      touch_va_kernel<<<bytes / 128, 128>>>(ptr);
      CHK_RNT(cudaPeekAtLastError());
      CHK_RNT(cudaDeviceSynchronize());
    }
    CHK_RNT(cudaFree(ptr));
  }

  random_device rd; // helps get random seed from hardware
  unsigned seed = rd(); // or 42, whatever
  auto engine = default_random_engine(seed);
  std::uniform_int_distribution<size_t> dist(1, max_fuzz_allocation_blocks);
  vector<FuzzAllocation> fuzz_allocations{};
  vector<FuzzAllocation *> fuzz_allocation_order{};
  fuzz_allocations.reserve(num_fuzz_allocations);
  fuzz_allocation_order.reserve(2 * num_fuzz_allocations); // visit each twice, once to allocate and once to free
  for (size_t i = 0; i < num_fuzz_allocations; i++) {
    fuzz_allocations.push_back({dist(engine), i});
    fuzz_allocation_order.push_back(&fuzz_allocations[i]);
    fuzz_allocation_order.push_back(&fuzz_allocations[i]);
  }
  shuffle(fuzz_allocation_order.begin(), fuzz_allocation_order.end(), engine);

  {
    EndlessVAAllocator allocator{};

    unordered_map<size_t, Addr> active_fuzz_allocations{};

    auto fuzz = [&]() {
      for (int i = 0; i < 2 * num_fuzz_allocations; i++) {
        FuzzAllocation *fuzz_allocation_ptr = fuzz_allocation_order[i];
        if (fuzz_allocation_ptr->allocated) {
          cout << "Freeing allocation id " << fuzz_allocation_ptr->id << endl;
          const auto it = active_fuzz_allocations.find(fuzz_allocation_ptr->id);
          assert(it != active_fuzz_allocations.end());
          allocator.free(it->second);
          active_fuzz_allocations.erase(it);
          fuzz_allocation_ptr->allocated = false;
        } else {
          cout << "Allocating allocation id " << fuzz_allocation_ptr->id << endl;
          Addr addr = allocator.alloc(fuzz_allocation_ptr->blocks);
          const bool ok = active_fuzz_allocations.insert({fuzz_allocation_ptr->id, addr}).second;
          assert(ok);
          fuzz_allocation_ptr->allocated = true;
        }
        // After every alloc and/or free, retry touching all active allocations.
        // This causes the test to take a few seconds, but i'd rather be thorough.
        for (auto &[id, addr] : active_fuzz_allocations) {
          touch_va_kernel<<<addr.size / 128, 128>>>((uint8_t *)addr.start);
          CHK_RNT(cudaPeekAtLastError());
          // CHK_RNT(cudaDeviceSynchronize());
        }
      }
    };

    // Dry run
    {
      nvtx3::scoped_range r{"Dry run"};
      fuzz();
      assert(active_fuzz_allocations.empty());
      CHK_RNT(cudaDeviceSynchronize());
      allocator.end_dry_run();
    }

    cout << "\n\n\n" << endl;

    // "Production" run
    {
      nvtx3::scoped_range r{"Replay"};
      fuzz();
      assert(active_fuzz_allocations.empty());
      CHK_RNT(cudaDeviceSynchronize());
      allocator.reset_alloc_sequence_id();
    }
  }

  // check for errors after EndlessVAAllocator destructor
  CHK_RNT(cudaDeviceSynchronize());
}
