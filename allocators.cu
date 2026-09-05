#include <cassert>
#include <cuda.h>
#include <iostream>
#include <stack>
#include <unordered_map>
#include <vector>

#include <algorithm>
#include <random>
#include <chrono>

#include "err.h"

// Generally possibly helpful things:
//  - https://www.sandordargo.com/blog/2026/09/02/cpp26-hive
//  - Stream ordered page table updates. Seems fine in principle.
//    Heck, a motivated user could implement page tables manually in cuda.
//    Already exists for cuMemMapArrayAsync.
//  - Maybe remapping helps makes graphs more dynamic by giving us a layer of indirection
//    beneath the captured VA.
//  - Why is cuMemMap so fast? How does it communicate updates to the device?
//  - Why is cuMemSetAccess slower? How does it communicate updates to the device?
//  - Why is cuMemUnmap slow? It's NOT synchronous (doesn't wait for prior kernels)
//    Could it be faster if it were stream ordered?
//  - Set the default behavior for cuMemSetAccess so we don't need to call it every time we map.
//    It's expensive (30ish usec/call)
//  - For cudaMallocAsync allocator, piggyback on stream wait/sync calls + cumulativity
//    to ensure better-than-opportunistic allocation reuse.

// Stream ordering design parameters:
//  - Each stream must know which handles are free for it.
//  - A free handle may be free for multiple streams.
//  - Mapping a handle for a stream makes it unavailable to other streams.
//  - Telling stream A to wait for stream B adds B's free handles to A's free handles,
//    but not vice versa.

using namespace std;

__global__ void touch_va_kernel(uint8_t *start, unsigned sleep_ns=0, bool disable_touch=false) {
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

// dumb "endless VA" allocator backed by fixed-size PA handles
// vector holds the allocated handles, and a stack holds any
// already-allocated-but-currently-free handles for remapping to new VA.
// Probably good for graphs!
class EndlessVAAllocator {
  private:
  struct MappedVARange {
      CUdeviceptr start;
      size_t size;
      vector<CUmemGenericAllocationHandle> pa_handles;
      MappedVARange(CUdeviceptr start, size_t size) : start{start}, size{size}, pa_handles{} {}
  };
  unordered_map<CUdeviceptr, MappedVARange> mapped_va_ranges;
  CUmemAllocationProp prop;
  CUmemAccessDesc access_desc;
  size_t granularity;
  size_t granularity_recommended;
  size_t granularity_minimum;
  size_t va_tail;
  vector<CUmemGenericAllocationHandle> handles;
  stack<CUmemGenericAllocationHandle> free_existing_handles;

  public:
  EndlessVAAllocator()
    : mapped_va_ranges{}, access_desc{}, prop{}, va_tail{0}, handles{}, free_existing_handles{}
  {
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

    // TODO (optional): reserve huge VA range up front, here, instead of on-the-fly in alloc.
  }

  Addr alloc(size_t blocks) {
    cout << "Allocating " << blocks << " blocks" << endl;

    size_t size = granularity * blocks;
    CUdeviceptr ptr;

    // No starting address hints, we don't care where this VA is created.
    CHK_DRV(cuMemAddressReserve(&ptr, size, 0ULL, 0U, 0));

    MappedVARange allocation{ptr, size};
    Addr addr{ptr, size};

    size_t size_remaining = size;

    // Reuse and map unused physical blocks.
    while (!free_existing_handles.empty() && size_remaining > 0) {
      cout << "Reusing existing handle" << endl;

      CUmemGenericAllocationHandle handle = free_existing_handles.top();
      free_existing_handles.pop();

      CHK_DRV(cuMemMap(ptr, granularity, 0ULL, handle, 0ULL));

      // I think this must be done on a per-handle basis.
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
      handles.push_back(new_handle);
      size_remaining -= granularity;
      ptr += granularity;
    }

    const bool ok = mapped_va_ranges.insert({allocation.start, allocation}).second;
    assert(ok);

    return addr;
  }

  void unmap_and_free_range(MappedVARange &allocation) {
    // optional?
    CHK_DRV(cuMemUnmap(allocation.start, allocation.size)); // before Free

    // optional?
    CHK_DRV(cuMemAddressFree(allocation.start, allocation.size)); // after Unmap

    for (auto handle : allocation.pa_handles)
      free_existing_handles.push(handle);

    allocation.pa_handles.clear();
  }

  void free(Addr addr) {
    cout << "Freeing " << addr.size << " bytes at " << addr.start << endl;

    // use find instead of at so we can delete it
    const auto it = mapped_va_ranges.find(addr.start);
    assert(it != mapped_va_ranges.end());
    assert(addr.start == it->second.start);

    unmap_and_free_range(it->second);

    mapped_va_ranges.erase(it);
  }

  ~EndlessVAAllocator() {
    for (auto it = mapped_va_ranges.begin(); it != mapped_va_ranges.end();) {
      unmap_and_free_range(it->second);

      it = mapped_va_ranges.erase(it);
    }

    assert(free_existing_handles.size() == handles.size());

    for (int i = 0; i < handles.size(); i++)
      CHK_DRV(cuMemRelease(handles[i]));
  }
};

// Alternative: "Hybrid"
// VA amoeba allocator backed by remapping PA stack allocator.
// PA-to-VA mapping is one-to-several.
// Each PA block remembers its mapped VA blocks.
//   Whenever a PA block is assigned to an active VA allocation, it also reserves its other mapped VA blocks
//   from the VA amoeba. Downside: more VA fragmentation.

struct FuzzAllocation {
  bool allocated;
  size_t blocks;
  size_t id;
  FuzzAllocation(size_t blocks, size_t id) : blocks{blocks}, id{id}, allocated{false} {}
};

int main() {
  // https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html#interoperability-between-runtime-and-driver-apis
  CHK_RNT(cudaSetDevice(0));

  const size_t num_fuzz_allocations = 100;
  vector<FuzzAllocation> fuzz_allocations{};
  vector<FuzzAllocation *> fuzz_allocation_order{};
  fuzz_allocations.reserve(num_fuzz_allocations);
  fuzz_allocation_order.reserve(2 * num_fuzz_allocations); // visit each twice, once to allocate and once to free
  for (size_t i = 0; i < num_fuzz_allocations; i++) {
    fuzz_allocations.push_back({(i % 10) + 1, i});
    fuzz_allocation_order.push_back(&fuzz_allocations[i]);
    fuzz_allocation_order.push_back(&fuzz_allocations[i]);
  }
  unsigned seed = 42; // chrono::system_clock::now().time_since_epoch().count();
  shuffle(fuzz_allocation_order.begin(), fuzz_allocation_order.end(), std::default_random_engine(seed));

  {
    EndlessVAAllocator allocator{};

    unordered_map<size_t, Addr> active_fuzz_allocations{};

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
      for (auto &[id, addr] : active_fuzz_allocations) {
        touch_va_kernel<<<addr.size / 128, 128>>>((uint8_t *)addr.start);
        CHK_RNT(cudaPeekAtLastError());
        CHK_RNT(cudaDeviceSynchronize());
      }
    }
  }

  CHK_RNT(cudaDeviceSynchronize());

  // minimal experiment
  {
    CUmemAllocationProp prop{};
    int device;
    CHK_RNT(cudaGetDevice(&device));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;

    CUmemAccessDesc access_desc{};
    access_desc.location = prop.location;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    size_t granularity = 0;
    CHK_DRV(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    cout << "granularity = " << granularity << endl;

    const size_t num_allocations = 2;
    const unsigned sleep_ns = 1e6;
    const bool disable_touch = true;

    vector<CUdeviceptr> ptrs;
    vector<CUmemGenericAllocationHandle> handles;
    ptrs.reserve(num_allocations);
    handles.reserve(num_allocations);

    for (size_t i = 0; i < num_allocations; i++) {
      cout << i << endl;
      CUdeviceptr ptr;
      CUmemGenericAllocationHandle handle; 
      CHK_DRV(cuMemAddressReserve(&ptr, granularity, 0ULL, 0U, 0));
      CHK_DRV(cuMemCreate(&handle, granularity, &prop, 0));
      CHK_DRV(cuMemMap(ptr, granularity, 0ULL, handle, 0ULL));
      CHK_DRV(cuMemSetAccess(ptr, granularity, &access_desc, 1ULL));
      touch_va_kernel<<<granularity / 128, 128>>>((uint8_t *)ptr, sleep_ns, disable_touch);
      CHK_RNT(cudaPeekAtLastError());
      // CHK_RNT(cudaDeviceSynchronize()); // Without this and with disable_touch=true, we get IMAs
      CHK_DRV(cuMemUnmap(ptr, granularity)); // overlaps with long touch_va_kernel, rugs ptr
      CHK_DRV(cuMemAddressFree(ptr, granularity));
      ptrs.push_back(ptr);
      handles.push_back(handle);
    }

    CHK_RNT(cudaDeviceSynchronize());

    for (auto handle : handles)
      CHK_DRV(cuMemRelease(handle));

    CHK_RNT(cudaDeviceSynchronize());
  }
}
