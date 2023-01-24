#include <cstdint>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>


#include "kernel.hpp"

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <class T>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
	  mySum += __shfl_down_sync(mask, mySum, offset);
  }

  return mySum;
}

// Performs a reduction step and updates numTotal with how many are remaining
template <typename T, typename Group>
__device__ T cg_reduce_n(T in, Group &threads) {
  return cg::reduce(threads, in, cg::plus<T>());
}

template <class T>
__global__ void cg_reduce(T *g_idata, T *g_odata, unsigned int n) {
  // Shared memory for intermediate steps
  T *sdata = SharedMemory<T>();
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Handle to tile in thread block
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

  unsigned int ctaSize = cta.size();
  unsigned int numCtas = gridDim.x;
  unsigned int threadRank = cta.thread_rank();
  unsigned int threadIndex = (blockIdx.x * ctaSize) + threadRank;

  T threadVal = 0;
  {
    unsigned int i = threadIndex;
    unsigned int indexStride = (numCtas * ctaSize);
    while (i < n) {
      threadVal += g_idata[i];
      i += indexStride;
    }
    sdata[threadRank] = threadVal;
  }

  // Wait for all tiles to finish and reduce within CTA
  {
    unsigned int ctaSteps = tile.meta_group_size();
    unsigned int ctaIndex = ctaSize >> 1;
    while (ctaIndex >= 32) {
      cta.sync();
      if (threadRank < ctaIndex) {
        threadVal += sdata[threadRank + ctaIndex];
        sdata[threadRank] = threadVal;
      }
      ctaSteps >>= 1;
      ctaIndex >>= 1;
    }
  }

  // Shuffle redux instead of smem redux
  {
    cta.sync();
    if (tile.meta_group_rank() == 0) {
      threadVal = cg_reduce_n(threadVal, tile);
    }
  }

  if (threadRank == 0) g_odata[blockIdx.x] = threadVal;
}

template <class T, size_t BlockSize, size_t MultiWarpGroupSize>
__global__ void multi_warp_cg_reduce(T *g_idata, T *g_odata, unsigned int n) {
  // Shared memory for intermediate steps
  T *sdata = SharedMemory<T>();
//   __shared__ cg::block_tile_memory<BlockSize> scratch;

  // Handle to thread block group
//   auto cta = cg::this_thread_block(scratch);
  auto cta = cg::this_thread_block();
  // Handle to multiWarpTile in thread block
  auto multiWarpTile = cg::tiled_partition<MultiWarpGroupSize>(cta);

  unsigned int gridSize = BlockSize * gridDim.x;
  T threadVal = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  int nIsPow2 = !(n & n-1);
  if (nIsPow2) {
    unsigned int i = blockIdx.x * BlockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      threadVal += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + BlockSize) < n) {
        threadVal += g_idata[i + blockDim.x];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * BlockSize + threadIdx.x;
    while (i < n) {
      threadVal += g_idata[i];
      i += gridSize;
    }
  }

  threadVal = cg_reduce_n(threadVal, multiWarpTile);

  if (multiWarpTile.thread_rank() == 0) {
    sdata[multiWarpTile.meta_group_rank()] = threadVal;
  }
  cg::sync(cta);

  if (threadIdx.x == 0) {
    threadVal = 0;
    for (int i=0; i < multiWarpTile.meta_group_size(); i++) {
      threadVal += sdata[i];
    }
    g_odata[blockIdx.x] = threadVal;
  }
}


template <class T>
void gpu_reduce(int size, int threads, int blocks, T* d_idata, T* d_odata) {
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of boundss
  int smemSize =
	  (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	cg_reduce<T><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
}


template <typename T>
T sum(T* data, size_t size) {
	T* output;
	cudaMalloc(&output, size * sizeof(T));
	cudaHostRegister(data, size * sizeof(T), cudaHostRegisterDefault);

	const size_t block_size = 256;
	const size_t grid_size = (size + block_size - 1) / block_size;

	gpu_reduce<T>(size, block_size, grid_size, data, output);

	int s = grid_size;
	while (s > 1) {
		gpu_reduce<T>(grid_size, block_size, grid_size, output, output);
		s = (s + (block_size * 2 - 1)) / (block_size * 2);
	}

  T result;
  cudaMemcpy(&result, output, sizeof(T), cudaMemcpyDeviceToHost);
	cudaFree(output);
  cudaHostUnregister(data);

	return result;
}

template double sum(double*, size_t);
template float sum(float*, size_t);
template int sum(int*, size_t);

// #include <iostream>
// #include <chrono>

// int main() {
// 	using dt = float;

// 	size_t array_size = 10000;
// 	dt* data = new dt[array_size];

// 	for (uint64_t i = 0; i < array_size; i++) {
// 		data[i] = i + 1;
// 	}


// 	auto start = std::chrono::high_resolution_clock::now();

// 	// gpu_reduce<dt>(array_size, block_size, grid_size, data, data);

// 	// int s = grid_size;
// 	// while (s > 1) {
// 	// 	gpu_reduce<dt>(grid_size, block_size, grid_size, data, data);
// 	// 	s = (s + (block_size * 2 - 1)) / (block_size * 2);
// 	// }

// 	auto res = sum<dt>(data, array_size);

// 	cudaDeviceSynchronize();
	
// 	auto end = std::chrono::high_resolution_clock::now();
// 	std::cout << "output: " << res << ", expected: " << ((float)array_size * (array_size + 1)) / 2 << std::endl;
// 	std::cout << "took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
// }
