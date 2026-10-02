// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// grid_sync: grid-wide barrier for kernels launched cooperatively
// (@wp.kernel(cooperative=True)), equivalent to
// cooperative_groups::this_grid().sync().
//
// Warp compiles kernels with NVRTC without the CUDA Toolkit include path, so
// this reproduces the barrier of cooperative_groups/details/sync.h and
// details/driver_abi.h directly. For a cooperative launch the driver allocates
// a per-grid workspace { unsigned int size; unsigned int barrier; } and passes
// its address in the %envreg1 (high) and %envreg2 (low) special registers.
//
// One thread per block arrives on the barrier word: block 0 adds
// 0x80000000 - (blocks - 1) and every other block adds 1, so the arrivals of
// one barrier sum to 0x80000000 and flip the top bit exactly once. Each block
// waits until the top bit differs from the value it observed on arrival. The
// arrival is a release and the polling load an acquire at GPU scope, so writes
// made before the barrier by any block are visible to every block after it.

#pragma once

#include "builtin.h"

namespace wp {

#if defined(__CUDA_ARCH__)

inline CUDA_CALLABLE unsigned int* grid_sync_barrier()
{
    unsigned int hi, lo;
    asm("mov.u32 %0, %%envreg1;" : "=r"(hi));
    asm("mov.u32 %0, %%envreg2;" : "=r"(lo));
    unsigned int* workspace = reinterpret_cast<unsigned int*>((static_cast<unsigned long long>(hi) << 32) | lo);
    return workspace + 1;
}

inline CUDA_CALLABLE void grid_sync()
{
    __syncthreads();

    if (threadIdx.x + threadIdx.y + threadIdx.z == 0) {
        unsigned int* barrier = grid_sync_barrier();
        const unsigned int blocks = gridDim.x * gridDim.y * gridDim.z;
        const bool first_block = blockIdx.x + blockIdx.y + blockIdx.z == 0;
        const unsigned int arrival = first_block ? 0x80000000u - (blocks - 1u) : 1u;

        unsigned int old_value, value;
#if __CUDA_ARCH__ >= 700
        asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;" : "=r"(old_value) : "l"(barrier), "r"(arrival) : "memory");
        do {
            asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(value) : "l"(barrier) : "memory");
        } while (((old_value ^ value) & 0x80000000u) == 0);
#else
        __threadfence();
        old_value = atomicAdd(barrier, arrival);
        do {
            value = *static_cast<volatile unsigned int*>(barrier);
        } while (((old_value ^ value) & 0x80000000u) == 0);
        __threadfence();
#endif
    }

    __syncthreads();
}

#else

// Cooperative kernels are rejected at launch on CPU devices.
inline CUDA_CALLABLE void grid_sync() { }

#endif

}  // namespace wp
