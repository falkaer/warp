# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import (
    add_function_test,
    get_cpu_test_devices,
    get_cuda_test_devices,
    get_test_devices,
)

BLOCK_DIM = 64


# Pairwise tree reduction over every block of the grid: after level k, data[i]
# holds the sum of data[i : i + 2^(k+1)] for every i divisible by 2^(k+1). Each
# level reads values written by other blocks in the previous level, so the
# result is exact only if wp.grid_sync() orders the levels across the grid.
@wp.kernel(cooperative=True, grid_stride=False, enable_backward=False)
def tree_sum(data: wp.array[wp.int32], n: int, levels: int):
    i = wp.tid()
    stride = int(1)
    for _level in range(levels):
        if i % (2 * stride) == 0 and i + stride < n:
            data[i] = data[i] + data[i + stride]
        wp.grid_sync()
        stride = stride * 2


@wp.func
def reduction_level(data: wp.array[wp.int32], n: int, i: int, stride: int):
    if i % (2 * stride) == 0 and i + stride < n:
        data[i] = data[i] + data[i + stride]
    wp.grid_sync()


# Same reduction with the barrier inside a user function.
@wp.kernel(cooperative=True, grid_stride=False, enable_backward=False)
def tree_sum_func(data: wp.array[wp.int32], n: int, levels: int):
    i = wp.tid()
    stride = int(1)
    for _level in range(levels):
        reduction_level(data, n, i, stride)
        stride = stride * 2


def run_tree_sum(kernel, device, num_blocks, block_dim=BLOCK_DIM, seed=0):
    """Launch ``kernel`` over ``num_blocks`` blocks and return (data, expected sum)."""
    n = num_blocks * block_dim
    values = np.random.default_rng(seed).integers(0, 100, n).astype(np.int32)
    data = wp.array(values, dtype=wp.int32, device=device)
    levels = max(int(np.ceil(np.log2(n))), 1)
    return data, n, levels, values


# -----------------------------------------------------------------------------
# Device-parametrized tests (registered at the bottom via add_function_test).
# -----------------------------------------------------------------------------


def test_grid_sync_reduction(test, device):
    """Reduce across the largest cooperative grid; levels depend on other blocks' writes."""
    max_blocks = wp.get_cuda_max_cooperative_blocks(tree_sum, device, block_dim=BLOCK_DIM)
    test.assertGreater(max_blocks, 1)

    for num_blocks in (1, 2, max_blocks):
        with test.subTest(num_blocks=num_blocks):
            data, n, levels, values = run_tree_sum(tree_sum, device, num_blocks)
            wp.launch(tree_sum, dim=n, inputs=[data, n, levels], block_dim=BLOCK_DIM, device=device)
            test.assertEqual(int(data.numpy()[0]), int(values.sum()))


def test_grid_sync_in_function(test, device):
    """A wp.grid_sync() inside a wp.func synchronizes the grid of the calling cooperative kernel."""
    max_blocks = wp.get_cuda_max_cooperative_blocks(tree_sum_func, device, block_dim=BLOCK_DIM)
    data, n, levels, values = run_tree_sum(tree_sum_func, device, max_blocks, seed=1)
    wp.launch(tree_sum_func, dim=n, inputs=[data, n, levels], block_dim=BLOCK_DIM, device=device)
    test.assertEqual(int(data.numpy()[0]), int(values.sum()))


def test_max_cooperative_blocks(test, device):
    """The cooperative limit is the per-SM occupancy times the SM count, and shrinks with larger blocks."""
    small = wp.get_cuda_max_cooperative_blocks(tree_sum, device, block_dim=64)
    large = wp.get_cuda_max_cooperative_blocks(tree_sum, device, block_dim=1024)
    test.assertGreater(small, 0)
    test.assertEqual(small % device.sm_count, 0)
    test.assertEqual(large % device.sm_count, 0)
    test.assertGreaterEqual(small, large)


def test_cooperative_launch_too_large(test, device):
    """A grid with one block more than the cooperative limit is rejected with the limit in the message."""
    max_blocks = wp.get_cuda_max_cooperative_blocks(tree_sum, device, block_dim=BLOCK_DIM)
    data, n, levels, _values = run_tree_sum(tree_sum, device, max_blocks + 1)
    with test.assertRaisesRegex(RuntimeError, f"exceeds the {max_blocks} blocks"):
        wp.launch(tree_sum, dim=n, inputs=[data, n, levels], block_dim=BLOCK_DIM, device=device)


def test_cooperative_kernel_in_cuda_graph(test, device):
    """A cooperative launch captured into a CUDA graph synchronizes the grid on replay."""
    max_blocks = wp.get_cuda_max_cooperative_blocks(tree_sum, device, block_dim=BLOCK_DIM)
    data, n, levels, values = run_tree_sum(tree_sum, device, max_blocks, seed=2)

    # Preload so no module load happens during capture.
    tree_sum.module.load(device, BLOCK_DIM)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        wp.launch(tree_sum, dim=n, inputs=[data, n, levels], block_dim=BLOCK_DIM, device=device)

    for _replay in range(3):
        data.assign(values)
        wp.capture_launch(capture.graph)
        test.assertEqual(int(data.numpy()[0]), int(values.sum()))


def test_apic_save_load_preserves_cooperative(test, device):
    """A cooperative launch survives APIC capture/save/load and is replayed cooperatively."""
    max_blocks = wp.get_cuda_max_cooperative_blocks(tree_sum, device, block_dim=BLOCK_DIM)
    data, n, levels, values = run_tree_sum(tree_sum, device, max_blocks, seed=3)

    # Preload so no module load happens during graph capture.
    module_exec = tree_sum.module.load(device, BLOCK_DIM)
    module_exec.get_kernel_hooks(tree_sum)

    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(tree_sum, dim=n, inputs=[data, n, levels], block_dim=BLOCK_DIM, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cooperative_save_load")
        wp.capture_save(capture.graph, path, inputs={"data": data}, outputs={"data": data})

        loaded = wp.capture_load(path, device=device)
        loaded.set_param("data", wp.array(values, dtype=wp.int32, device=device))
        wp.capture_launch(loaded)

        result = wp.zeros(n, dtype=wp.int32, device=device)
        loaded.get_param("data", result)
        test.assertEqual(int(result.numpy()[0]), int(values.sum()))


def test_cooperative_launch_on_cpu_rejected(test, device):
    """Cooperative kernels are supported on CUDA devices only."""
    data = wp.zeros(4, dtype=wp.int32, device=device)
    with test.assertRaisesRegex(RuntimeError, "supported on CUDA devices only"):
        wp.launch(tree_sum, dim=4, inputs=[data, 4, 2], device=device)


def test_max_cooperative_blocks_on_cpu_is_zero(test, device):
    test.assertEqual(wp.get_cuda_max_cooperative_blocks(tree_sum, device), 0)


def test_get_cuda_max_cooperative_blocks_preserves_module_block_dim(test, device):
    """Verify ``get_cuda_max_cooperative_blocks`` restores ``module.options["block_dim"]`` after probing."""

    @wp.kernel(cooperative=True, grid_stride=False, module="unique")
    def k(a: wp.array[int]):
        a[wp.tid()] = 0

    prior = k.module.options["block_dim"]
    wp.get_cuda_max_cooperative_blocks(k, device, block_dim=512)
    test.assertEqual(k.module.options["block_dim"], prior)


class TestCooperative(unittest.TestCase):
    """``cooperative`` kernel option and ``wp.grid_sync()``: validation and the codegen check.

    Device-dependent behavior (grid synchronization, the occupancy query, launch limits,
    CUDA graph and APIC capture) is registered below via ``add_function_test``.
    """

    def test_cooperative_accepted(self):
        # Default: no option key is added, so existing kernels are unaffected.
        @wp.kernel(module="unique")
        def k_default(a: wp.array[int]):
            a[wp.tid()] = 0

        self.assertNotIn("cooperative", k_default.options)

        @wp.kernel(cooperative=True, grid_stride=False, module="unique")
        def k_cooperative(a: wp.array[int]):
            a[wp.tid()] = 0

        self.assertTrue(k_cooperative.options["cooperative"])

    def test_invalid_cooperative_rejected(self):
        # A grid-stride loop would run the barrier a different number of times per thread.
        for grid_stride in (None, True):
            with self.subTest(grid_stride=grid_stride), self.assertRaisesRegex(ValueError, "grid_stride=False"):

                @wp.kernel(cooperative=True, grid_stride=grid_stride, module="unique")
                def k(a: wp.array[int]):
                    a[wp.tid()] = 0

        # The driver's cooperative launch does not support thread block clusters.
        with self.assertRaisesRegex(ValueError, "clusters"):

            @wp.kernel(cooperative=True, grid_stride=False, cluster_dim=2, module="unique")
            def k_cluster(a: wp.array[int]):
                a[wp.tid()] = 0

    def test_grid_sync_requires_cooperative_kernel(self):
        # Rejected when building the module, on any device.
        @wp.kernel(grid_stride=False, module="unique")
        def k_direct(a: wp.array[int]):
            wp.grid_sync()

        @wp.func
        def synchronize():
            wp.grid_sync()

        @wp.kernel(grid_stride=False, module="unique")
        def k_function(a: wp.array[int]):
            synchronize()

        for kernel, where in ((k_direct, "In kernel"), (k_function, "In function 'synchronize' called from kernel")):
            with self.subTest(kernel=kernel.key):
                a = wp.zeros(1, dtype=int, device="cpu")
                with self.assertRaisesRegex(wp.WarpCodegenError, f"{where}.*requires a cooperative kernel"):
                    wp.launch(kernel, dim=1, inputs=[a], device="cpu")


cuda_devices = get_cuda_test_devices()
cpu_devices = get_cpu_test_devices()

add_function_test(TestCooperative, "test_grid_sync_reduction", test_grid_sync_reduction, devices=cuda_devices)
add_function_test(TestCooperative, "test_grid_sync_in_function", test_grid_sync_in_function, devices=cuda_devices)
add_function_test(TestCooperative, "test_max_cooperative_blocks", test_max_cooperative_blocks, devices=cuda_devices)
add_function_test(
    TestCooperative, "test_cooperative_launch_too_large", test_cooperative_launch_too_large, devices=cuda_devices
)
add_function_test(
    TestCooperative,
    "test_cooperative_kernel_in_cuda_graph",
    test_cooperative_kernel_in_cuda_graph,
    devices=cuda_devices,
)
add_function_test(
    TestCooperative,
    "test_apic_save_load_preserves_cooperative",
    test_apic_save_load_preserves_cooperative,
    devices=cuda_devices,
)
add_function_test(
    TestCooperative,
    "test_cooperative_launch_on_cpu_rejected",
    test_cooperative_launch_on_cpu_rejected,
    devices=cpu_devices,
)
add_function_test(
    TestCooperative,
    "test_max_cooperative_blocks_on_cpu_is_zero",
    test_max_cooperative_blocks_on_cpu_is_zero,
    devices=cpu_devices,
)
add_function_test(
    TestCooperative,
    "test_get_cuda_max_cooperative_blocks_preserves_module_block_dim",
    test_get_cuda_max_cooperative_blocks_preserves_module_block_dim,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
