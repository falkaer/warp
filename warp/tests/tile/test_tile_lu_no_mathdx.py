# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile LU tests with mathdx solver disabled (cooperative scalar fallback path).

Setting ``enable_mathdx_solver=False`` at module scope routes
``tile_lu``, ``tile_lu_inplace`` and the LU adjoint through the cooperative
scalar implementation in ``tile_lu.h`` on GPU and cooperative CPU blocks,
exercising the path that runs whenever Warp is built without libmathdx or
when a user disables the option per-module.

Mirrors ``test_tile_cholesky_no_mathdx.py``. LU solve coverage with the solver
disabled lives in ``test_tile_solve_no_mathdx.py``.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# Disable mathdx solver ops (LU and triangular solves) for all kernels
# defined in this module.
wp.set_module_options({"enable_mathdx_solver": False})

TILE_DIM = 32
N = 8


# -----------------------------------------------------------------------------
# LU factorization
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def tile_lu_kernel(gA: wp.array2d[wp.float64], gLU: wp.array2d[wp.float64], gP: wp.array1d[wp.int32]):
    A = wp.tile_load(gA, shape=(N, N))
    LU, piv = wp.tile_lu(A)
    wp.tile_store(gLU, LU)
    wp.tile_store(gP, piv)


@wp.kernel(enable_backward=False)
def tile_lu_inplace_kernel(gA: wp.array2d[wp.float64], gP: wp.array1d[wp.int32]):
    A = wp.tile_load(gA, shape=(N, N))
    piv = wp.tile_lu_inplace(A)
    wp.tile_store(gA, A)
    wp.tile_store(gP, piv)


# -----------------------------------------------------------------------------
# LU factorization of a transposed (column-major) tile
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def tile_lu_transposed_kernel(gA: wp.array2d[wp.float64], gLU: wp.array2d[wp.float64], gP: wp.array1d[wp.int32]):
    A = wp.tile_load(gA, shape=(N, N))
    LU, piv = wp.tile_lu(wp.tile_transpose(A))
    wp.tile_store(gLU, LU)
    wp.tile_store(gP, piv)


@wp.kernel(enable_backward=False)
def tile_lu_transposed_inplace_kernel(gA: wp.array2d[wp.float64], gP: wp.array1d[wp.int32]):
    A = wp.tile_load(gA, shape=(N, N))
    A_t = wp.tile_transpose(A)
    piv = wp.tile_lu_inplace(A_t)
    wp.tile_store(gA, A_t)
    wp.tile_store(gP, piv)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _lu_factor_numpy(A):
    """LU factorization with partial pivoting, returning packed factors and 1-based pivots as LAPACK getrf."""
    LU = np.array(A, dtype=np.float64)
    n = LU.shape[0]
    piv = np.zeros(n, dtype=np.int32)
    for j in range(n):
        p = j + int(np.argmax(np.abs(LU[j:, j])))
        piv[j] = p + 1
        LU[[j, p]] = LU[[p, j]]
        LU[j + 1 :, j] /= LU[j, j]
        LU[j + 1 :, j + 1 :] -= np.outer(LU[j + 1 :, j], LU[j, j + 1 :])
    return LU, piv


def _general(n, seed=0):
    """Return a general (non-symmetric) matrix that needs row interchanges, its packed LU factors and pivots."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    LU, piv = _lu_factor_numpy(A)
    return A, LU, piv


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_lu(test, device):
    A_np, LU_ref, P_ref = _general(N, seed=1)

    A = wp.array(A_np, dtype=wp.float64, device=device)
    LU = wp.zeros((N, N), dtype=wp.float64, device=device)
    P = wp.zeros(N, dtype=wp.int32, device=device)

    wp.launch_tiled(tile_lu_kernel, dim=[1], inputs=[A, LU, P], block_dim=TILE_DIM, device=device)
    assert_np_equal(LU.numpy(), LU_ref, tol=1e-10)
    assert_np_equal(P.numpy(), P_ref, tol=0.0)


def test_lu_inplace(test, device):
    A_np, LU_ref, P_ref = _general(N, seed=2)

    A = wp.array(A_np.copy(), dtype=wp.float64, device=device)
    P = wp.zeros(N, dtype=wp.int32, device=device)

    wp.launch_tiled(tile_lu_inplace_kernel, dim=[1], inputs=[A, P], block_dim=TILE_DIM, device=device)
    assert_np_equal(A.numpy(), LU_ref, tol=1e-10)
    assert_np_equal(P.numpy(), P_ref, tol=0.0)


def test_lu_transposed(test, device):
    A_np, _, _ = _general(N, seed=3)
    LU_ref, P_ref = _lu_factor_numpy(A_np.T)

    A = wp.array(A_np, dtype=wp.float64, device=device)
    LU = wp.zeros((N, N), dtype=wp.float64, device=device)
    P = wp.zeros(N, dtype=wp.int32, device=device)

    wp.launch_tiled(tile_lu_transposed_kernel, dim=[1], inputs=[A, LU, P], block_dim=TILE_DIM, device=device)
    assert_np_equal(LU.numpy(), LU_ref, tol=1e-10)
    assert_np_equal(P.numpy(), P_ref, tol=0.0)


def test_lu_transposed_inplace(test, device):
    A_np, _, _ = _general(N, seed=4)
    LU_ref, P_ref = _lu_factor_numpy(A_np.T)

    A = wp.array(A_np.copy(), dtype=wp.float64, device=device)
    P = wp.zeros(N, dtype=wp.int32, device=device)

    wp.launch_tiled(tile_lu_transposed_inplace_kernel, dim=[1], inputs=[A, P], block_dim=TILE_DIM, device=device)
    assert_np_equal(A.numpy(), LU_ref, tol=1e-10)
    assert_np_equal(P.numpy(), P_ref, tol=0.0)


# -----------------------------------------------------------------------------
# LU adjoint -- exercises the cooperative scalar LU adjoint via the
# `enable_mathdx_solver=False` module setting. Kernels here are the same
# shape as the backward kernels in test_tile_lu.py.
# -----------------------------------------------------------------------------


@wp.kernel
def tile_lu_backward_kernel(gA: wp.array2d[wp.float64], gLU: wp.array2d[wp.float64]):
    A = wp.tile_load(gA, shape=(N, N), storage="shared")
    LU, _piv = wp.tile_lu(A)
    wp.tile_store(gLU, LU)


@wp.kernel
def tile_lu_transposed_backward_kernel(gA: wp.array2d[wp.float64], gLU: wp.array2d[wp.float64]):
    A = wp.tile_load(gA, shape=(N, N), storage="shared")
    LU, _piv = wp.tile_lu(wp.tile_transpose(A))
    wp.tile_store(gLU, LU)


def _lu_adjoint_numpy(LU, piv, adj_LU):
    n = LU.shape[0]
    L = np.tril(LU, -1) + np.eye(n)
    U = np.triu(LU)
    G = np.tril(L.T @ np.tril(adj_LU, -1), -1) + np.triu(np.triu(adj_LU) @ U.T)
    X = np.linalg.solve(L.T, G)
    grad_Ap = np.linalg.solve(U, X.T).T
    sigma = np.arange(n)
    for k, p in enumerate(piv - 1):
        sigma[[k, p]] = sigma[[p, k]]
    grad_A = np.zeros_like(grad_Ap)
    grad_A[sigma] = grad_Ap
    return grad_A


def test_lu_backward(test, device):
    A_np, LU_ref, P_ref = _general(N, seed=20)
    rng = np.random.default_rng(21)
    adj_LU = rng.standard_normal((N, N))

    A = wp.array(A_np, dtype=wp.float64, requires_grad=True, device=device)
    LU = wp.zeros((N, N), dtype=wp.float64, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_lu_backward_kernel, dim=[1], inputs=[A, LU], block_dim=TILE_DIM, device=device)
    tape.backward(grads={LU: wp.array(adj_LU, dtype=wp.float64, device=device)})

    grad_A_ref = _lu_adjoint_numpy(LU_ref, P_ref, adj_LU)
    assert_np_equal(LU.numpy(), LU_ref, tol=1e-10)
    assert_np_equal(A.grad.numpy(), grad_A_ref, tol=1e-8)


def test_lu_transposed_backward(test, device):
    A_np, _, _ = _general(N, seed=22)
    LU_ref, P_ref = _lu_factor_numpy(A_np.T)
    rng = np.random.default_rng(23)
    adj_LU = rng.standard_normal((N, N))

    A = wp.array(A_np, dtype=wp.float64, requires_grad=True, device=device)
    LU = wp.zeros((N, N), dtype=wp.float64, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_lu_transposed_backward_kernel, dim=[1], inputs=[A, LU], block_dim=TILE_DIM, device=device)
    tape.backward(grads={LU: wp.array(adj_LU, dtype=wp.float64, device=device)})

    grad_A_ref = _lu_adjoint_numpy(LU_ref, P_ref, adj_LU).T
    assert_np_equal(LU.numpy(), LU_ref, tol=1e-10)
    assert_np_equal(A.grad.numpy(), grad_A_ref, tol=1e-8)


# -----------------------------------------------------------------------------
# Larger-N adjoint smoke -- exercises the cooperative scalar LU adjoint's
# shared-mem scratch budget at a non-trivial tile size. Two __shared__ T W[n*n]
# buffers in float64 total 16 KiB at n=32 (before counting the shared input,
# output and pivot tiles). Catch budget failures in CI rather than at runtime.
# -----------------------------------------------------------------------------

N32 = 32


@wp.kernel
def tile_lu_backward_n32_kernel(gA: wp.array2d[wp.float64], gLU: wp.array2d[wp.float64]):
    A = wp.tile_load(gA, shape=(N32, N32), storage="shared")
    LU, _piv = wp.tile_lu(A)
    wp.tile_store(gLU, LU)


def test_lu_backward_n32(test, device):
    A_np, LU_ref, P_ref = _general(N32, seed=40)
    rng = np.random.default_rng(41)
    adj_LU = rng.standard_normal((N32, N32))

    A = wp.array(A_np, dtype=wp.float64, requires_grad=True, device=device)
    LU = wp.zeros((N32, N32), dtype=wp.float64, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_lu_backward_n32_kernel, dim=[1], inputs=[A, LU], block_dim=TILE_DIM, device=device)
    tape.backward(grads={LU: wp.array(adj_LU, dtype=wp.float64, device=device)})

    grad_A_ref = _lu_adjoint_numpy(LU_ref, P_ref, adj_LU)
    assert_np_equal(LU.numpy(), LU_ref, tol=1e-10)
    assert_np_equal(A.grad.numpy(), grad_A_ref, tol=1e-7)


# This kernel lives in its own module so that the block_dim == 1 launch
# specializes only it instead of recompiling every kernel in this module,
# including the unrelated N=32 LU adjoint. The module-scope
# enable_mathdx_solver option does not carry over to a unique module, so it is
# repeated here. The launch is always a single block, so the grid-stride
# wrapper is not needed either.
@wp.kernel(enable_backward=False, grid_stride=False, module="unique", module_options={"enable_mathdx_solver": False})
def tile_lu_isolated_kernel(gA: wp.array2d[wp.float64], gLU: wp.array2d[wp.float64], gP: wp.array1d[wp.int32]):
    A = wp.tile_load(gA, shape=(N, N))
    LU, piv = wp.tile_lu(A)
    wp.tile_store(gLU, LU)
    wp.tile_store(gP, piv)


def test_lu_block_dim_1(test, device):
    """Check the LU factorization with a single-thread block.

    At ``block_dim == 1`` the thread-strided loops collapse to sequential, but
    the codegen path must still compile and run correctly on every device.
    """
    A_np, LU_ref, P_ref = _general(N, seed=42)

    A = wp.array(A_np, dtype=wp.float64, device=device)
    LU = wp.zeros((N, N), dtype=wp.float64, device=device)
    P = wp.zeros(N, dtype=wp.int32, device=device)

    wp.launch_tiled(tile_lu_isolated_kernel, dim=[1], inputs=[A, LU, P], block_dim=1, device=device)
    assert_np_equal(LU.numpy(), LU_ref, tol=1e-10)
    assert_np_equal(P.numpy(), P_ref, tol=0.0)


# -----------------------------------------------------------------------------
# Suite registration
# -----------------------------------------------------------------------------


class TestTileLUNoMathDx(unittest.TestCase):
    pass


_devices = get_test_devices()

lu_tests = [
    ("test_lu", test_lu),
    ("test_lu_inplace", test_lu_inplace),
    ("test_lu_transposed", test_lu_transposed),
    ("test_lu_transposed_inplace", test_lu_transposed_inplace),
    ("test_lu_backward", test_lu_backward),
    ("test_lu_transposed_backward", test_lu_transposed_backward),
    ("test_lu_backward_n32", test_lu_backward_n32),
    ("test_lu_block_dim_1", test_lu_block_dim_1),
]

for name, func in lu_tests:
    add_function_test(TestTileLUNoMathDx, name, func, devices=_devices, check_output=False)

for name, func in lu_tests:
    if name.endswith("block_dim_1"):
        continue
    add_function_test(
        TestTileLUNoMathDx,
        f"{name}_cpu_blocks",
        func,
        devices=get_cpu_test_devices(),
        check_output=False,
        enable_cpu_blocks=True,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
