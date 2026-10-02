# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.init()  # For wp._src.context.runtime.core.wp_is_mathdx_enabled()

TILE_M = wp.constant(8)

# num threads per-tile
TILE_DIM = 32

# Forward-only kernels skip adjoint codegen
wp.get_module("test_lu_fwd").options["enable_backward"] = False


def lu_factor_numpy(A):
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


def lu_permutation_numpy(piv):
    """Return sigma such that (P^T A)[i] = A[sigma[i]] for the 1-based pivots ``piv``."""
    sigma = np.arange(len(piv))
    for k, p in enumerate(piv - 1):
        sigma[[k, p]] = sigma[[p, k]]
    return sigma


@wp.kernel(module="test_lu_fwd")
def tile_math_lu(
    gA: wp.array2d[wp.float64],
    gD: wp.array1d[wp.float64],
    gLU: wp.array2d[wp.float64],
    gP: wp.array1d[wp.int32],
    gy: wp.array1d[wp.float64],
    gx: wp.array1d[wp.float64],
):
    # Load A, D & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    d = wp.tile_load(gD, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    # Ensure tile_diag_add() and tile_lu_solve() work with transposed matrices
    a_t = wp.tile_transpose(a)
    # Compute LU, piv st P^T (A^T + diag(D)) = LU
    b = wp.tile_diag_add(a_t, d)
    lu, piv = wp.tile_lu(b)
    # Solve for x in (A^T + diag(D)) x = y
    x = wp.tile_lu_solve(lu, piv, y)
    # Store LU, piv & x
    wp.tile_store(gLU, lu)
    wp.tile_store(gP, piv)
    wp.tile_store(gx, x)


def test_tile_lu_factor_and_solve(test, device):
    """Compute an LU factorization with partial pivoting and solve a vector right-hand side."""
    rng = np.random.default_rng(42)
    A_h = rng.standard_normal((TILE_M, TILE_M))
    D_h = rng.standard_normal(TILE_M)
    Y_h = np.arange(TILE_M, dtype=np.float64)

    A_np = A_h.T + np.diag(D_h)
    LU_np, P_np = lu_factor_numpy(A_np)
    X_np = np.linalg.solve(A_np, Y_h)

    A_wp = wp.array(A_h, dtype=wp.float64, device=device)
    D_wp = wp.array(D_h, dtype=wp.float64, device=device)
    LU_wp = wp.zeros((TILE_M, TILE_M), dtype=wp.float64, device=device)
    P_wp = wp.zeros(TILE_M, dtype=wp.int32, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
    X_wp = wp.zeros_like(Y_wp)

    wp.launch_tiled(
        tile_math_lu, dim=[1, 1], inputs=[A_wp, D_wp, LU_wp, P_wp, Y_wp, X_wp], block_dim=TILE_DIM, device=device
    )

    np.testing.assert_array_equal(P_wp.numpy(), P_np)
    np.testing.assert_allclose(LU_wp.numpy(), LU_np)
    np.testing.assert_allclose(X_wp.numpy(), X_np)


@wp.kernel(module="test_lu_fwd")
def tile_math_lu_inplace(
    gA: wp.array2d[wp.float64],
    gP: wp.array1d[wp.int32],
    gy: wp.array1d[wp.float64],
):
    # Load A & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    # Compute LU, piv st P^T A = LU inplace
    piv = wp.tile_lu_inplace(a)
    # Solve for x in A x = y inplace
    wp.tile_lu_solve_inplace(a, piv, y)
    # Store LU, piv & x
    wp.tile_store(gA, a)
    wp.tile_store(gP, piv)
    wp.tile_store(gy, y)


def test_tile_lu_factor_and_solve_inplace(test, device):
    """Compute an LU factorization with partial pivoting and solve a vector right-hand side in place."""
    rng = np.random.default_rng(42)
    A_h = rng.standard_normal((TILE_M, TILE_M))
    Y_h = np.arange(TILE_M, dtype=np.float64)

    LU_np, P_np = lu_factor_numpy(A_h)
    Y_sol_np = np.linalg.solve(A_h, Y_h)

    A_wp = wp.array(A_h, dtype=wp.float64, device=device)
    P_wp = wp.zeros(TILE_M, dtype=wp.int32, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)

    wp.launch_tiled(tile_math_lu_inplace, dim=[1, 1], inputs=[A_wp, P_wp, Y_wp], block_dim=TILE_DIM, device=device)

    np.testing.assert_array_equal(P_wp.numpy(), P_np)
    np.testing.assert_allclose(A_wp.numpy(), LU_np)
    np.testing.assert_allclose(Y_wp.numpy(), Y_sol_np)


@wp.kernel(module="test_lu_fwd")
def tile_math_lu_multiple_rhs(
    gA: wp.array2d[wp.float64],
    gD: wp.array1d[wp.float64],
    gLU: wp.array2d[wp.float64],
    gP: wp.array1d[wp.int32],
    gy: wp.array2d[wp.float64],
    gx: wp.array2d[wp.float64],
    gz: wp.array2d[wp.float64],
):
    # Load A, D & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    d = wp.tile_load(gD, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=(TILE_M, TILE_M), storage="shared")
    # Ensure tile_diag_add() and tile_lu_solve() work with transposed matrices
    a_t = wp.tile_transpose(a)
    # Compute LU, piv st P^T (A^T + diag(D)) = LU
    b = wp.tile_diag_add(a_t, d)
    lu, piv = wp.tile_lu(b)
    # Solve for x in (A^T + diag(D)) x = y.T
    y_t = wp.tile_transpose(y)
    x = wp.tile_lu_solve(lu, piv, y_t)
    # Ensure matmul receives correct layout information
    z = wp.tile_matmul(x, x)
    # Store LU, piv, x & z
    wp.tile_store(gLU, lu)
    wp.tile_store(gP, piv)
    wp.tile_store(gx, x)
    wp.tile_store(gz, z)


def test_tile_lu_factor_and_solve_multiple_rhs(test, device):
    """Compute an LU factorization with partial pivoting and solve multiple right-hand sides."""
    rng = np.random.default_rng(42)
    A_h = rng.standard_normal((TILE_M, TILE_M))
    D_h = rng.standard_normal(TILE_M)
    Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))

    A_np = A_h.T + np.diag(D_h)
    LU_np, P_np = lu_factor_numpy(A_np)
    X_np = np.linalg.solve(A_np, Y_h.T)
    Z_np = X_np @ X_np

    A_wp = wp.array(A_h, dtype=wp.float64, device=device)
    D_wp = wp.array(D_h, dtype=wp.float64, device=device)
    LU_wp = wp.zeros((TILE_M, TILE_M), dtype=wp.float64, device=device)
    P_wp = wp.zeros(TILE_M, dtype=wp.int32, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
    X_wp = wp.zeros_like(Y_wp)
    Z_wp = wp.zeros_like(Y_wp)

    wp.launch_tiled(
        tile_math_lu_multiple_rhs,
        dim=[1, 1],
        inputs=[A_wp, D_wp, LU_wp, P_wp, Y_wp, X_wp, Z_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    np.testing.assert_array_equal(P_wp.numpy(), P_np)
    np.testing.assert_allclose(LU_wp.numpy(), LU_np)
    np.testing.assert_allclose(X_wp.numpy(), X_np)
    np.testing.assert_allclose(Z_wp.numpy(), Z_np)


@wp.kernel(module="test_lu_fwd")
def tile_math_lu_multiple_rhs_inplace(
    gA: wp.array2d[wp.float64],
    gP: wp.array1d[wp.int32],
    gy: wp.array2d[wp.float64],
    gz: wp.array2d[wp.float64],
):
    # Load A & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=(TILE_M, TILE_M), storage="shared")
    # Compute LU, piv st P^T A = LU inplace
    piv = wp.tile_lu_inplace(a)
    # Solve for x in A x = y.T inplace
    y_t = wp.tile_transpose(y)
    wp.tile_lu_solve_inplace(a, piv, y_t)
    y = wp.tile_transpose(y_t)
    # Ensure matmul receives correct layout information
    z = wp.tile_matmul(y, y)
    # Store LU, piv, y & z
    wp.tile_store(gA, a)
    wp.tile_store(gP, piv)
    wp.tile_store(gy, y)
    wp.tile_store(gz, z)


def test_tile_lu_factor_and_solve_multiple_rhs_inplace(test, device):
    """Compute an LU factorization with partial pivoting and solve multiple right-hand sides in place."""
    rng = np.random.default_rng(42)
    A_h = rng.standard_normal((TILE_M, TILE_M))
    Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))

    LU_np, P_np = lu_factor_numpy(A_h)
    Y_sol_np = np.linalg.solve(A_h, Y_h.T).T
    Z_np = Y_sol_np @ Y_sol_np

    A_wp = wp.array(A_h, dtype=wp.float64, device=device)
    P_wp = wp.zeros(TILE_M, dtype=wp.int32, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
    Z_wp = wp.zeros_like(Y_wp)

    wp.launch_tiled(
        tile_math_lu_multiple_rhs_inplace,
        dim=[1, 1],
        inputs=[A_wp, P_wp, Y_wp, Z_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    np.testing.assert_array_equal(P_wp.numpy(), P_np)
    np.testing.assert_allclose(A_wp.numpy(), LU_np)
    np.testing.assert_allclose(Y_wp.numpy(), Y_sol_np)
    np.testing.assert_allclose(Z_wp.numpy(), Z_np)


@wp.kernel()
def tile_lu_backward_kernel(
    gA: wp.array2d[Any],
    gLU: wp.array2d[Any],
):
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    lu, _piv = wp.tile_lu(a)
    wp.tile_store(gLU, lu)


wp.overload(tile_lu_backward_kernel, {"gA": wp.array2d[wp.float32], "gLU": wp.array2d[wp.float32]})
wp.overload(tile_lu_backward_kernel, {"gA": wp.array2d[wp.float64], "gLU": wp.array2d[wp.float64]})


@wp.kernel()
def tile_lu_transposed_backward_kernel(
    gA: wp.array2d[Any],
    gLU: wp.array2d[Any],
):
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    lu, _piv = wp.tile_lu(wp.tile_transpose(a))
    wp.tile_store(gLU, lu)


wp.overload(tile_lu_transposed_backward_kernel, {"gA": wp.array2d[wp.float32], "gLU": wp.array2d[wp.float32]})
wp.overload(tile_lu_transposed_backward_kernel, {"gA": wp.array2d[wp.float64], "gLU": wp.array2d[wp.float64]})


def lu_adjoint_numpy(LU, piv, adj_LU):
    """Analytic adjoint of the packed LU factorization with the pivots held fixed."""
    n = LU.shape[0]
    L = np.tril(LU, -1) + np.eye(n)
    U = np.triu(LU)
    G = np.tril(L.T @ np.tril(adj_LU, -1), -1) + np.triu(np.triu(adj_LU) @ U.T)
    X = np.linalg.solve(L.T, G)
    grad_Ap = np.linalg.solve(U, X.T).T
    grad_A = np.zeros_like(grad_Ap)
    grad_A[lu_permutation_numpy(piv)] = grad_Ap
    return grad_A


def _lu_backward(A_np, adj_LU, device, wp_dtype=wp.float64, transposed=False):
    """Run tile LU forward+backward, return (LU_wp, grad_A) as NumPy arrays."""
    np_dtype = wp.dtype_to_numpy(wp_dtype)
    A_wp = wp.array(A_np.astype(np_dtype), dtype=wp_dtype, requires_grad=True, device=device)
    LU_wp = wp.zeros((TILE_M, TILE_M), dtype=wp_dtype, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_lu_transposed_backward_kernel if transposed else tile_lu_backward_kernel,
            dim=[1, 1],
            inputs=[A_wp, LU_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    tape.backward(grads={LU_wp: wp.array(adj_LU.astype(np_dtype), dtype=wp_dtype, device=device)})
    return LU_wp.numpy(), A_wp.grad.numpy()


def _test_tile_lu_backward(transposed):
    def test_tile_lu_backward(dtype):
        def test(test, device):
            np_dtype = wp.dtype_to_numpy(dtype)
            fwd_atol = 1e-10 if dtype == wp.float64 else 1e-4
            bwd_atol = 1e-8 if dtype == wp.float64 else 1e-3

            def check(A_np, adj_LU):
                # the kernel factors A_np^T when transposed, so its gradient is the transpose
                B_np = A_np.T if transposed else A_np
                LU_np, P_np = lu_factor_numpy(B_np)
                LU_wp, grad_A = _lu_backward(A_np, adj_LU, device, wp_dtype=dtype, transposed=transposed)
                grad_B_ref = lu_adjoint_numpy(LU_np, P_np, adj_LU)
                grad_A_ref = grad_B_ref.T if transposed else grad_B_ref
                np.testing.assert_allclose(LU_wp, LU_np.astype(np_dtype), atol=fwd_atol)
                np.testing.assert_allclose(grad_A, grad_A_ref.astype(np_dtype), atol=bwd_atol)

            # Random general matrix (row interchanges at most steps)
            rng = np.random.default_rng(42)
            check(rng.standard_normal((TILE_M, TILE_M)), rng.standard_normal((TILE_M, TILE_M)))

            # Identity - closed-form adjoint (independent of the formula above):
            # Linearize A = LU at L = U = I: dA = dL + dU, so adj_A = adj_LU.
            rng = np.random.default_rng(100)
            adj_LU = rng.standard_normal((TILE_M, TILE_M))
            LU_wp, grad_A = _lu_backward(np.eye(TILE_M), adj_LU, device, wp_dtype=dtype, transposed=transposed)
            grad_A_ref = adj_LU.T if transposed else adj_LU
            np.testing.assert_allclose(LU_wp, np.eye(TILE_M, dtype=np_dtype), atol=fwd_atol)
            np.testing.assert_allclose(grad_A, grad_A_ref.astype(np_dtype), atol=bwd_atol)

            # Permuted diagonal - closed-form adjoint (independent of the formula above):
            # With B = Q diag(d) for a permutation Q, partial pivoting selects P = Q, so
            # L = I and U = diag(d). Linearize P^T B = LU there: P^T dB = dL diag(d) + dU,
            # so adj_B = P (tril(adj_LU, -1) / d[newaxis,:] + triu(adj_LU)).
            rng = np.random.default_rng(200)
            d = rng.uniform(1.0, 10.0, TILE_M) * rng.choice([-1.0, 1.0], TILE_M)
            Q = np.eye(TILE_M)[rng.permutation(TILE_M)]
            B_np = Q @ np.diag(d)
            adj_LU = rng.standard_normal((TILE_M, TILE_M))
            A_np = B_np.T if transposed else B_np
            LU_wp, grad_A = _lu_backward(A_np, adj_LU, device, wp_dtype=dtype, transposed=transposed)
            grad_B_ref = Q @ (np.tril(adj_LU, -1) / d[np.newaxis, :] + np.triu(adj_LU))
            grad_A_ref = grad_B_ref.T if transposed else grad_B_ref
            np.testing.assert_allclose(LU_wp, np.diag(d).astype(np_dtype), atol=fwd_atol)
            np.testing.assert_allclose(grad_A, grad_A_ref.astype(np_dtype), atol=bwd_atol)

        return test

    return test_tile_lu_backward


test_tile_lu_backward = _test_tile_lu_backward(transposed=False)
test_tile_lu_transposed_backward = _test_tile_lu_backward(transposed=True)


# tests a composition of the libmathdx LU calls with tile_matmul
def test_tile_lu_inverse(test, device):
    # enable_backward belongs in module_options, not as a kernel kwarg: the LTO
    # dispatch for tile_matmul/tile_lu reads the module builder's options, which
    # kernel kwargs are not merged into.
    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def lu_inverse_kernel(
        A: wp.array2d[float],
        E: wp.array2d[float],
        A_inv: wp.array2d[float],
        A_inv_T: wp.array2d[float],
        R: wp.array2d[float],
    ):
        """Compute the inverse of ``A`` and of ``A^T`` from one LU factorization and check ``A A^{-1}``."""
        a = wp.tile_load(A, shape=(TILE_M, TILE_M), storage="shared")
        e = wp.tile_load(E, shape=(TILE_M, TILE_M), storage="shared")

        lu, piv = wp.tile_lu(a)
        a_inv = wp.tile_lu_solve(lu, piv, e)
        a_inv_t = wp.tile_lu_solve(lu, piv, e, transpose=True)
        r = wp.tile_matmul(a, a_inv)

        wp.tile_store(A_inv, a_inv)
        wp.tile_store(A_inv_T, a_inv_t)
        wp.tile_store(R, r)

    # an orthogonal matrix is well conditioned and still needs row interchanges
    rng = np.random.default_rng(42)
    Q_np, _ = np.linalg.qr(rng.standard_normal((TILE_M, TILE_M)))
    Q_np = np.array(Q_np, dtype=float)

    A_wp = wp.array(Q_np, dtype=float, device=device)
    E_wp = wp.array(np.eye(TILE_M), dtype=float, device=device)
    A_inv_wp = wp.zeros_like(A_wp)
    A_inv_T_wp = wp.zeros_like(A_wp)
    R_wp = wp.zeros_like(A_wp)

    wp.launch_tiled(
        lu_inverse_kernel,
        dim=1,
        inputs=[A_wp, E_wp],
        outputs=[A_inv_wp, A_inv_T_wp, R_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(A_inv_wp.numpy(), Q_np.T, tol=1e-5)
    assert_np_equal(A_inv_T_wp.numpy(), Q_np, tol=1e-5)
    assert_np_equal(R_wp.numpy(), np.eye(TILE_M), tol=1e-5)


@wp.kernel(module="test_lu_fwd")
def tile_math_lu_transpose(
    gA: wp.array2d[wp.float64],
    gD: wp.array1d[wp.float64],
    gLU: wp.array2d[wp.float64],
    gP: wp.array1d[wp.int32],
    gy: wp.array1d[wp.float64],
    gx: wp.array1d[wp.float64],
):
    # Load A, D & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    d = wp.tile_load(gD, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    # Ensure tile_diag_add() works with transposed matrices
    a_t = wp.tile_transpose(a)
    # Compute LU, piv st P^T (A^T + diag(D)) = LU
    b = wp.tile_diag_add(a_t, d)
    lu, piv = wp.tile_lu(b)
    # Solve for x in (A^T + diag(D))^T x = y
    x = wp.tile_lu_solve(lu, piv, y, transpose=True)
    # Store LU, piv & x
    wp.tile_store(gLU, lu)
    wp.tile_store(gP, piv)
    wp.tile_store(gx, x)


def _test_tile_lu_transpose_out_of_place(test, device, kernel, multiple_rhs):
    """Shared test logic for tile_lu_solve(transpose=True) after tile_lu() with vector or matrix RHS."""
    rng = np.random.default_rng(42)
    A_h = rng.standard_normal((TILE_M, TILE_M))
    D_h = rng.standard_normal(TILE_M)

    A_np = A_h.T + np.diag(D_h)
    LU_np, P_np = lu_factor_numpy(A_np)

    if multiple_rhs:
        Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
        X_np = np.linalg.solve(A_np.T, Y_h.T)
        Z_np = X_np @ X_np

        A_wp = wp.array(A_h, dtype=wp.float64, device=device)
        D_wp = wp.array(D_h, dtype=wp.float64, device=device)
        LU_wp = wp.zeros((TILE_M, TILE_M), dtype=wp.float64, device=device)
        P_wp = wp.zeros(TILE_M, dtype=wp.int32, device=device)
        Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
        X_wp = wp.zeros_like(Y_wp)
        Z_wp = wp.zeros_like(Y_wp)

        wp.launch_tiled(
            kernel, dim=[1, 1], inputs=[A_wp, D_wp, LU_wp, P_wp, Y_wp, X_wp, Z_wp], block_dim=TILE_DIM, device=device
        )

        np.testing.assert_array_equal(P_wp.numpy(), P_np)
        np.testing.assert_allclose(LU_wp.numpy(), LU_np)
        np.testing.assert_allclose(X_wp.numpy(), X_np)
        np.testing.assert_allclose(Z_wp.numpy(), Z_np)
    else:
        Y_h = np.arange(TILE_M, dtype=np.float64)
        X_np = np.linalg.solve(A_np.T, Y_h)

        A_wp = wp.array(A_h, dtype=wp.float64, device=device)
        D_wp = wp.array(D_h, dtype=wp.float64, device=device)
        LU_wp = wp.zeros((TILE_M, TILE_M), dtype=wp.float64, device=device)
        P_wp = wp.zeros(TILE_M, dtype=wp.int32, device=device)
        Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
        X_wp = wp.zeros_like(Y_wp)

        wp.launch_tiled(
            kernel, dim=[1, 1], inputs=[A_wp, D_wp, LU_wp, P_wp, Y_wp, X_wp], block_dim=TILE_DIM, device=device
        )

        np.testing.assert_array_equal(P_wp.numpy(), P_np)
        np.testing.assert_allclose(LU_wp.numpy(), LU_np)
        np.testing.assert_allclose(X_wp.numpy(), X_np)


def test_tile_lu_transpose(test, device):
    _test_tile_lu_transpose_out_of_place(test, device, tile_math_lu_transpose, multiple_rhs=False)


@wp.kernel(module="test_lu_fwd")
def tile_math_lu_transpose_inplace(
    gA: wp.array2d[wp.float64],
    gP: wp.array1d[wp.int32],
    gy: wp.array1d[wp.float64],
):
    # Load A & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    # Compute LU, piv st P^T A = LU inplace
    piv = wp.tile_lu_inplace(a)
    # Solve for x in A^T x = y inplace
    wp.tile_lu_solve_inplace(a, piv, y, transpose=True)
    # Store LU, piv & y
    wp.tile_store(gA, a)
    wp.tile_store(gP, piv)
    wp.tile_store(gy, y)


def _test_tile_lu_transpose_inplace(test, device, kernel, multiple_rhs):
    """Shared test logic for tile_lu_solve_inplace(transpose=True) after tile_lu_inplace() with vector or matrix RHS."""
    rng = np.random.default_rng(42)
    A_h = rng.standard_normal((TILE_M, TILE_M))
    LU_np, P_np = lu_factor_numpy(A_h)

    if multiple_rhs:
        Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
        Y_sol_np = np.linalg.solve(A_h.T, Y_h.T)
        Z_np = Y_sol_np @ Y_sol_np

        A_wp = wp.array(A_h, dtype=wp.float64, device=device)
        P_wp = wp.zeros(TILE_M, dtype=wp.int32, device=device)
        Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
        Z_wp = wp.zeros_like(Y_wp)

        wp.launch_tiled(kernel, dim=[1, 1], inputs=[A_wp, P_wp, Y_wp, Z_wp], block_dim=TILE_DIM, device=device)

        np.testing.assert_array_equal(P_wp.numpy(), P_np)
        np.testing.assert_allclose(A_wp.numpy(), LU_np)
        np.testing.assert_allclose(Y_wp.numpy(), Y_sol_np)
        np.testing.assert_allclose(Z_wp.numpy(), Z_np)
    else:
        Y_h = np.arange(TILE_M, dtype=np.float64)
        Y_sol_np = np.linalg.solve(A_h.T, Y_h)

        A_wp = wp.array(A_h, dtype=wp.float64, device=device)
        P_wp = wp.zeros(TILE_M, dtype=wp.int32, device=device)
        Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)

        wp.launch_tiled(kernel, dim=[1, 1], inputs=[A_wp, P_wp, Y_wp], block_dim=TILE_DIM, device=device)

        np.testing.assert_array_equal(P_wp.numpy(), P_np)
        np.testing.assert_allclose(Y_wp.numpy(), Y_sol_np)
        np.testing.assert_allclose(A_wp.numpy(), LU_np)


def test_tile_lu_transpose_inplace(test, device):
    _test_tile_lu_transpose_inplace(test, device, tile_math_lu_transpose_inplace, multiple_rhs=False)


@wp.kernel(module="test_lu_fwd")
def tile_math_lu_transpose_multiple_rhs(
    gA: wp.array2d[wp.float64],
    gD: wp.array1d[wp.float64],
    gLU: wp.array2d[wp.float64],
    gP: wp.array1d[wp.int32],
    gy: wp.array2d[wp.float64],
    gx: wp.array2d[wp.float64],
    gz: wp.array2d[wp.float64],
):
    # Load A, D & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    d = wp.tile_load(gD, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=(TILE_M, TILE_M), storage="shared")
    # Compute LU, piv st P^T (A.T + diag(D)) = LU
    a_t = wp.tile_transpose(a)
    b = wp.tile_diag_add(a_t, d)
    lu, piv = wp.tile_lu(b)
    # Solve for x in (A.T + diag(D))^T x = y.T
    y_t = wp.tile_transpose(y)
    x = wp.tile_lu_solve(lu, piv, y_t, transpose=True)
    # Ensure matmul receives correct layout information
    z = wp.tile_matmul(x, x)
    # Store LU, piv, x & z
    wp.tile_store(gLU, lu)
    wp.tile_store(gP, piv)
    wp.tile_store(gx, x)
    wp.tile_store(gz, z)


def test_tile_lu_transpose_multiple_rhs(test, device):
    _test_tile_lu_transpose_out_of_place(test, device, tile_math_lu_transpose_multiple_rhs, multiple_rhs=True)


@wp.kernel(module="test_lu_fwd")
def tile_math_lu_transpose_multiple_rhs_inplace(
    gA: wp.array2d[wp.float64],
    gP: wp.array1d[wp.int32],
    gy: wp.array2d[wp.float64],
    gz: wp.array2d[wp.float64],
):
    # Load A & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=(TILE_M, TILE_M), storage="shared")
    # Compute LU, piv st P^T A = LU inplace
    piv = wp.tile_lu_inplace(a)
    # Solve for x in A^T x = y.T inplace
    y_t = wp.tile_transpose(y)
    wp.tile_lu_solve_inplace(a, piv, y_t, transpose=True)
    # Ensure matmul receives correct layout information
    z = wp.tile_matmul(y_t, y_t)
    # Store LU, piv, y & z
    wp.tile_store(gA, a)
    wp.tile_store(gP, piv)
    wp.tile_store(gy, y_t)
    wp.tile_store(gz, z)


def test_tile_lu_transpose_multiple_rhs_inplace(test, device):
    _test_tile_lu_transpose_inplace(test, device, tile_math_lu_transpose_multiple_rhs_inplace, multiple_rhs=True)


@wp.kernel(module="test_lu_fwd")
def tile_math_lu_solve_transpose(
    gLU: wp.array2d[wp.float64],
    gP: wp.array1d[wp.int32],
    gy: wp.array1d[wp.float64],
    gx: wp.array1d[wp.float64],
):
    LU = wp.tile_load(gLU, shape=(TILE_M, TILE_M), storage="shared")
    piv = wp.tile_load(gP, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    x = wp.tile_lu_solve(LU, piv, y, transpose=True)
    wp.tile_store(gx, x)


def test_tile_lu_solve_transpose(test, device):
    rng = np.random.default_rng(42)
    A_h = rng.standard_normal((TILE_M, TILE_M))
    LU_np, P_np = lu_factor_numpy(A_h)

    Y_h = np.arange(TILE_M, dtype=np.float64)
    X_np = np.linalg.solve(A_h.T, Y_h)

    LU_wp = wp.array(LU_np, dtype=wp.float64, device=device)
    P_wp = wp.array(P_np, dtype=wp.int32, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
    X_wp = wp.zeros(TILE_M, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_lu_solve_transpose, dim=[1, 1], inputs=[LU_wp, P_wp, Y_wp, X_wp], block_dim=TILE_DIM, device=device
    )

    np.testing.assert_allclose(X_wp.numpy(), X_np)


@wp.kernel(module="test_lu_fwd")
def tile_math_lu_solve_transpose_inplace(
    gLU: wp.array2d[wp.float64],
    gP: wp.array1d[wp.int32],
    gy: wp.array1d[wp.float64],
):
    LU = wp.tile_load(gLU, shape=(TILE_M, TILE_M), storage="shared")
    piv = wp.tile_load(gP, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    wp.tile_lu_solve_inplace(LU, piv, y, transpose=True)
    wp.tile_store(gy, y)


def test_tile_lu_solve_transpose_inplace(test, device):
    rng = np.random.default_rng(42)
    A_h = rng.standard_normal((TILE_M, TILE_M))
    LU_np, P_np = lu_factor_numpy(A_h)

    Y_h = np.arange(TILE_M, dtype=np.float64)
    X_np = np.linalg.solve(A_h.T, Y_h)

    LU_wp = wp.array(LU_np, dtype=wp.float64, device=device)
    P_wp = wp.array(P_np, dtype=wp.int32, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_lu_solve_transpose_inplace, dim=[1, 1], inputs=[LU_wp, P_wp, Y_wp], block_dim=TILE_DIM, device=device
    )

    np.testing.assert_allclose(Y_wp.numpy(), X_np)


@wp.kernel(module="test_lu_fwd")
def tile_math_lu_solve_transpose_multiple_rhs(
    gLU: wp.array2d[wp.float64],
    gP: wp.array1d[wp.int32],
    gY: wp.array2d[wp.float64],
    gX: wp.array2d[wp.float64],
):
    LU = wp.tile_load(gLU, shape=(TILE_M, TILE_M), storage="shared")
    piv = wp.tile_load(gP, shape=TILE_M, storage="shared")
    Y = wp.tile_load(gY, shape=(TILE_M, TILE_M), storage="shared")
    X = wp.tile_lu_solve(LU, piv, Y, transpose=True)
    wp.tile_store(gX, X)


def test_tile_lu_solve_transpose_multiple_rhs(test, device):
    rng = np.random.default_rng(42)
    A_h = rng.standard_normal((TILE_M, TILE_M))
    LU_np, P_np = lu_factor_numpy(A_h)

    Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
    X_np = np.linalg.solve(A_h.T, Y_h)

    LU_wp = wp.array(LU_np, dtype=wp.float64, device=device)
    P_wp = wp.array(P_np, dtype=wp.int32, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
    X_wp = wp.zeros((TILE_M, TILE_M), dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_lu_solve_transpose_multiple_rhs,
        dim=[1, 1],
        inputs=[LU_wp, P_wp, Y_wp, X_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    np.testing.assert_allclose(X_wp.numpy(), X_np)


@wp.kernel(module="test_lu_fwd")
def tile_math_lu_solve_transpose_multiple_rhs_inplace(
    gLU: wp.array2d[wp.float64],
    gP: wp.array1d[wp.int32],
    gY: wp.array2d[wp.float64],
):
    LU = wp.tile_load(gLU, shape=(TILE_M, TILE_M), storage="shared")
    piv = wp.tile_load(gP, shape=TILE_M, storage="shared")
    Y = wp.tile_load(gY, shape=(TILE_M, TILE_M), storage="shared")
    wp.tile_lu_solve_inplace(LU, piv, Y, transpose=True)
    wp.tile_store(gY, Y)


def test_tile_lu_solve_transpose_multiple_rhs_inplace(test, device):
    rng = np.random.default_rng(42)
    A_h = rng.standard_normal((TILE_M, TILE_M))
    LU_np, P_np = lu_factor_numpy(A_h)

    Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
    X_np = np.linalg.solve(A_h.T, Y_h)

    LU_wp = wp.array(LU_np, dtype=wp.float64, device=device)
    P_wp = wp.array(P_np, dtype=wp.int32, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_lu_solve_transpose_multiple_rhs_inplace,
        dim=[1, 1],
        inputs=[LU_wp, P_wp, Y_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    np.testing.assert_allclose(Y_wp.numpy(), X_np)


all_devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


@unittest.skipUnless(
    not wp._src.context.runtime.core.wp_is_mathdx_enabled()
    or (
        wp._src.context.runtime.core.wp_is_mathdx_enabled()
        and wp._src.context.runtime.core.wp_cuda_toolkit_version() >= 12060
    ),
    "MathDx is not enabled or is enabled but CUDA toolkit version is less than 12.6",
)
class TestTileLU(unittest.TestCase):
    pass


add_function_test(
    TestTileLU,
    "test_tile_lu_factor_and_solve",
    test_tile_lu_factor_and_solve,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileLU,
    "test_tile_lu_factor_and_solve_inplace",
    test_tile_lu_factor_and_solve_inplace,
    devices=all_devices,
    check_output=False,
)

add_function_test(
    TestTileLU,
    "test_tile_lu_factor_and_solve_multiple_rhs",
    test_tile_lu_factor_and_solve_multiple_rhs,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileLU,
    "test_tile_lu_factor_and_solve_multiple_rhs_inplace",
    test_tile_lu_factor_and_solve_multiple_rhs_inplace,
    devices=all_devices,
    check_output=False,
)


add_function_test(
    TestTileLU,
    "test_tile_lu_inverse",
    test_tile_lu_inverse,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileLU,
    "test_tile_lu_transpose",
    test_tile_lu_transpose,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileLU,
    "test_tile_lu_transpose_inplace",
    test_tile_lu_transpose_inplace,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileLU,
    "test_tile_lu_transpose_multiple_rhs",
    test_tile_lu_transpose_multiple_rhs,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileLU,
    "test_tile_lu_transpose_multiple_rhs_inplace",
    test_tile_lu_transpose_multiple_rhs_inplace,
    devices=all_devices,
    check_output=False,
)

add_function_test(TestTileLU, "test_tile_lu_solve_transpose", test_tile_lu_solve_transpose, devices=all_devices)
add_function_test(
    TestTileLU,
    "test_tile_lu_solve_transpose_inplace",
    test_tile_lu_solve_transpose_inplace,
    devices=all_devices,
)
add_function_test(
    TestTileLU,
    "test_tile_lu_solve_transpose_multiple_rhs",
    test_tile_lu_solve_transpose_multiple_rhs,
    devices=all_devices,
)
add_function_test(
    TestTileLU,
    "test_tile_lu_solve_transpose_multiple_rhs_inplace",
    test_tile_lu_solve_transpose_multiple_rhs_inplace,
    devices=all_devices,
)


add_function_test(
    TestTileLU,
    "test_tile_lu_backward_fp32",
    test_tile_lu_backward(wp.float32),
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileLU,
    "test_tile_lu_backward_fp64",
    test_tile_lu_backward(wp.float64),
    devices=all_devices,
    check_output=False,
)

add_function_test(
    TestTileLU,
    "test_tile_lu_transposed_backward_fp32",
    test_tile_lu_transposed_backward(wp.float32),
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileLU,
    "test_tile_lu_transposed_backward_fp64",
    test_tile_lu_transposed_backward(wp.float64),
    devices=all_devices,
    check_output=False,
)

cpu_block_tests = (
    ("test_tile_lu_factor_and_solve", test_tile_lu_factor_and_solve),
    ("test_tile_lu_factor_and_solve_inplace", test_tile_lu_factor_and_solve_inplace),
    ("test_tile_lu_factor_and_solve_multiple_rhs", test_tile_lu_factor_and_solve_multiple_rhs),
    ("test_tile_lu_transpose", test_tile_lu_transpose),
    ("test_tile_lu_transpose_inplace", test_tile_lu_transpose_inplace),
    ("test_tile_lu_solve_transpose", test_tile_lu_solve_transpose),
    ("test_tile_lu_solve_transpose_multiple_rhs", test_tile_lu_solve_transpose_multiple_rhs),
    ("test_tile_lu_backward_fp32", test_tile_lu_backward(wp.float32)),
    ("test_tile_lu_transposed_backward_fp32", test_tile_lu_transposed_backward(wp.float32)),
)
for name, func in cpu_block_tests:
    add_function_test(
        TestTileLU,
        f"{name}_cpu_blocks",
        func,
        devices=get_cpu_test_devices(),
        check_output=False,
        enable_cpu_blocks=True,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
