// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// tile_lu: cooperative scalar LU factorization with partial pivoting, LU
// solve helper, and adjoint, plus the tile_lu / tile_lu_inplace /
// tile_lu_solve / tile_lu_solve_inplace entry templates and the tile_lu
// adjoint dispatch.
//
// Conventions follow LAPACK getrf / getrs, which is also what cuSolverDx
// produces: the factorization overwrites A with L (strictly lower triangle,
// unit diagonal implied) and U (upper triangle including the diagonal), and
// the 1-based pivot vector ipiv records that row k was interchanged with row
// ipiv[k] - 1, applied in order k = 0..n-1, so that P^T A = L U.
//
// The cooperative scalar path is a correctness fallback for builds without
// libmathdx and for users who want to skip the slow LTO compilation cost
// during development. Users can route a kernel through the scalar path on a
// libmathdx-enabled build by setting the module option
// `enable_mathdx_solver=False` (or globally via
// `wp.config.enable_mathdx_solver = False`).

#pragma once

#include "tile.h"
#include "tile_solve.h"

#ifdef __clang__
// disable warnings related to C++17 extensions on CPU JIT builds
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif  // __clang__

namespace wp {

namespace partitioned_gemm {


// Scalar LU factorization with partial pivoting, cooperative across
// WP_TILE_BLOCK_DIM threads, operating in place on A.
//
// Cooperative structure (right-looking, unblocked, as LAPACK getf2):
//   - Outer j loop (column index): SEQUENTIAL -- column j depends on the
//     updates of columns 0..j-1. All threads execute it in lockstep.
//   - Pivot search over A[j:n, j]: ALL threads compute the arg-max
//     redundantly into local registers (first maximum wins, as idamax).
//     Only thread 0 writes Piv[j]. A sync follows so no thread swaps rows
//     while another is still reading column j.
//   - Row interchange: distributed over columns.
//   - Column scaling (multipliers below the diagonal): distributed over rows.
//     Skipped for an exactly zero pivot, as in getf2.
//   - Trailing rank-1 update: distributed over the (n-j-1)^2 entries.
//   - WP_TILE_SYNC() between each phase.
//
// On CPU, WP_TILE_BLOCK_DIM == 1 collapses the thread-strided loops to plain
// sequential and WP_TILE_SYNC() is a no-op.
template <typename TileA, typename TilePiv> inline CUDA_CALLABLE void scalar_lu_impl(TileA& A, TilePiv& Piv)
{
    using T = typename TileA::Type;
    constexpr int n = TileA::Layout::Shape::dim(1);

    for (int j = 0; j < n; ++j) {
        // Pivot search: redundant compute on all threads.
        int p = j;
        T best = wp::abs(A.data(tile_coord(j, j)));

        for (int i = j + 1; i < n; ++i) {
            T a = wp::abs(A.data(tile_coord(i, j)));
            if (a > best) {
                best = a;
                p = i;
            }
        }

        WP_TILE_SYNC();

        // Only thread 0 writes the (1-based) pivot.
        if (WP_TILE_THREAD_IDX == 0) {
            Piv.data(tile_coord(j)) = p + 1;
        }

        // Row interchange -- distributed across columns.
        if (p != j) {
            for (int k = WP_TILE_THREAD_IDX; k < n; k += WP_TILE_BLOCK_DIM) {
                T tmp = A.data(tile_coord(j, k));
                A.data(tile_coord(j, k)) = A.data(tile_coord(p, k));
                A.data(tile_coord(p, k)) = tmp;
            }
        }

        WP_TILE_SYNC();

        // Multipliers below the diagonal -- distributed across rows.
        T pivot = A.data(tile_coord(j, j));

        if (pivot != T(0.0f)) {
            T invPivot = T(1.0) / pivot;

            for (int i = j + 1 + WP_TILE_THREAD_IDX; i < n; i += WP_TILE_BLOCK_DIM) {
                A.data(tile_coord(i, j)) *= invPivot;
            }
        }

        WP_TILE_SYNC();

        // Trailing rank-1 update -- distributed across entries.
        const int m = n - j - 1;

        for (int idx = WP_TILE_THREAD_IDX; idx < m * m; idx += WP_TILE_BLOCK_DIM) {
            int i = j + 1 + idx / m;
            int k = j + 1 + idx % m;
            A.data(tile_coord(i, k)) -= A.data(tile_coord(i, j)) * A.data(tile_coord(j, k));
        }

        WP_TILE_SYNC();
    }
}


// Solve A X = B (Transpose=false) or A^T X = B (Transpose=true) in place on X,
// given the packed factors LU and pivots Piv of A.
//
// A = P L U, so
//   A X = B:   apply the interchanges to B in order k = 0..n-1, then solve
//              L Z = B (unit diagonal) and U X = Z.
//   A^T X = B: solve U^T Z = B and L^T W = Z (unit diagonal), then apply the
//              interchanges to W in reverse order k = n-1..0.
//
// Same cooperative split as scalar_cholesky_forward_substitution: vector RHS
// is fully gated on thread 0; matrix RHS distributes the outer column loop
// across threads.
template <bool Transpose, typename TileLU, typename TilePiv, typename TileX>
inline CUDA_CALLABLE void scalar_lu_solve(TileLU& LU, TilePiv& Piv, TileX& X)
{
    using T = typename TileLU::Type;
    constexpr int n = TileLU::Layout::Shape::dim(1);

    auto solve_column = [&](auto x) {
        if constexpr (!Transpose) {
            for (int k = 0; k < n; ++k) {
                int p = Piv.data(tile_coord(k)) - 1;
                if (p != k) {
                    T tmp = x(k);
                    x(k) = x(p);
                    x(p) = tmp;
                }
            }

            for (int i = 0; i < n; ++i) {
                T s = x(i);
                for (int j = 0; j < i; ++j)
                    s -= LU.data(tile_coord(i, j)) * x(j);
                x(i) = s;
            }

            for (int i = n - 1; i >= 0; --i) {
                T s = x(i);
                for (int j = i + 1; j < n; ++j)
                    s -= LU.data(tile_coord(i, j)) * x(j);
                T diag = LU.data(tile_coord(i, i));
                x(i) = (diag != T(0.0f)) ? s / diag : s;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                T s = x(i);
                for (int j = 0; j < i; ++j)
                    s -= LU.data(tile_coord(j, i)) * x(j);
                T diag = LU.data(tile_coord(i, i));
                x(i) = (diag != T(0.0f)) ? s / diag : s;
            }

            for (int i = n - 1; i >= 0; --i) {
                T s = x(i);
                for (int j = i + 1; j < n; ++j)
                    s -= LU.data(tile_coord(j, i)) * x(j);
                x(i) = s;
            }

            for (int k = n - 1; k >= 0; --k) {
                int p = Piv.data(tile_coord(k)) - 1;
                if (p != k) {
                    T tmp = x(k);
                    x(k) = x(p);
                    x(p) = tmp;
                }
            }
        }
    };

    if constexpr (TileX::Layout::Shape::N == 1) {
        if (WP_TILE_THREAD_IDX == 0) {
            solve_column([&](int i) -> T& { return X.data(tile_coord(i)); });
        }
        WP_TILE_SYNC();
    } else if constexpr (TileX::Layout::Shape::N == 2) {
        constexpr int m = TileX::Layout::Shape::dim(1);

        for (int k = WP_TILE_THREAD_IDX; k < m; k += WP_TILE_BLOCK_DIM) {
            solve_column([&](int i) -> T& { return X.data(tile_coord(i, k)); });
        }
        WP_TILE_SYNC();
    }
}


// Solve L^T X = W in place on the raw row-major n x n buffer W, with L the
// unit lower triangle of the packed factors Out. Distributes the columns of W
// across threads; sequential descending i within a column.
template <typename TileOut, typename T>
inline CUDA_CALLABLE void scalar_lu_adj_lower_transposed_solve(TileOut& Out, T* W)
{
    constexpr int n = TileOut::Layout::Shape::dim(1);

    for (int k = WP_TILE_THREAD_IDX; k < n; k += WP_TILE_BLOCK_DIM) {
        for (int i = n - 1; i >= 0; --i) {
            T s = W[i * n + k];
            for (int j = i + 1; j < n; ++j)
                s -= Out.data(tile_coord(j, i)) * W[j * n + k];
            W[i * n + k] = s;
        }
    }
}

// Solve U X = W in place on the raw row-major n x n buffer W, with U the
// upper triangle of the packed factors Out. Same split as above.
template <typename TileOut, typename T> inline CUDA_CALLABLE void scalar_lu_adj_upper_solve(TileOut& Out, T* W)
{
    constexpr int n = TileOut::Layout::Shape::dim(1);

    for (int k = WP_TILE_THREAD_IDX; k < n; k += WP_TILE_BLOCK_DIM) {
        for (int i = n - 1; i >= 0; --i) {
            T s = W[i * n + k];
            for (int j = i + 1; j < n; ++j)
                s -= Out.data(tile_coord(i, j)) * W[j * n + k];
            T diag = Out.data(tile_coord(i, i));
            W[i * n + k] = (diag != T(0.0f)) ? s / diag : s;
        }
    }
}


// Adjoint of the packed LU factorization with the pivots held fixed.
//
// With P^T A = L U, linearizing gives L^{-1} dA_p U^{-1} = L^{-1} dL + dU U^{-1},
// whose strictly lower part is L^{-1} dL and upper part is dU U^{-1}. With
// adj_L = strictly lower and adj_U = upper part of adj_Out, the adjoint is
//   G      = tril(L^T adj_L, -1) + triu(adj_U U^T)
//   adj_Ap = L^{-T} G U^{-T}
//   adj_A  = P adj_Ap   (undo the row interchanges)
//
// Six thread-strided phases over raw row-major scratch buffers W1, W2:
//   1. form G in W1 (the masked products only touch the stored triangles)
//   2. solve L^T X = W1 in place (unit diagonal)
//   3. transpose W1 into W2
//   4. solve U Y = W2 in place, so W2 = adj_Ap^T and row j of W2 is
//      column j of adj_Ap
//   5. undo the row interchanges of adj_Ap within each row of W2
//   6. accumulate W2^T into adj_A.grad
//
// The two triangular solves use the cuSolverDx TRSM LTOs when available
// (fun_bkwd_trsm_l reads the packed factor with flipped layout as L^T, upper
// fill mode and unit diagonal; fun_bkwd_trsm_u reads it as U, upper fill mode
// and non-unit diagonal), and thread-strided column substitutions otherwise.
template <typename BkwdTrsmL, typename BkwdTrsmU, typename TileOut, typename TilePiv, typename TileA>
inline CUDA_CALLABLE void adj_tile_lu_impl(
    BkwdTrsmL fun_bkwd_trsm_l, BkwdTrsmU fun_bkwd_trsm_u, TileOut& Out, TilePiv& Piv, TileA& adj_A, TileOut& adj_Out
)
{
    using T = typename TileA::Type;
    constexpr int n = TileA::Layout::Shape::dim(1);

#if defined(__CUDA_ARCH__)
    __shared__ T W1[n * n];
    __shared__ T W2[n * n];
#else
    T W1_local[WP_TILE_BLOCK_DIM == 1 ? n * n : 1];
    T W2_local[WP_TILE_BLOCK_DIM == 1 ? n * n : 1];
    T* W1;
    T* W2;
    if constexpr (WP_TILE_BLOCK_DIM == 1) {
        W1 = W1_local;
        W2 = W2_local;
    } else {
        W1 = (T*)tile_shared_storage_t::alloc(int(sizeof(T) * n * n));
        W2 = (T*)tile_shared_storage_t::alloc(int(sizeof(T) * n * n));
    }
#endif

    WP_TILE_SYNC();

    // Phase 1: G = tril(L^T adj_L, -1) + triu(adj_U U^T) into W1.
    //   i > j:  G[i,j] = adj_Out[i,j] + sum_{k>i} Out[k,i] * adj_Out[k,j]
    //   i <= j: G[i,j] = sum_{k>=j} adj_Out[i,k] * Out[j,k]
    for (int ij = WP_TILE_THREAD_IDX; ij < n * n; ij += WP_TILE_BLOCK_DIM) {
        int i = ij / n;
        int j = ij % n;
        T s = T(0);
        if (i > j) {
            s = adj_Out.grad(tile_coord(i, j));
            for (int k = i + 1; k < n; ++k)
                s += Out.data(tile_coord(k, i)) * adj_Out.grad(tile_coord(k, j));
        } else {
            for (int k = j; k < n; ++k)
                s += adj_Out.grad(tile_coord(i, k)) * Out.data(tile_coord(j, k));
        }
        W1[ij] = s;
    }
    WP_TILE_SYNC();

    // Phase 2: solve L^T X = W1 in place (unit diagonal).
#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    scalar_lu_adj_lower_transposed_solve(Out, W1);
#else
    if constexpr (wp_is_null_func<BkwdTrsmL>::value) {
        scalar_lu_adj_lower_transposed_solve(Out, W1);
    } else {
        fun_bkwd_trsm_l(Out.data.ptr, W1);
    }
#endif
    WP_TILE_SYNC();

    // Phase 3: transpose W1 into W2.
    for (int ij = WP_TILE_THREAD_IDX; ij < n * n; ij += WP_TILE_BLOCK_DIM) {
        int row = ij / n;
        int col = ij % n;
        W2[ij] = W1[col * n + row];
    }
    WP_TILE_SYNC();

    // Phase 4: solve U Y = W2 in place.
#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    scalar_lu_adj_upper_solve(Out, W2);
#else
    if constexpr (wp_is_null_func<BkwdTrsmU>::value) {
        scalar_lu_adj_upper_solve(Out, W2);
    } else {
        fun_bkwd_trsm_u(Out.data.ptr, W2);
    }
#endif
    WP_TILE_SYNC();

    // Phase 5: undo the row interchanges, adj_A = P adj_Ap. Row j of W2 holds
    // column j of adj_Ap, so each thread owns whole rows and applies the
    // interchanges in reverse order k = n-1..0 without further syncs.
    for (int j = WP_TILE_THREAD_IDX; j < n; j += WP_TILE_BLOCK_DIM) {
        for (int k = n - 1; k >= 0; --k) {
            int p = Piv.data(tile_coord(k)) - 1;
            if (p != k) {
                T tmp = W2[j * n + k];
                W2[j * n + k] = W2[j * n + p];
                W2[j * n + p] = tmp;
            }
        }
    }
    WP_TILE_SYNC();

    // Phase 6: accumulate adj_A.grad[i,j] += adj_A[i,j] = W2[j,i].
    for (int ij = WP_TILE_THREAD_IDX; ij < n * n; ij += WP_TILE_BLOCK_DIM) {
        int row = ij / n;
        int col = ij % n;
        adj_A.grad(tile_coord(row, col)) += W2[col * n + row];
    }
    WP_TILE_SYNC();

#if !defined(__CUDA_ARCH__)
    if constexpr (WP_TILE_BLOCK_DIM > 1) {
        tile_shared_storage_t::alloc(-int(sizeof(T) * n * n));
        tile_shared_storage_t::alloc(-int(sizeof(T) * n * n));
    }
#endif
}


}  // namespace partitioned_gemm


// LU factorization (in place) implementation.
template <typename Fwd, typename TileA, typename TilePiv>
CUDA_CALLABLE TilePiv& tile_lu_inplace_impl(Fwd fun_forward, TileA& A, TilePiv& Piv)
{
    static_assert(TileA::Layout::Shape::N == 2, "Expected TileA::Layout::Shape::N == 2");
    static_assert(TilePiv::Layout::Shape::N == 1, "Expected TilePiv::Layout::Shape::N == 1");
    static_assert(TileA::Layout::Shape::dim(0) == TileA::Layout::Shape::dim(1), "Expected TileA to be square");
    static_assert(
        TilePiv::Layout::Shape::dim(0) == TileA::Layout::Shape::dim(0),
        "Expected Piv to have as many entries as A has rows"
    );

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_lu_impl(A, Piv);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_lu_impl(A, Piv);
    } else {
        // TODO: for batched LU, need one info per batch
        __shared__ int info[1];

        if (WP_TILE_THREAD_IDX == 0) {
            info[0] = 0;
        }

        WP_TILE_SYNC();

        fun_forward(A.data.ptr, Piv.data.ptr, info);

        WP_TILE_SYNC();

        // TODO: for batched LU, check all batches
#if defined(_DEBUG)
        if (WP_TILE_THREAD_IDX == 0 && info[0] != 0) {
            printf("Non-zero status in LU factorization, got %d\n", info[0]);
        }
#endif
    }
#endif

    return Piv;
}

// LU factorization (out-of-place): writes the packed factors to Out and the
// pivots to Piv.
template <typename Fwd, typename BkwdTrsmL, typename BkwdTrsmU, typename TileA, typename TileOut, typename TilePiv>
CUDA_CALLABLE void
tile_lu(Fwd fun_forward, BkwdTrsmL fun_bkwd_trsm_l, BkwdTrsmU fun_bkwd_trsm_u, TileA& A, TileOut& Out, TilePiv& Piv)
{
    static_assert(TileOut::Layout::Shape::N == 2, "Expected TileOut::Layout::Shape::N == 2");
    static_assert(
        TileA::Layout::Shape::dim(0) == TileOut::Layout::Shape::dim(0),
        "Expected A and Out to have the same number of rows"
    );
    static_assert(
        TileA::Layout::Shape::dim(1) == TileOut::Layout::Shape::dim(1),
        "Expected A and Out to have the same number of columns"
    );

    Out = A;

    tile_lu_inplace_impl(fun_forward, Out, Piv);
}

// Adjoint of LU (out-of-place), differentiating through the factors with the
// pivots held fixed.
template <
    typename Fwd,
    typename BkwdTrsmL,
    typename BkwdTrsmU,
    typename TileA,
    typename TileOut,
    typename TilePiv,
    typename AdjFwd,
    typename AdjBkwdTrsmL,
    typename AdjBkwdTrsmU,
    typename AdjTileA,
    typename AdjTileOut,
    typename AdjTilePiv>
CUDA_CALLABLE void adj_tile_lu(
    Fwd fun_forward,
    BkwdTrsmL fun_bkwd_trsm_l,
    BkwdTrsmU fun_bkwd_trsm_u,
    TileA& A,
    TileOut& Out,
    TilePiv& Piv,
    AdjFwd adj_fun_forward,
    AdjBkwdTrsmL adj_fun_bkwd_trsm_l,
    AdjBkwdTrsmU adj_fun_bkwd_trsm_u,
    AdjTileA& adj_A,
    AdjTileOut& adj_Out,
    AdjTilePiv& adj_Piv
)
{
    partitioned_gemm::adj_tile_lu_impl(fun_bkwd_trsm_l, fun_bkwd_trsm_u, Out, Piv, adj_A, adj_Out);
}

// LU factorization (in place): overwrites A with the packed factors and returns the pivots.
template <typename Fwd, typename TileA, typename TilePiv>
CUDA_CALLABLE TilePiv& tile_lu_inplace(Fwd fun_forward, TileA& A, TilePiv& Piv)
{
    return tile_lu_inplace_impl(fun_forward, A, Piv);
}

template <
    typename Fwd,
    typename TileA,
    typename TilePiv,
    typename AdjFwd,
    typename AdjTileA,
    typename AdjTilePiv,
    typename AdjRet>
void adj_tile_lu_inplace(
    Fwd fun_forward,
    TileA& A,
    TilePiv& Piv,
    AdjFwd adj_fun_forward,
    AdjTileA& adj_A,
    AdjTilePiv& adj_Piv,
    AdjRet& adj_ret
)
{
    // MISSINGADJOINT: apply the LU adjoint in place; on entry A holds the packed
    // factors, adj_A holds their adjoint; on exit adj_A holds the adjoint of the original matrix
}


// LU solve (out-of-place): tile_lu_solve<false>(...) solves A x = y, tile_lu_solve<true>(...) solves A^T x = y
template <bool Transpose, typename Fwd, typename TileLU, typename TilePiv, typename TileY, typename TileX>
TileX& tile_lu_solve(Fwd fun_forward, TileLU& LU, TilePiv& Piv, TileY& Y, TileX& X)
{
    // Copy y to x
    X = Y;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_lu_solve<Transpose>(LU, Piv, X);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_lu_solve<Transpose>(LU, Piv, X);
    } else {
        WP_TILE_SYNC();
        fun_forward(LU.data.ptr, Piv.data.ptr, X.data.ptr);
        WP_TILE_SYNC();
    }
#endif

    return X;
}

template <bool Transpose, typename Fwd, typename TileLU, typename TilePiv, typename TileY>
void tile_lu_solve_inplace(Fwd fun_forward, TileLU& LU, TilePiv& Piv, TileY& Y)
{
#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_lu_solve<Transpose>(LU, Piv, Y);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_lu_solve<Transpose>(LU, Piv, Y);
    } else {
        WP_TILE_SYNC();
        fun_forward(LU.data.ptr, Piv.data.ptr, Y.data.ptr);
        WP_TILE_SYNC();
    }
#endif
}

template <
    bool Transpose,
    typename Fwd,
    typename TileLU,
    typename TilePiv,
    typename TileY,
    typename TileX,
    typename AdjFwd,
    typename AdjTileLU,
    typename AdjTilePiv,
    typename AdjTileY,
    typename AdjTileX,
    typename AdjRet>
void adj_tile_lu_solve(
    Fwd fun_forward,
    TileLU& LU,
    TilePiv& Piv,
    TileY& Y,
    TileX& X,
    AdjFwd adj_fun_forward,
    AdjTileLU& adj_LU,
    AdjTilePiv& adj_Piv,
    AdjTileY& adj_Y,
    AdjTileX& adj_X,
    AdjRet& adj_ret
)
{
    // MISSINGADJOINT: implicit differentiation through A X = Y (or A^T X = Y): solve the
    // transposed system for Z = adj_ret, then adj_Y += Z and back-propagate -Z X^T into
    // the packed factors
}

template <
    bool Transpose,
    typename Fwd,
    typename TileLU,
    typename TilePiv,
    typename TileY,
    typename AdjFwd,
    typename AdjTileLU,
    typename AdjTilePiv,
    typename AdjTileY>
void adj_tile_lu_solve_inplace(
    Fwd fun_forward,
    TileLU& LU,
    TilePiv& Piv,
    TileY& Y,
    AdjFwd adj_fun_forward,
    AdjTileLU& adj_LU,
    AdjTilePiv& adj_Piv,
    AdjTileY& adj_Y
)
{
    // MISSINGADJOINT: same math as adj_tile_lu_solve operating in place on adj_Y
}


}  // namespace wp

#ifdef __clang__
#pragma clang diagnostic pop
#endif
