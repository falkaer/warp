Add `wp.tile_lu()`, `wp.tile_lu_inplace()`, `wp.tile_lu_solve()`, and `wp.tile_lu_solve_inplace()` for LU
factorization with partial pivoting and the corresponding solves, backed by cuSolverDx `getrf`/`getrs` on GPU
with a cooperative scalar fallback. The factorization returns the packed factors and 1-based pivots as LAPACK
does, `wp.tile_lu()` supports backward propagation through the factors, and `transpose=True` solves with the
transposed matrix using the same factors.
