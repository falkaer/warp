Add CUDA cooperative launches through `@wp.kernel(cooperative=True)` and the `wp.grid_sync()` built-in, which
synchronizes all threads of all blocks of the grid, so that computations with dependent steps such as the levels
of a reduction or a scan run in a single kernel launch. `wp.get_cuda_max_cooperative_blocks()` returns the largest
grid a cooperative launch of a kernel accepts on a device.
