#pragma once

// Minimal .npy I/O for 2-D float32 matrices in C order, used by
// -postout and -postdir. Plain stdio only (MUSCLE's file helpers update
// global state), so these may be called from OpenMP threads.

// Writes Data (Rows*Cols floats, row-major) as format 1.0 with
// descr '<f4': the same bytes as numpy's np.save (numpy >= 1.24).
void WriteNpyFloat2D(const string &FileName, const float *Data,
  uint Rows, uint Cols);
