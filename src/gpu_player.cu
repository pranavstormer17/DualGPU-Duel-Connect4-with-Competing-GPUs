// GPU player kernels (simple deterministic heuristics & a random-ish kernel).
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

// Kernel: deterministic heuristic - select the leftmost non-full column
__global__ void heuristic_player_kernel(const int *d_board /* size 42 */, int rows, int cols, int player, int *d_out_col) {
  // Single-thread selection: leftmost non-full column with some basic score calc
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int best_col = -1;
    int best_score = -1000000;
    for (int c = 0; c < cols; ++c) {
      // find first free row
      int free_row = -1;
      for (int r = rows - 1; r >= 0; --r) {
        int idx = r * cols + c;
        if (d_board[idx] == 0) { free_row = r; break; }
      }
      if (free_row == -1) continue;
      // compute a simple score: how many same-player pieces in this column
      int score = 0;
      for (int r = 0; r < rows; ++r) {
        if (d_board[r*cols + c] == player) score++;
      }
      // prefer center columns slightly to make it a little smarter
      int center_bias = (cols/2) - abs(c - cols/2);
      score = score * 10 + center_bias;
      if (score > best_score) { best_score = score; best_col = c; }
    }
    if (best_col == -1) best_col = 0; // fallback
    d_out_col[0] = best_col;
  }
}

// Kernel: pseudorandom-ish player - uses simple xorshift seeded by input to pick start column
__global__ void random_player_kernel(const int *d_board /* size 42 */, int rows, int cols, unsigned long long seed, int *d_out_col) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // very small xorshift
    unsigned long long x = seed;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    int start = (int)(x % cols);
    int chosen = -1;
    for (int i = 0; i < cols; ++i) {
      int c = (start + i) % cols;
      // find first free row
      for (int r = rows - 1; r >= 0; --r) {
        int idx = r * cols + c;
        if (d_board[idx] == 0) { chosen = c; break; }
      }
      if (chosen != -1) break;
    }
    if (chosen == -1) chosen = 0;
    d_out_col[0] = chosen;
  }
}

} // extern "C"
