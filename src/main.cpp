#include <stdio.h>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

#include "board.h"

// Declaration for replay writer from replay_writer.cpp
void append_replay_line(const std::string &filename, int move_number, const std::string &player,
                        int chosen_col, const std::vector<int> &board);

// extern kernels in GPU file
extern "C" {
  __global__ void heuristic_player_kernel(const int *d_board, int rows, int cols, int player, int *d_out_col);
  __global__ void random_player_kernel(const int *d_board, int rows, int cols, unsigned long long seed, int *d_out_col);
}

static inline void checkCuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error - %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// Game utility: check for winner (1 or 2), return 0 if none
int check_winner(const std::vector<int>& b) {
  // horizontal, vertical, diag
  for (int r = 0; r < ROWS; ++r) {
    for (int c = 0; c < COLS; ++c) {
      int p = b[index_rc(r,c)];
      if (p == 0) continue;
      // horizontal
      if (c+3 < COLS) {
        bool ok = true;
        for (int k=1;k<4;k++) if (b[index_rc(r,c+k)] != p) ok=false;
        if (ok) return p;
      }
      // vertical
      if (r+3 < ROWS) {
        bool ok = true;
        for (int k=1;k<4;k++) if (b[index_rc(r+k,c)] != p) ok=false;
        if (ok) return p;
      }
      // diag down-right
      if (r+3 < ROWS && c+3 < COLS) {
        bool ok = true;
        for (int k=1;k<4;k++) if (b[index_rc(r+k,c+k)] != p) ok=false;
        if (ok) return p;
      }
      // diag down-left
      if (r+3 < ROWS && c-3 >= 0) {
        bool ok = true;
        for (int k=1;k<4;k++) if (b[index_rc(r+k,c-k)] != p) ok=false;
        if (ok) return p;
      }
    }
  }
  return 0;
}

int main() {
  int nDevices = 0;
  checkCuda(cudaGetDeviceCount(&nDevices), "cudaGetDeviceCount");
  printf("Number of GPU Devices: %d\n", nDevices);

  int currentChosenDeviceNumber = -1; // Will not choose a device by default

  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties");
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Device Compute Major: %d Minor: %d\n", prop.major, prop.minor);
    printf("  Max Thread Dimensions: [%d][%d][%d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Number of Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Device Clock Rate (KHz): %d\n", prop.clockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Registers Per Block: %d\n", prop.regsPerBlock);
    printf("  Registers Per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("  Shared Memory Per Block: %zu\n", prop.sharedMemPerBlock);
    printf("  Shared Memory Per Multiprocessor: %zu\n", prop.sharedMemPerMultiprocessor);
    printf("  Total Constant Memory (bytes): %zu\n", prop.totalConstMem);
    printf("  Total Global Memory (bytes): %zu\n", prop.totalGlobalMem);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }

  // Choose device(s) to use:
  // Player 1 device (p1_dev) and Player 2 device (p2_dev)
  int p1_dev = -1, p2_dev = -1;
  if (nDevices >= 2) { p1_dev = 0; p2_dev = 1; }
  else if (nDevices == 1) { p1_dev = 0; p2_dev = 0; }
  else { p1_dev = -1; p2_dev = -1; }

  currentChosenDeviceNumber = p1_dev;
  printf("The chosen GPU device has an index of: %d\n", currentChosenDeviceNumber);

  if (p1_dev == -1) {
    printf("No CUDA-capable devices found. Exiting.\n");
    return 0;
  }

  // prepare initial board
  std::vector<int> board(CELLS, 0);
  int move_number = 0;
  std::string replay_file = "replay/replay.txt";
  // clear old replay
  { FILE *f = fopen(replay_file.c_str(), "w"); if (f) fclose(f); }

  // device-side buffers per GPU: we'll allocate input board and output move
  int *d_board_p1 = nullptr;
  int *d_board_p2 = nullptr;
  int *d_move_p1 = nullptr;
  int *d_move_p2 = nullptr;

  // allocate on p1
  checkCuda(cudaSetDevice(p1_dev), "cudaSetDevice p1");
  checkCuda(cudaMalloc((void**)&d_board_p1, CELLS * sizeof(int)), "cudaMalloc p1 board");
  checkCuda(cudaMalloc((void**)&d_move_p1, sizeof(int)), "cudaMalloc p1 move");

  // allocate on p2
  checkCuda(cudaSetDevice(p2_dev), "cudaSetDevice p2");
  checkCuda(cudaMalloc((void**)&d_board_p2, CELLS * sizeof(int)), "cudaMalloc p2 board");
  checkCuda(cudaMalloc((void**)&d_move_p2, sizeof(int)), "cudaMalloc p2 move");

  // We'll alternate moves until winner or draw
  int current_player = 1; // 1 or 2
  int winner = 0;
  const int max_moves = CELLS;

  while (move_number < max_moves && winner == 0) {
    move_number++;
    int chosen_col = -1;

    if (current_player == 1) {
      // copy board to p1 device and run heuristic kernel
      checkCuda(cudaSetDevice(p1_dev), "cudaSetDevice p1 pre-launch");
      checkCuda(cudaMemcpy(d_board_p1, board.data(), CELLS * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy H->d p1");
      // launch heuristic kernel: single block single thread enough for our impl
      heuristic_player_kernel<<<1, 1>>>(d_board_p1, ROWS, COLS, 1, d_move_p1);
      checkCuda(cudaGetLastError(), "heuristic kernel launch p1");
      checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize p1");
      // copy back move
      checkCuda(cudaMemcpy(&chosen_col, d_move_p1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy d->H p1");
    } else {
      // Player 2 runs random kernel on p2
      checkCuda(cudaSetDevice(p2_dev), "cudaSetDevice p2 pre-launch");
      checkCuda(cudaMemcpy(d_board_p2, board.data(), CELLS * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy H->d p2");
      unsigned long long seed = (unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count() ^ (unsigned long long)move_number;
      random_player_kernel<<<1,1>>>(d_board_p2, ROWS, COLS, seed, d_move_p2);
      checkCuda(cudaGetLastError(), "random kernel launch p2");
      checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize p2");
      checkCuda(cudaMemcpy(&chosen_col, d_move_p2, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy d->H p2");
    }

    // validate chosen_col and apply to board
    if (chosen_col < 0 || chosen_col >= COLS) {
      // invalid from GPU; try to pick leftmost available
      for (int c=0;c<COLS;c++) if (find_row_for_col(board,c) != -1) { chosen_col = c; break; }
      if (chosen_col == -1) { // board full
        break;
      }
    }
    int row_for = find_row_for_col(board, chosen_col);
    if (row_for == -1) {
      // chosen column full; pick first available
      bool applied = false;
      for (int c=0;c<COLS && !applied;c++) {
        int rr = find_row_for_col(board, c);
        if (rr != -1) { board[index_rc(rr,c)] = current_player; applied = true; chosen_col = c; }
      }
      if (!applied) break;
    } else {
      board[index_rc(row_for, chosen_col)] = current_player;
    }

    // print current state and append to replay
    printf("Move %d: Player %d chose column %d\n", move_number, current_player, chosen_col);
    print_board_cli(board);
    append_replay_line(replay_file, move_number, (current_player==1? "PLAYER_1":"PLAYER_2"), chosen_col, board);

    // check winner
    winner = check_winner(board);
    if (winner != 0) break;

    // switch player
    current_player = (current_player == 1) ? 2 : 1;

    // small delay so output is readable in demo (optional)
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
  }

  if (winner != 0) {
    printf("Game Over - Winner: Player %d\n", winner);
  } else {
    printf("Game Over - Draw or Max Moves Reached\n");
  }

  // free device buffers
  checkCuda(cudaSetDevice(p1_dev), "cudaSetDevice p1 cleanup");
  cudaFree(d_board_p1); cudaFree(d_move_p1);
  checkCuda(cudaSetDevice(p2_dev), "cudaSetDevice p2 cleanup");
  cudaFree(d_board_p2); cudaFree(d_move_p2);

  return 0;
}
