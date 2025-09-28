#ifndef BOARD_H
#define BOARD_H

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

static const int ROWS = 6;
static const int COLS = 7;
static const int CELLS = ROWS * COLS;

// Board stored row-major top-to-bottom, left-to-right
inline int index_rc(int r, int c) { return r * COLS + c; }

inline std::string serialize_board(const std::vector<int>& board) {
  std::ostringstream ss;
  for (int i = 0; i < CELLS; ++i) ss << board[i];
  return ss.str();
}

inline void print_board_cli(const std::vector<int>& board) {
  for (int r = 0; r < ROWS; ++r) {
    for (int c = 0; c < COLS; ++c) {
      int v = board[index_rc(r,c)];
      char ch = (v==0? '.' : (v==1? 'X' : 'O'));
      printf("%c ", ch);
    }
    printf("\n");
  }
  printf("0 1 2 3 4 5 6\n");
}

// Return first empty row index (from bottom) for the column, -1 if full.
// Rows are stored top-to-bottom so bottom row index is ROWS-1
inline int find_row_for_col(const std::vector<int>& board, int col) {
  for (int r = ROWS-1; r >= 0; --r) {
    if (board[index_rc(r,col)] == 0) return r;
  }
  return -1;
}

#endif // BOARD_H
