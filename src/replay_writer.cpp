#include <fstream>
#include <string>
#include <vector>
#include "board.h"

// Simple utility to append replay lines in the expected CSV-like format:
void append_replay_line(const std::string &filename, int move_number, const std::string &player,
                        int chosen_col, const std::vector<int> &board) {
  std::ofstream ofs;
  ofs.open(filename, std::ios::app);
  if (!ofs.is_open()) return;
  ofs << move_number << "," << player << "," << chosen_col << "," << serialize_board(board) << "\n";
  ofs.close();
}
