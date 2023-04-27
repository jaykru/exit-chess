#include <iostream>
#include <vector>
#include <optional>
#include <string>
// class of tictactoe boards
class TicTacToeBoard {
public:  
  char board[9] = {' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '};
  char player = 'X';
 
  bool operator==(const TicTacToeBoard& other) const {
    for (int i = 0; i < 9; i++) {
      if (board[i] != other.board[i]) {
        return false;
      }
    }
    return true;
  }

  // check if some player is near a win
  bool has_two_adj(char p) {
    // check rows
    for (int i = 0; i < 3; i++) {
      if (board[i * 3] == p && board[i * 3 + 1] == p && board[i * 3 + 2] == ' ') {
        return true;
      }
      if (board[i * 3] == p && board[i * 3 + 1] == ' ' && board[i * 3 + 2] == p) {
        return true;
      }
      if (board[i * 3] == ' ' && board[i * 3 + 1] == p && board[i * 3 + 2] == p) {
        return true;
      }
    }
    // check columns
    for (int i = 0; i < 3; i++) {
      if (board[i] == p && board[i + 3] == p && board[i + 6] == ' ') {
        return true;
      }
      if (board[i] == p && board[i + 3] == ' ' && board[i + 6] == p) {
        return true;
      }
      if (board[i] == ' ' && board[i + 3] == p && board[i + 6] == p) {
        return true;
      }
    }
    // check diagonals
    if (board[0] == p && board[4] == p && board[8] == ' ') {
      return true;
    }
    if (board[0] == p && board[4] == ' ' && board[8] == p) {
      return true;
    }
    if (board[0] == ' ' && board[4] == p && board[8] == p) {
      return true;
    }
    if (board[2] == p && board[4] == p && board[6] == ' ') {
      return true;
    }
    if (board[2] == p && board[4] == ' ' && board[6] == p) {
      return true;
    }
    if (board[2] == ' ' && board[4] == p && board[6] == p) {
      return true;
    }
    return false;
  }

  std::optional<char> winner() {
    // check rows
    for (int i = 0; i < 3; i++) {
      if (board[i * 3] == board[i * 3 + 1] && board[i * 3 + 1] == board[i * 3 + 2] && board[i * 3] != ' ') {
        return std::optional(board[i * 3]);
      }
    }
    // check columns
    for (int i = 0; i < 3; i++) {
      if (board[i] == board[i + 3] && board[i + 3] == board[i + 6] && board[i] != ' ') {
        return std::optional(board[i]);
      }
    }
    // check diagonals
    if (board[0] == board[4] && board[4] == board[8] && board[0] != ' ') {
      return std::optional(board[0]);
    }
    if (board[2] == board[4] && board[4] == board[6] && board[2] != ' ') {
      return std::optional(board[2]);
    }

    // check if board is full
    for (int i = 0; i < 9; i++) {
      if (board[i] == ' ') {
        return std::nullopt;
      }
    }

    // if board is full and no winner, return tie
    return std::optional('T');
  }

  // print board
  void print() {
    std::cout << " " << board[0] << " | " << board[1] << " | " << board[2] << " " << std::endl;
    std::cout << "---+---+---" << std::endl;
    std::cout << " " << board[3] << " | " << board[4] << " | " << board[5] << " " << std::endl;
    std::cout << "---+---+---" << std::endl;
    std::cout << " " << board[6] << " | " << board[7] << " | " << board[8] << " " << std::endl;
  }

  // check if move is valid
  bool valid(int move) {
    return board[move] == ' ';
  }

  std::vector<int> getMoves() {
    std::vector<int> moves;
    for (int i = 0; i < 9; i++) {
      if (valid(i)) {
        moves.push_back(i);
      }
    }
    return moves;
  }

  // make move
  TicTacToeBoard move(int move) {
    assert(valid(move));
    TicTacToeBoard newBoard = *this;
    newBoard.board[move] = this->player;
    newBoard.player = player == 'X' ? 'O' : 'X';
    return newBoard;
  }

  // get player
  char getPlayer() {
    return player;
  }

  // check if game is over
  bool over() {
    return winner().has_value();
  }
};