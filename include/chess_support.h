#include "thc.h"
#include <iostream>
#include <vector>
#include <torch/torch.h>

bool board_is_draw(thc::ChessRules board) {
  thc::DRAWTYPE draw_type;
  board.IsDraw(false, draw_type); // why does white asks matter for a draw...?
  return draw_type != thc::NOT_DRAW;
}

bool board_is_terminal(thc::ChessRules board) {
    // checks whether the game is done, i.e. if there is a checkmate or stalemate
    // "terminal" differs from the thc parlance but is consistent with our MDP language.
    thc::TERMINAL eval;
    board.Evaluate(eval);
    
    if ((eval != thc::NOT_TERMINAL) || board_is_draw(board)) {
        return true;
    }
    return false;
}

torch::Tensor board_to_tensor(thc::ChessRules board) {
  // returns a 8x8x13 tensor representing the board
  // the 13 channels are:
  // 0: white pawn
  // 1: white knight
  // 2: white bishop
  // 3: white rook
  // 4: white queen
  // 5: white king
  // 6: black pawn
  // 7: black knight
  // 8: black bishop
  // 9: black rook
  // 10: black queen
  // 11: black king
  // 12: blank
  torch::Tensor tensor = torch::zeros({8, 8, 13});
  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      char piece = board.squares[row*8 + col];
      if (piece == '.') {
        continue;
      }
      int channel = 0;
      if (piece == 'P') {
        channel = 0;
      } else if (piece == 'N') {
        channel = 1;
      } else if (piece == 'B') {
        channel = 2;
      } else if (piece == 'R') {
        channel = 3;
      } else if (piece == 'Q') {
        channel = 4;
      } else if (piece == 'K') {
        channel = 5;
      } else if (piece == 'p') {
        channel = 6;
      } else if (piece == 'n') {
        channel = 7;
      } else if (piece == 'b') {
        channel = 8;
      } else if (piece == 'r') {
        channel = 9;
      } else if (piece == 'q') {
        channel = 10;
      } else if (piece == 'k') {
        channel = 11;
      } else {
        channel = 12;
      }
      tensor[row][col][channel] = 1;
    }
  }
  return tensor;
}

std::vector<thc::Move> get_legal_moves(thc::ChessRules cr) {
  thc::MOVELIST movelist;
  cr.GenLegalMoveList(&movelist);
  std::vector<thc::Move> moves;
  for (size_t idx = 0; idx < movelist.count; idx++) {
    moves.push_back(movelist.moves[idx]);
  }
  return moves;
}

std::string move_to_str(thc::ChessRules cr, thc::Move move) { // FIXME: cr argument is useless now
  return move.TerseOut();
}

thc::Move str_to_move(thc::ChessRules cr, std::string str) {
  thc::Move move;
  if (!move.TerseIn(&cr, str.c_str())) {
    std::cout << "invalid move string: " << str << std::endl;
    std::cout << "not in legal moves: " << std::endl;
    for (auto mv : get_legal_moves(cr)) {
      std::cout << move_to_str(cr, mv) << std::endl;
    }
    throw std::runtime_error("[ERROR]: invalid move string");
  }
  return move;
}

void display_position( thc::ChessRules &cr, const std::string &description ) {
    std::string fen = cr.ForsythPublish();
    std::string s = cr.ToDebugStr();
    printf( "%s\n", description.c_str() );
    printf( "FEN (Forsyth Edwards Notation) = %s\n", fen.c_str() );
    printf( "Position = %s\n", s.c_str() );
}

void display_position(thc::ChessRules &cr) {
    display_position(cr, "Position");
}
