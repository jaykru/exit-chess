#include "thc.h"
#include <iostream>
#include <vector>

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
