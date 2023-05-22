#include <iostream>
#include <filesystem>
#include <cmath>
#include <vector>
#include <optional>
#include <algorithm>
#include <string>
#include <random>
#include <ranges>
#include <cstdio>
#include <cassert>
#include <thread>
#include <chrono>
#include <mutex>
#include <ranges>
#include <torch/torch.h>
#include <torch/script.h>
#include "util.h"
#include "tictactoe.h"
#include "thc.h"
#include "chess_support.h"

std::random_device rd;
std::mt19937 g(rd());
auto durations = std::vector<double>();
std::mutex durations_m;

template <class S>
class Apprentice {
  public:
    std::function<double(S)> eval;
    std::function<void(S, double)> train;
    Apprentice(std::function<double(S)> eval, std::function<void(S, double)> train): eval(eval), train(train) {  };
};

template <typename S, typename A>
class MDP {
  public:
    std::function<S(S s, A a)> tr; // transition function, not really a MDP 
    std::function<std::vector<A>(S s)> actions; // actions at s
    std::function<std::optional<double>(S s)> reward; // reward at s
    std::function<bool(S s)> is_terminal; // is s terminal?

    MDP(std::function<S(S s, A a)> tr, std::function<std::optional<double>(S s)> reward, std::function<std::vector<A>(S s)> actions, std::function<bool(S s)> is_terminal)
    : tr(tr), reward(reward), actions(actions), is_terminal(is_terminal) {  };
};

template <typename S, typename A>
class MCTSNode {
public:
  MDP<S,A> mdp;
  S state;
  std::vector<MCTSNode<S,A>*> children;
  std::optional<MCTSNode<S,A>*> parent;
  std::optional<double> expected;
  double tot;
  int count;

  MCTSNode(MDP<S,A> mdp, S state, std::vector<MCTSNode<S,A>*> children, std::optional<MCTSNode<S,A>*> parent) 
  : mdp(mdp), state(state), children(children), parent(parent), expected(std::nullopt), tot(0), count(0) {  };

  MCTSNode(const MCTSNode<S,A> &other, std::optional<MCTSNode<S,A>*> parent): 
    mdp(other.mdp), state(other.state), children(std::vector<MCTSNode<S,A>*>()), parent(parent), expected(other.expected), tot(other.tot), count(other.count) 
    {
      for (auto child : other.children) {
        this->children.push_back(new MCTSNode<S,A>(*child, this));
      }
    }

  MCTSNode(MCTSNode<S,A>* parent, S state)
  : mdp(parent->mdp),
    state(state),
    parent(parent),
    tot(0),
    count(0)
  { };

  ~MCTSNode() {
    for (auto child : children) {
      delete child;
    }
  }

  void merge(MCTSNode<S,A> *other) {
    // TODO: fill this in
    // if (this->is_root() && other->is_root() && this->state != other->state) {
    //   throw std::runtime_error("Can't merge two roots with different states");
    // }
    if (this->is_root() && !other->is_root() || !this->is_root() && other->is_root()) {
      throw std::runtime_error("Can't merge a root with a non-root");
    }
    if (this->state != other->state) {
      throw std::runtime_error("Can't merge two nodes with different states");
    }
    this->tot += other->tot;
    this->count += other->count;

    for (auto their_child : other->children) {
      auto our_child = std::find_if(this->children.begin(),
                                    this->children.end(),
                                    [their_child](MCTSNode<S,A>* our_child) {
                                      return our_child->state == their_child->state;
                                    });
      if (our_child != this->children.end()) {
        (*our_child)->merge(their_child);
      } else {
        this->children.push_back(new MCTSNode<S,A>(*their_child,this));
      }
    }
  }

  MCTSNode* play(std::vector<A> actions) {  
    MCTSNode* cur = this;
    for (auto action : actions) {
      auto child = std::find_if(cur->children.begin(), cur->children.end(),
                                [&cur, &action](MCTSNode<S,A>* child)
                                  { return cur->mdp.tr(cur->state, action) == child->state; });
      if (child != children.end() && !cur->children.empty()) {
        cur = *child;
      } else {
        auto next_state = cur->mdp.tr(cur->state, action);
        auto next_node = new MCTSNode(cur, next_state);
        cur->children.push_back(next_node);
        cur = next_node;
      }
    }
    return cur;    
  }

  void debug() {
    // action is the action from parent actions that got us from parent state to this state
    int action;
    if (!parent.has_value()) {
      action = -1;
    } else {
      auto actions = mdp.actions(parent.value()->state);
      action = *std::find_if(actions.begin(), actions.end(), [this](A a) { return mdp.tr(parent.value()->state, a) == this->state; });
    }
    printf("[node info] player: %c; E = %f; A = %d; R = %f; tot = %f; count = %d\n", this->state.player, this->expected.value_or(0.0), action, *mdp.reward(this), this->tot, this->count);
  }

  inline bool is_root() {
    return !parent.has_value();
  }

  inline bool is_leaf() {
    return children.size() == 0;
  }

  inline void backprop() {
    MCTSNode* cur = this;  
    auto mreward = mdp.reward(cur->state);
    if (!mreward.has_value()) {
      throw std::runtime_error("[ERROR]: no reward at terminal state; check your MDP.");
    }
    auto reward = mreward.value();

    // FIXME: generalize this with mdp.stride or something
    int parity = -1;
    while (cur->parent.has_value()) {
      cur->tot += parity*reward;
      cur->count += 1;
      cur->expected = cur->tot / cur->count;
      cur = cur->parent.value();
      parity *= -1;
    }
    // annotate root; this is required for UCT to compute the correct score for the root's direct children
    cur->tot += parity*reward;
    cur->count += 1;
    cur->expected = cur->tot / cur->count;
  }
  
  inline double score(int cur_itersm1, double exploration_bias, Apprentice<S> apprentice) {
      auto bonus_weight = 0.5;
      auto exploration_term = exploration_bias * sqrt((double)log((double)this->parent.value()->count + (double)1.0) / ((double)this->count + (double)1.0));
      auto exp = this->expected.value_or(0.0);
      return exp + bonus_weight * apprentice.eval(this->state) + exploration_bias * exploration_term;
  }

  inline std::optional<MCTSNode<S,A>*> select(int cur_itersm1, double exploration_bias, Apprentice<S> apprentice) {
    if (children.size() == 0) {
      return std::nullopt;
    }

    // if all of the children have a null expected value, then select one at random
    if (std::all_of(children.begin(), children.end(), [](MCTSNode<S,A>* child) { return child->expected == std::nullopt; })) {
      return select_randomly(g, children);
    }

    auto nonapprentice = Apprentice<S>([](S s) { return 0.0; }, [](S s, double d) { });

    auto choice = *argmax(children.begin(), children.end(), [&](MCTSNode<S,A>* child) { return child->score(cur_itersm1, exploration_bias, apprentice); });
    auto nonchoice = *argmax(children.begin(), children.end(), [&](MCTSNode<S,A>* child) { return child->score(cur_itersm1, exploration_bias, nonapprentice); });
    // FIXME: right bias if all scores are equal
    return choice;
  }

  // Expands `node` and returns a randomly selected child node.
  inline MCTSNode<S,A> *expand() {
    if (this->children.size() == 0) {
      auto actions = mdp.actions(state);
      if (actions.size() == 0) {
          throw std::runtime_error("[ERROR]: no actions available for expansion");
      }

      auto new_children = std::vector<MCTSNode<S,A>*>();
      for (auto action : actions) {
        auto child = new MCTSNode(this, this->mdp.tr(this->state, action));
        new_children.push_back(child);
      }
      this->children.clear();
      this->children = new_children;
    }
    
    auto choice = select_randomly(g, this->children);
    return choice;
  }

  // search for iters iterations, starting from start
  // exploration_bias is the exploration term in the UCB1 formula
  // apprentice is what it sounds like. FIXME: better comment here.
  A search(int iters, float exploration_bias, Apprentice<S> apprentice) {
    if (mdp.actions(this->state).size() == 0) {
      throw std::runtime_error("[ERROR]: search called on state we can't act in");
    }
    for (auto cur_itersm1 = 0; cur_itersm1 < iters; cur_itersm1++) {
      MCTSNode<S,A>* cur = this;

      // SELECTION      
      while (!cur->is_leaf()) {
        cur = cur->select(cur_itersm1, exploration_bias, apprentice).value(); // FIXME?: unsafe? what if select returns a nullopt?
      }

      auto start = std::chrono::high_resolution_clock::now();

      // EXPANSION
      if (!mdp.is_terminal(cur->state)) {
        auto expanded_child = cur->expand();
        cur = expanded_child;
      }

      // ROLLOUT
      std::vector<std::unique_ptr<MCTSNode<S,A>>> rollout_nodes;
      while (!mdp.is_terminal(cur->state)) {
        auto actions = mdp.actions(cur->state);
        if (actions.size() == 0) {
           throw std::runtime_error("[ERROR]: no actions available at non-terminal state");
        }
        std::unique_ptr<MCTSNode<S,A>> choice = std::make_unique<MCTSNode<S,A>>(cur, mdp.tr(cur->state, select_randomly(g, actions)));
        cur = choice.get();
        rollout_nodes.push_back(std::move(choice));
      }

      // BACKPROPAGATION
      cur->backprop();

      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      durations_m.lock();
      durations.push_back(duration.count());
      durations_m.unlock();
    }

    // return the action resulting in the child with the highest expected value
    auto actions = mdp.actions(this->state);
    auto ret = argmax(actions.begin(), actions.end(), [&,this](auto action) {
      auto child = std::find_if(this->children.begin(), this->children.end(),[&,this](auto child){ return child->state == this->mdp.tr(this->state, action); });
      if (child == this->children.end()) {
        std::cout << "[ERROR]: no child found for action: " << action << std::endl;
        std::cout << "state is_terminal: " << mdp.is_terminal(this->state) << std::endl;
        // std::cout << "children " << "(" << this->children.size() << "): " << this->children << std::endl;
        // std::cout << "actions " << "(" << actions.size() << "): " << actions << std::endl;
        throw std::runtime_error("[ERROR]: no child found for action");
      }
      return (*child)->expected.value_or(-std::numeric_limits<double>::infinity()); // we never pick an unexplored child
    });

    if (ret == actions.end()) {
      throw std::runtime_error("[ERROR]: no actions available at non-terminal state");
    }

    auto child = *std::find_if(this->children.begin(), this->children.end(), [&,this](auto child){ return child->state == this->mdp.tr(this->state, *ret); });    
    return *ret;
  };

  // root-parallel search
  A par_search(int iters, float exploration_bias, Apprentice<S> apprentice) {
    assert (!this->mdp.is_terminal(this->state));
    auto num_threads = std::thread::hardware_concurrency();
    auto num_iters_per_thread = iters / num_threads;
    auto num_iters_last_thread = iters - (num_threads - 1) * num_iters_per_thread;

    auto threads = std::vector<std::thread>();
    std::mutex trees_m;
    auto trees = std::vector<MCTSNode<S,A>*>();
    for (auto i = 0; i < num_threads; i++) {
      auto num_iters = i == num_threads - 1 ? num_iters_last_thread : num_iters_per_thread;
      threads.push_back(std::thread([=, &trees_m, &trees,this]() {
        MCTSNode<S,A> *copy = new MCTSNode<S,A>(*this, this->parent);
        copy->search(num_iters, exploration_bias, apprentice);
        trees_m.lock();
        trees.push_back(copy);
        trees_m.unlock();
      }));
    }

    for (auto& thread : threads) {
      thread.join();
    }

    MCTSNode<S,A> *tree = trees[0];
    for (auto i = 1; i < trees.size(); i++) {
      tree->merge(trees[i]);
    }

    for (auto child : this->children) {
      delete child;
    }
    *this = *new MCTSNode<S,A>(*tree, tree->parent);

    for (auto tree : trees) {
      delete tree;
    }

    // return the action resulting in the child with the highest expected value
    auto actions = mdp.actions(this->state);
    auto ret = *argmax(actions.begin(), actions.end(), [&,this](auto action) {
      auto child = std::find_if(this->children.begin(), this->children.end(), [&,this](auto child){ return child->state == this->mdp.tr(this->state, action); });
      if (child == this->children.end()) {
        std::cout << "[ERROR]: no child found for action" << std::endl;
        std::cout << "state is_terminal: " << mdp.is_terminal(this->state) << std::endl;
        throw std::runtime_error("[ERROR]: no child found for action");
      }
      return (*child)->expected.value_or(-std::numeric_limits<double>::infinity()); // we never pick an unexplored child
    });

    auto child = *std::find_if(this->children.begin(), this->children.end(), [&,this](auto child){ return child->state == this->mdp.tr(this->state, ret); });
    return ret;
  };
};

int uci_chess() {
  // make a transition function pointer that takes a position and a move and returns a new position
  // this is a lambda function that takes a position and a move and returns a new position
  thc::ChessRules (*tr)(thc::ChessRules s, std::string a) = [](thc::ChessRules cr, std::string mv) {
    auto new_board = thc::ChessRules(cr);
    new_board.PlayMove(str_to_move(cr, mv));
    return new_board;
  };

  std::vector<std::string> (*actions)(thc::ChessRules s) = [](thc::ChessRules cr) {
    std::vector<std::string> moves = std::vector<std::string>();
    for (auto mv : get_legal_moves(cr)) {
      moves.push_back(move_to_str(cr, mv));
    }
    return moves;
  };

  std::optional<double> (*reward)(thc::ChessRules s) = [](thc::ChessRules cr) {
    thc::TERMINAL eval;
    cr.Evaluate(eval);
    if (eval == thc::TERMINAL_WCHECKMATE) { // White is checkmated
      if (cr.white) {
        return std::optional(-1.0);
      } else {
        return std::optional(1.0);
      }
    } else if (eval == thc::TERMINAL_BCHECKMATE) { // Black is checkmated
      if (!cr.white) {
        return std::optional(-1.0);
      } else {
        return std::optional(1.0);
      }
    } else {
      return std::optional(0.0);
    }
  };

  auto mdp = MDP<thc::ChessRules, std::string>(tr, reward, actions, board_is_terminal);
  int stalemates = 0;
  int wins = 0;
  int losses = 0;
  // if apprentice.pt exists, load it into `model` using torch::load
  torch::jit::script::Module model;
  if (std::filesystem::exists("apprentice.pt")) {
    try {
        model = torch::jit::load("apprentice.pt");
    } catch (const c10::Error &error) {
        std::cerr << error.what() << std::endl;  
        std::cerr << "Error loading the model" << std::endl;
        return -1;
    }
  } else {
    std::cerr << "[ERROR] Model not found." << std::endl;
    return -1;
  }

  model.to(torch::kCUDA);
  auto evalf = [&model](thc::ChessRules state) { return model.forward({board_to_tensor(state).to(torch::kCUDA).view({1,119,8,8})}).toTensor()[-1].item<double>(); };
  auto trainf = [&model](thc::ChessRules state, double reward) {
    auto loss = torch::nn::MSELoss();
    auto output = model.forward({board_to_tensor(state).to(torch::kCUDA).view({1,119,8,8})}).toTensor()[-1].view({1});
    auto target = torch::tensor({reward}).to(torch::kCUDA);
    auto l = loss(output, target);
    l.backward();
    std::vector<torch::Tensor> parameters;
    for (auto parameter : model.parameters()) {
      parameters.push_back(parameter);
    }
    torch::optim::SGD(parameters, 0.01).step();
    torch::optim::SGD(parameters, 0.01).zero_grad();
  };
  auto apprentice = new Apprentice<thc::ChessRules>(evalf, trainf);
  auto root = std::make_unique<MCTSNode<thc::ChessRules, std::string>>(mdp, thc::ChessRules(), std::vector<MCTSNode<thc::ChessRules, std::string>*>(), std::nullopt);
  
  auto cur_node = root.get();
  auto played = std::vector<std::string>();
    // create a new board (initial position
  thc::ChessRules board = thc::ChessRules(); 
  auto num_turns = 0;
  std::string best_move_str;

  // read `uci` command in from stdin and respond
  for (;;) {
    std::string cmd;
    std::getline(std::cin, cmd);
    auto toks = std::vector<std::string>();
    std::string cur = "";
    for (auto c : cmd) {
      if (c == ' ') {
        toks.push_back(cur);
        cur = "";
      } else {
        cur.push_back(c);
      }
    }
    toks.push_back(cur);
    if (toks.size() == 0) {
      continue;
    }
    if (toks[0] == "uci") {
      std::cout << "id name " << "jaybot9000" << std::endl;
      std::cout << "id author " << "jay" << std::endl;
      std::cout << "uciok" << std::endl;
    }
    if (toks[0] == "isready") {
      std::cout << "readyok" << std::endl;
    }
    if (toks[0] == "ucinewgame") {
      // create a new board (initial position)
      board = thc::ChessRules(); 
      num_turns = 0;
      played = std::vector<std::string>();
      root.reset(new MCTSNode<thc::ChessRules, std::string>(mdp, thc::ChessRules(), std::vector<MCTSNode<thc::ChessRules, std::string>*>(), std::nullopt));
      cur_node = root.get();
    }
    // if cmd matches the regular expression position (pos) (.*)
    if (toks[0] == "position") {
      std::string fen = toks[1];
      std::vector<std::string> moves;
      if (toks.size() >= 3 && toks[2] == "moves") {
        moves = std::vector<std::string>(toks.begin() + 3, toks.end());
      } else {
        moves = std::vector<std::string>();
      }

      if (fen == "startpos") {
        board = thc::ChessRules();
      } else {
        throw std::runtime_error("custom fen not supported"); // FIXME: add support for custom FEN
      }
      root.reset(new MCTSNode<thc::ChessRules, std::string>(mdp, board, std::vector<MCTSNode<thc::ChessRules, std::string>*>(), std::nullopt));
      std::vector<std::string> played = std::vector<std::string>();
      for (auto mv : moves) {
        board.PlayMove(str_to_move(board, mv));
        played.push_back(mv);
      }
      cur_node = root.get()->play(played);
    }
    // if cmd matches the regular expression go (.*)
    if (toks[0] == "go") {
      if (mdp.is_terminal(cur_node->state) && !mdp.actions(cur_node->state).empty()) {
        best_move_str = select_randomly(g, mdp.actions(cur_node->state)); // FIXME: this is a big bug,
      } else {
        best_move_str = cur_node->par_search(150000, 0.5, *apprentice);
      }
      std::cout << "bestmove " << best_move_str << std::endl;
    }

    if (toks[0] == "stop") {
      std::cout << "bestmove " << best_move_str << std::endl;
    }
    if (toks[0] == "quit") {
      // dump apprentice model to disk
      // if apprentice.pt already exists, delete it
      // if (std::filesystem::exists("apprentice.pt")) {
      //   std::filesystem::remove("apprentice.pt");
      // }
      model.save("apprentice.pt");
      delete apprentice;
      return 0;
    }
    if (toks[0] == "selfplay") {
      // perform toks[1] steps of self-play and use that to train the model
      int steps = std::stoi(toks[1]);
      std::cout << "Doing selfplay for " << steps << " steps" << std::endl;
      bool over = false;
      auto played = std::vector<std::string>();
      auto num_turns = 0;
      std::vector<thc::ChessRules> states;

      for (int num_turns = 0; num_turns < steps; num_turns += 1) {
        if (num_turns == 0 || over) {
          if (over) {
            // train step
            int parity = 1;
            double reward = *mdp.reward(*(states.end()-1));
            for (auto state = states.end(); state-- != states.begin();) {
              apprentice->train(*state, reward*parity);
              parity *= -1;
            }
          }

          std::cout << "Starting new game" << std::endl;
          states = std::vector<thc::ChessRules>();
          board = thc::ChessRules();
          root.reset(new MCTSNode<thc::ChessRules, std::string>(mdp, board, std::vector<MCTSNode<thc::ChessRules, std::string>*>(), std::nullopt));
          cur_node = root.get();
          played = std::vector<std::string>();
          display_position(board, "Initial position");
          over = false;
        }
        states.push_back(board);
        std::cout << "Step " << num_turns << std::endl;
        std::cout << "\tWins: " << wins << std::endl;
        std::cout << "\tLosses: " << losses << std::endl;
        std::cout << "\tStalemates/draws: " << stalemates << std::endl;
        if (num_turns > 0 && num_turns % 5 == 0) {
          std::cout << "\tClearing\n";
          root.reset(new MCTSNode<thc::ChessRules, std::string>(mdp, thc::ChessRules(), std::vector<MCTSNode<thc::ChessRules, std::string>*>(), std::nullopt));
        }

        // play a move
        cur_node = root.get()->play(played);
        cur_node->state = board;
        auto best_move_str = cur_node->par_search(800, 0.5, *apprentice);
        thc::Move best_move;
        best_move.TerseIn(&board, best_move_str.c_str());
        board.PushMove(best_move);
        played.push_back(move_to_str(board, best_move));

        std::cout << ((num_turns % 2 == 0) ? "White" : "Black") << " played: " << best_move.TerseOut() << std::endl;
        display_position(board, "");

        thc::TERMINAL eval;
        board.Evaluate(eval);
        if (eval == thc::TERMINAL_BCHECKMATE) { // AI player (black) is checkmated
          std::cout << "White won!" << std::endl;
          losses += 1;
          over = true;
        } else if (eval == thc::TERMINAL_WCHECKMATE) { // Human player (white) is checkmated
          std::cout << "Black won!" << std::endl;
          wins += 1;
          over = true;
        } else if (eval == thc::TERMINAL_WSTALEMATE || eval == thc::TERMINAL_BSTALEMATE || mdp.is_terminal(board) || (num_turns != 0 && num_turns % 50 == 0)) {
          std::cout << "Draw!" << std::endl;
          stalemates += 1;
          over = true;
        }
      }
      std::cout << "Done with selfplay" << std::endl;
    }
  }
}

int main() {
  return uci_chess();
}
