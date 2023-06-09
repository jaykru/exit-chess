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

template <class S, class A>
class Apprentice {
  public:
    std::function<torch::Tensor(S)> action_dist;
    std::function<double(S)> eval;
    std::function<void(std::vector<S>, std::vector<A>, double)> train;
    Apprentice(std::function<torch::Tensor(S)> action_dist, std::function<double(S)> eval, std::function<void(std::vector<S>, std::vector<A>, double)> train): action_dist(action_dist), eval(eval), train(train) {  };
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
class ExItNode {
public:
  MDP<S,A> mdp;
  Apprentice<S,A> apprentice;
  S state;
  std::vector<ExItNode<S,A>*> children;
  std::optional<ExItNode<S,A>*> parent;
  std::optional<double> expected;
  double tot;
  int count;

  ExItNode(MDP<S,A> mdp, Apprentice<S,A> apprentice, S state, std::vector<ExItNode<S,A>*> children, std::optional<ExItNode<S,A>*> parent)
    : mdp(mdp),
      state(state),
      children(children),
      parent(parent),
      expected(std::nullopt),
      tot(0),
      count(0),
      apprentice(apprentice)
    {
      // assert(!this->parent.has_value() || this->parent.value() != nullptr);
    };

    ExItNode(const ExItNode<S,A> &other, std::optional<ExItNode<S,A>*> parent):
      apprentice(other.apprentice),
      mdp(other.mdp),
      state(other.state),
      children(std::vector<ExItNode<S,A>*>()),
      parent(parent),
      expected(other.expected),
      tot(other.tot),
      count(other.count)
    {
      // assert(!this->parent.has_value() || this->parent.value() != nullptr);
      for (auto child : other.children) {
        this->children.push_back(new ExItNode<S,A>(*child, this));
      }
    }

  ExItNode(ExItNode<S,A>* parent, S state)
  : mdp(parent->mdp),
    apprentice(parent->apprentice),
    state(state),
    parent(parent),
    tot(0),
    count(0)
  { };

  ~ExItNode() {
    for (auto child : children) {
      delete child;
    }
  }

  void merge(ExItNode<S,A> *other) {
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
                                    [their_child](ExItNode<S,A>* our_child) {
                                      return our_child->state == their_child->state;
                                    });
      if (our_child != this->children.end()) {
        (*our_child)->merge(their_child);
      } else {
        this->children.push_back(new ExItNode<S,A>(*their_child,this));
      }
    }
  }

  ExItNode* play(std::vector<A> actions) {
    ExItNode* cur = this;
    for (auto action : actions) {
      auto child = std::find_if(cur->children.begin(), cur->children.end(),
                                [&cur, &action](ExItNode<S,A>* child)
                                  { return cur->mdp.tr(cur->state, action) == child->state; });
      if (child != children.end() && !cur->children.empty()) {
        cur = *child;
      } else {
        auto next_state = cur->mdp.tr(cur->state, action);
        auto next_node = new ExItNode(cur, next_state);
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
    ExItNode* cur = this;
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

  inline double score(int cur_itersm1, double exploration_bias, bool bootstrap) {
      auto bonus_weight = 0.5;
      auto exploration_term = exploration_bias * sqrt((double)log((double)this->parent.value()->count + (double)1.0) / ((double)this->count + (double)1.0));
      auto exp = this->expected.value_or(0.0);
      auto bonus_term = bootstrap ? 0.0 : bonus_weight * this->apprentice.eval(this->state);
      return exp + bonus_term + exploration_term;
  }

  inline std::optional<ExItNode<S,A>*> select(int cur_itersm1, double exploration_bias, bool bootstrap) {
    if (children.size() == 0) {
      return std::nullopt;
    }

    // if all of the children have a null expected value, then select one at random
    if (std::all_of(children.begin(), children.end(), [](ExItNode<S,A>* child) { return child->expected == std::nullopt; })) {
      return select_randomly(g, children);
    }

    // otherwise pick the child with the best UCT score
    auto best = argmax(children.begin(), children.end(),
                       [&](ExItNode<S,A>* child) {
                         return child->score(cur_itersm1, exploration_bias, bootstrap);
                       });
    return *best;
  }

  // Expands `node` and returns a randomly selected child node.
  inline ExItNode<S,A> *expand() {
    if (this->children.size() == 0) {
      auto actions = mdp.actions(state);
      if (actions.size() == 0) {
          throw std::runtime_error("[ERROR]: no actions available for expansion");
      }

      auto new_children = std::vector<ExItNode<S,A>*>();
      for (auto action : actions) {
        auto child = new ExItNode(this, this->mdp.tr(this->state, action));
        new_children.push_back(child);
      }
      this->children.clear();
      this->children = new_children;
    }
    
    auto choice = select_randomly(g, this->children);
    return choice;
  }

    inline std::vector<ExItNode<S, A>*> dm_rollout() {
      std::vector<ExItNode*> rollout_nodes;
      ExItNode<S,A>* cur = this;
      while (!mdp.is_terminal(cur->state)) {
        auto legal_moves = mdp.actions(cur->state);
        if (legal_moves.size() < 1) {
          throw std::runtime_error("[ERROR]: no actions available at non-terminal state");
        }
        int tries = 0;
        for (;;) {
          // assert(legal_moves.size() > 0);
          if (tries > 0) {
            // too many tries to sample a legal move from the net, we're just
            // going to do a random rollout here.
            cur = new ExItNode<S,A>(cur->mdp, cur->apprentice, mdp.tr(cur->state, select_randomly(g, legal_moves)), std::vector<ExItNode<S,A>*>(), cur);
            rollout_nodes.push_back(cur);
            break;
          }
          tries += 1;
          // get the distribution from the apprentice
          auto dist = cur->apprentice.action_dist(this->state);
          // sample from the distribution tensor with libtorch
          at::Tensor tmp = torch::multinomial(dist, 1, true)[0];
          auto sample = tmp.item<int>();

          // convert the index into a move

          // FIXME: this is hardcoded. it would be nicer if we could actually just roll
          // this into the apprentice somehow, with a sample method that returns an
          // action.
          int src = sample / 64;
          int tgt = sample % 64;

          // create a string representation of the move
          std::string move = "";
          move += (char)(src % 8 + 'a');
          move += (char)(src / 8 + '1');
          move += (char)(tgt % 8 + 'a');
          move += (char)(tgt / 8 + '1');

          // re-sample if move isn't legal
          if (std::find(legal_moves.begin(), legal_moves.end(), move) == legal_moves.end()) {
            continue;
          }

          // make the child that would result from playing `move`
          cur = new ExItNode<S,A>(cur->mdp, cur->apprentice, mdp.tr(cur->state, move), std::vector<ExItNode<S,A>*>(), cur);
          rollout_nodes.push_back(cur);
          break;
        }
      }
      return rollout_nodes;
    }

    inline std::vector<ExItNode<S, A>*> basic_rollout() {
      // ROLLOUT
      std::vector<ExItNode<S, A>*> rollout_nodes; // nodes to be freed by caller
      ExItNode* cur = this;
      while (!mdp.is_terminal(cur->state)) {
        auto actions = mdp.actions(cur->state);
        if (actions.size() == 0) {
           throw std::runtime_error("[ERROR]: no actions available at non-terminal state");
        }
        auto action = select_randomly(g, actions);
        cur = new ExItNode<S,A>(cur, mdp.tr(cur->state, action));
        rollout_nodes.push_back(cur);
      }
      return rollout_nodes;
    }

  // search for iters iterations, starting from start
  // exploration_bias is the exploration term in the UCB1 formula
  // apprentice is what it sounds like. FIXME: better comment here.
  A search(int iters, float exploration_bias, bool bootstrap) {
    if (mdp.actions(this->state).size() == 0) {
      throw std::runtime_error("[ERROR]: search called on state we can't act in");
    }
    for (auto cur_itersm1 = 0; cur_itersm1 < iters; cur_itersm1++) {
      ExItNode<S,A>* cur = this;

      // SELECTION
      // std::cout << "selecting..." << std::endl;
      while (!cur->is_leaf()) {
        cur = cur->select(cur_itersm1, exploration_bias, bootstrap).value(); // FIXME?: unsafe? what if select returns a nullopt?
      }

      // std::cout << "expanding..." << std::endl;
      // EXPANSION
      if (!mdp.is_terminal(cur->state)) {
        auto expanded_child = cur->expand();
        cur = expanded_child;
      }

      // ROLLOUT
      std::vector<ExItNode*> rollout_nodes;
      if (!mdp.is_terminal(cur->state)) {
        rollout_nodes = bootstrap ? cur->basic_rollout() : cur->dm_rollout();
        cur = rollout_nodes.back();
      }

      // std::cout << "backpropagating..." << std::endl;
      // BACKPROPAGATION
      cur->backprop();
      // free the rollout nodes
      for (auto node : rollout_nodes) {
        delete node;
      }
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
  A par_search(int iters, float exploration_bias, bool bootstrap) {
    // assert (!this->mdp.is_terminal(this->state));
    auto num_threads = std::thread::hardware_concurrency();
    auto num_iters_per_thread = iters / num_threads;
    auto num_iters_last_thread = iters - (num_threads - 1) * num_iters_per_thread;

    auto threads = std::vector<std::thread>();
    std::mutex trees_m;
    auto trees = std::vector<ExItNode<S,A>*>();
    for (auto i = 0; i < num_threads; i++) {
      auto num_iters = i == num_threads - 1 ? num_iters_last_thread : num_iters_per_thread;
      threads.push_back(std::thread([=, &trees_m, &trees,this]() {
        ExItNode<S,A> *copy = new ExItNode<S,A>(*this, this->parent);
        copy->search(num_iters, exploration_bias, bootstrap);
        trees_m.lock();
        trees.push_back(copy);
        trees_m.unlock();
      }));
    }

    for (auto& thread : threads) {
      thread.join();
    }

    ExItNode<S,A> *tree = trees[0];
    for (auto i = 1; i < trees.size(); i++) {
      tree->merge(trees[i]);
    }

    for (auto child : this->children) {
      delete child;
    }
    *this = *new ExItNode<S,A>(*tree, tree->parent);

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
    // assert(get_legal_moves(new_board) == get_legal_moves(cr));
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

  std::cout << "cuda is available: " << (torch::cuda::is_available() ? "yes" : "no") << std::endl;
  model.to(torch::kCUDA);
  auto evalf = [&model](thc::ChessRules state) { return model.forward({board_to_tensor(state).to(torch::kCUDA).view({1,119,8,8})}).toTensor()[-1].item<double>(); };
  auto trainf = [&model](std::vector<thc::ChessRules> states, std::vector<std::string> actions, double reward) {
    // trains on the results from a single step of self-play
    int parity = 1;
    for (int i = 0; i < states.size()-1; i++) {
      auto loss = torch::nn::MSELoss();
      torch::Tensor output = model.forward({board_to_tensor(states[i]).to(torch::kCUDA).view({1,119,8,8})}).toTensor().to(torch::kCUDA);

      // convert the action to a tensor
      torch::Tensor action_tensor = torch::zeros({4096}).to(torch::kCUDA);

      // action[i] is a string representing the chess move taken at states[i],
      // we need to convert it to a pair of indices ((i1,j1),(i2,j2)) indicating
      // the source and target of the move.
      auto src_str = actions[i].substr(0,2);
      auto dst_str = actions[i].substr(2,2);
      auto src = std::make_pair(src_str[0]-'a', src_str[1]-'1');
      auto dst = std::make_pair(dst_str[0]-'a', dst_str[1]-'1');
      action_tensor[src.first*src.second * dst.first*dst.second] = 1;

      std::vector<torch::Tensor> tgt = {action_tensor,torch::tensor({reward*parity}).to(torch::kCUDA)};
      auto target = torch::cat(tgt, 0).to(torch::kCUDA);
      auto l = loss(output, target).to(torch::kCUDA);
      l.backward();
      std::vector<torch::Tensor> parameters;
      for (auto parameter : model.parameters()) {
        parameters.push_back(parameter.to(torch::kCUDA));
      }
      torch::optim::SGD(parameters, 0.01).step();
      torch::optim::SGD(parameters, 0.01).zero_grad();
      parity *= -1;
    }
  };
  auto action_dist = [&model](thc::ChessRules state) {
    torch::Tensor fwd_tensor = model.forward({board_to_tensor(state).to(torch::kCUDA).view({1,119,8,8})}).toTensor().to(torch::kCUDA);
    return fwd_tensor.slice(0, 0, fwd_tensor.size(0) - 1);
  };
  auto apprentice = Apprentice<thc::ChessRules, std::string>(action_dist, evalf, trainf);
  auto root = std::make_unique<ExItNode<thc::ChessRules, std::string>>(mdp, apprentice, thc::ChessRules(), std::vector<ExItNode<thc::ChessRules, std::string>*>(), std::nullopt);
  
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
      root.reset(new ExItNode<thc::ChessRules, std::string>(mdp, apprentice, thc::ChessRules(), std::vector<ExItNode<thc::ChessRules, std::string>*>(), std::nullopt));
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
      root.reset(new ExItNode<thc::ChessRules, std::string>(mdp, apprentice, board, std::vector<ExItNode<thc::ChessRules, std::string>*>(), std::nullopt));
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
        best_move_str = cur_node->par_search(800, 0.5, false);
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
      std::vector<std::string> actions;

      for (int num_turns = 0; num_turns < steps; num_turns += 1) {
        if (num_turns % 5 == 0) {
          std::cout << "Saving the current model." << std::endl;
          model.save("apprentice.pt");
        }
        if (num_turns == 0 || over) {
          if (over) {
            // train step
            // TODO: batch this
            // TODO: train with state-action pairs as well
            //
            // apprentice.train() needs to take in a list of states, a list of actions, and a reward
            // the reward is the reward for the last state

            // train will handle all of the parity concerns internally.
            double reward = *mdp.reward(*(states.end()-1));
            apprentice.train(states, actions, reward);
          }

          std::cout << "Starting new game" << std::endl;
          states = std::vector<thc::ChessRules>();
          board = thc::ChessRules();
          root.reset(new ExItNode<thc::ChessRules, std::string>(mdp, apprentice, board, std::vector<ExItNode<thc::ChessRules, std::string>*>(), std::nullopt));
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
          root.reset(new ExItNode<thc::ChessRules, std::string>(mdp, apprentice, thc::ChessRules(), std::vector<ExItNode<thc::ChessRules, std::string>*>(), std::nullopt));
        }

        // play a move
        cur_node = root.get()->play(played);
        cur_node->state = board;
        auto best_move_str = cur_node->par_search(800, 0.5, false);
        actions.push_back(best_move_str);
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
