#include <random>
#include <vector>

template <class Iter, class Pred>
Iter argmax(Iter begin, Iter end, Pred f) {
  if (begin == end) {
    return end;
  }
  Iter best = begin;  
  auto bestVal = f(*begin);
  for (Iter it = begin; it != end; it++) {
    auto val = f(*it);
    if (val > bestVal) {
      bestVal = val;
      best = it;
    }
  }
  return best;
}

template<typename T>
T select_randomly(std::mt19937& g, std::vector<T> in) {
  assert(in.size() > 0);
  std::vector<T> out;
  std::sample(in.begin(), in.end(), std::back_inserter(out), 
              1,
              g);
  return out[0];
}