default:
	clang++ -std=c++20 -I./include/pytorch/torch/include -I./include/pytorch/torch/include/torch/csrc/api/include -I./include -L./include/pytorch/torch/lib -ltorch -lc10 -o main src/mcts.cpp include/thc.cpp
