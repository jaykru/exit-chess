# Root-Parallel Expert Iteration in C++

This is a fast, [root-parallel](https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf) C++ implementation of the Expert Iteration algorithm proposed by Anthony et al. The implementation is generic in a choice of a Markov decision process as well as your choice of an *apprentice*.

The current version of the program presents a UCI interface to be used with Lichess as a chess bot. The apprentice used in the chess bot is a trivial one, which evaluates every state to 0 and performs no training. With the trivial apprentice, the Expert Iteration algorithm degenerates to plain Monte-Carlo Tree Search. Even with pure MCTS, the program has managed to beat a human player :) A better apprentice model is in progress.

## Running it

You will need to add libtorch as a submodule under `include/` and build it with CMake.[^1] You need to have SConstruct installed. The following incanation builds and runs the program on macOS and Linux.
```
git clone https://github.com/jaykru/mcts-chess
cd mcts-chess
scons && ./run.sh
```

## Usage

The main executable presents a [UCI](https://wbec-ridderkerk.nl/html/UCIProtocol.html) chess interface. You can play manually with this, but it's recommended that you instead hook it up with [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot). Some tweaking to lichess-bot is required to make it tolerant of long thinking time when using high iteration counts for the tree search.

## License

Copyright Jay Kruer 2023. You probably won't want to use the code (yet) but
contact me if you do. I haven't decided on a license yet.

[^1]: I'd like to make this less manual and kludgy in the future, but this is a hobby project for now...
