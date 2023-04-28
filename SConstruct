# SConstruct file
import os
env = DefaultEnvironment()

env.Append(LIBS=['pthread'] if os.name == 'posix' else [])
env.Append(CPPPATH=["include"])

CPPFLAGS=["-std=c++20"]

if os.getenv("DEBUG") is not None:
    CPPFLAGS.append(["-g"])
else:
    CPPFLAGS.append("-O3")
env.Program("main", source=["src/mcts.cpp", "include/thc.cpp"], CPPFLAGS=CPPFLAGS)
