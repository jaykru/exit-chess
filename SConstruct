# SConstruct file
import os
env = DefaultEnvironment()

libtorch_path = "/usr/include/torch"

env.Append(LINKFLAGS=[# "-static",
                      '-Wl,-L'+ libtorch_path +'/lib'])

env.Append(LIBS=['torch', 'torch_cpu', 'torch_global_deps', 'c10'] + ['pthread'] if os.name == 'posix' else [])
env.Append(LIBPATH=[libtorch_path + '/lib'])

# Add the vendored dependencies to the include and library paths

env.Append(CPPPATH=[libtorch_path
                   , libtorch_path + "/csrc/api/include"
                   , "include"])
# env.Append(LIBPATH=["vendor/projA/lib", "vendor/projB/lib"])


CPPFLAGS=["-std=c++20"]
# if DEBUG is defined
if os.getenv("DEBUG") is not None:
    CPPFLAGS.append(["-g"])
else:
    CPPFLAGS.append("-O3")
# Build the main program and link it with the vendored libraries
# use c++20 as the standard
env.Program("main", source=["src/mcts.cpp", "include/thc.cpp"], CPPFLAGS=CPPFLAGS)
