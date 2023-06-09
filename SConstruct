# SConstruct file
import os
env = DefaultEnvironment()

libtorch_path = "./libtorch"

env.Append(LINKFLAGS=[# "-static",
                      '-Wl,-L'+ libtorch_path +'/lib']
                      # + ['-Wl,-L' + os.path.join(libtorch_path+'/lib', dir)
                      #    for _,dirs,_ in os.walk(libtorch_path+'/lib')
                      #    for dir in dirs]
           )

env.Append(LIBS=['torch', 'torch_cpu', 'torch_hip', 'rocblas', 'torch_global_deps', 'c10'] + ['pthread'] if os.name == 'posix' else [])
env.Append(LIBPATH=[libtorch_path + '/lib'])

# Add the vendored dependencies to the include and library paths

# Recursively collect all directories in a given path
directories = []
for root, dirs, files in os.walk("./libtorch"):
    directories.extend([os.path.join(root, d) for d in dirs])

env.Append(CPPPATH=[ libtorch_path + "/include"
                     , "include"] + directories)

CPPFLAGS=["-std=c++20"]
# if DEBUG is defined
if os.getenv("DEBUG") is not None:
    CPPFLAGS.append(["-g"])
else:
    CPPFLAGS.append("-O3")
# Build the main program and link it with the vendored libraries
# use c++20 as the standard
env.Program("main", source=["src/mcts.cpp", "include/thc.cpp"], CPPFLAGS=CPPFLAGS)
