#!/bin/bash

# check if we're on darwin
if [[ "$OSTYPE" == "darwin"* ]]; then
    # we're on darwin, so we need to use DYLD_LIBRARY_PATH
    DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)/include/pytorch/torch/lib ./main
else
    # we're on linux, so we need to use LD_LIBRARY_PATH
    # if debug is set wrap in lldb
    if [[ -z "${DEBUG}" ]]; then
        LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/include/pytorch/torch/lib ./main
    else
        LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/include/pytorch/torch/lib lldb ./main
    fi
fi