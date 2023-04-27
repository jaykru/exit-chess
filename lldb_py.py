#!/usr/bin/env python3
import lldb
target = lldb.debugger.GetSelectedTarget()

def pp():
    for i in range(6):
        val = target.EvaluateExpression(f"this->children[{i}]->debug()")
        print(f"Child {i}:\n \debug: {val};")
