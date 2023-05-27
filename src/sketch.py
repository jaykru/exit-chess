import torch
from torch.nn import *

class Skipped(Module):
    def __init__(self,inner,in_channels,out_channels):
        super().__init__()
        self.inner = inner
        if (out_channels != in_channels):
            self.fix = Sequential(Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
                                  , BatchNorm2d(out_channels))
        else:
            self.fix = Identity()

    def forward(self, x):
        out = self.inner(x)
        return out + self.fix(x)

class Forked(Module):
    def __init__(self, parent, left, right):
        super().__init__()
        self.parent = parent
        self.left = left
        self.right = right

    def forward(self, x):
        thru_parent = self.parent(x)
        # FIXME: these reshapes are valid when we have effectively one-dim'l
        # tensors, but i'm not sure it makes sense in the general case.
        return torch.concat([self.left(thru_parent).reshape(-1), self.right(thru_parent).reshape(-1)])

def pre_block():
    return Sequential(
                Conv2d(in_channels=119, out_channels=256, kernel_size=3, padding='same', stride=1),
                BatchNorm2d(256),
                ReLU()
           )

def res_sub_block():
    return Sequential(
        Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same', stride=1),
        BatchNorm2d(256),
        ReLU()
    )

def res_block():
    return Skipped(Sequential(res_sub_block(), res_sub_block()), in_channels=256, out_channels=256)



def policy_head():
    return Sequential(
          Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1)
        , BatchNorm2d(2)
        , ReLU()
        , Flatten(0,-1)
        , Linear(2*8*8, 64**2) # 64**2 outputs, ~> not all of these are legal
                               # moves; e.g. everything along the diagonal is an
                               # illegal "pass" move.
        , Softmax(dim=0)
    )

def value_head():
    return Sequential(
          Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1)
        , BatchNorm2d(1)
        , ReLU()
        , Flatten(0,-1) # , Reshaper([-1,8*8])
        , Linear(8*8, 256)
        , ReLU()
        , Linear(256, 1)
        , Tanh()
    )

alpha_prefix = Sequential(*([pre_block()] + [res_block()]*39))
alpha_net = Forked(parent = Sequential(*([pre_block()] + [res_block()]*39)),
                   left = policy_head(),
                   right = value_head())

example_board = torch.randn(1,119,8,8)
traced_model = torch.jit.trace(alpha_net, example_board)
traced_model.save("alphanet.pt")
