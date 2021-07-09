# Importing Torch is required by generated Files to be loaded!

# noinspection PyUnresolvedReferences
import torch

# Generated Files for Cuda Extension

# noinspection PyUnresolvedReferences,PyProtectedMember
import lettuce._CudaExtension


def stream_and_collide(f, f_next, tau):
    # noinspection PyUnresolvedReferences,PyProtectedMember
    lettuce._CudaExtension.stream_and_collide(f, f_next, tau)
    torch.cuda.synchronize()  # TODO


def stream(f, f_next):
    # noinspection PyUnresolvedReferences,PyProtectedMember
    lettuce._CudaExtension.stream(f, f_next)
    torch.cuda.synchronize()  # TODO


def collide(f, f_next, tau):
    # noinspection PyUnresolvedReferences,PyProtectedMember
    lettuce._CudaExtension.collide(f, f_next, tau)
    torch.cuda.synchronize()  # TODO
