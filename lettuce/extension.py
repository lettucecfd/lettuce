# Importing Torch is required by generated Files to be loaded!

# noinspection PyUnresolvedReferences
import torch

# Generated Files for Cuda Extension

# noinspection PyUnresolvedReferences,PyProtectedMember
import lettuce._CudaExtension


def stream_and_collide(f, f_next, tau):
    # noinspection PyUnresolvedReferences,PyProtectedMember
    lettuce._CudaExtension.stream_and_collide(f, f_next, tau)
