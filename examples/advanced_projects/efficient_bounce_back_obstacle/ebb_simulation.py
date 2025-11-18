import warnings

import torch
import numpy as np

from timeit import default_timer as timer
from typing import List, Optional
from abc import ABC, abstractmethod

from lettuce.cuda_native import NativeCollision, Generator, StreamingStrategy

from lettuce._simulation import *
from lettuce import Flow

__all__ = ['EbbSimulation']

class EbbSimulation(Simulation):
    def __init__(self, flow: 'Flow', collision: 'Collision',
                 reporter: List['Reporter']):
        flow.context.use_native = False
        super().__init__(flow, collision, reporter)

        if hasattr(flow, 'post_streaming_boundaries'):
            # list of boundaries that are applied AFTER the streaming step
            self.post_streaming_boundaries = flow.post_streaming_boundaries
        else:
            self.post_streaming_boundaries = []

        # get indices of post_streaming_boundaries, that need f_collided to be stored
        self.store_f_collided_post_streaming_boundaries_index = []

        for i, boundary in enumerate(self.post_streaming_boundaries):
            if hasattr(boundary, "store_f_collided"):
                self.store_f_collided_post_streaming_boundaries_index.append(i)

        # redefine collide_and_stream() to include the storage of f_collided for use in post_streaming_boundaries
        def collide_and_stream(*_, **__):
            self._collide()
            self._pass_f_collided() # pass f to post_streaming_boundaries between collision and streaming substep
            self._stream()

        self._collide_and_stream = collide_and_stream


    def _post_streaming_boundaries(self):
        # runs all the post_streaming_boundaries; required for efficient BBBC

        for boundary in self.post_streaming_boundaries:
            boundary(self.flow)

    def _pass_f_collided(self):
        # passes f to post_streaming_boundaries, for later use

        for idx in self.store_f_collided_post_streaming_boundaries_index:
            self.post_streaming_boundaries[idx].store_f_collided(self.flow.f)

    def __call__(self, num_steps):
        beg = timer()

        if self.flow.i == 0:
            self._report()

        for _ in range(num_steps):
            self._collide_and_stream(self)
            self._post_streaming_boundaries()
            self.flow.i += 1
            self._report()

        end = timer()
        return num_steps * self.flow.rho().numel() / 1e6 / (end - beg)
