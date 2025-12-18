import torch

from timeit import default_timer as timer
from typing import List

from lettuce._simulation import *
from lettuce import Flow

__all__ = ['EbbSimulation']

class EbbSimulation(Simulation):
    """
        advanced simulation class, with
        - (additional) post_streaming_boundaries (substeps: C -> S -> B -> R)
        - post_collision population information for post_streaming_boundaries
    """
    def __init__(self, flow: 'Flow', collision: 'Collision',
                 reporter: List['Reporter']):
        flow.context.use_native = False
        super().__init__(flow, collision, reporter)

        if hasattr(flow.__class__, 'post_streaming_boundaries'):
            # list of boundaries that are applied AFTER the streaming step
            self.post_streaming_boundaries = flow.post_streaming_boundaries
        else:
            self.post_streaming_boundaries = []

        # adjust No_collision_ and no_streaming_mask for use of post_streaming_boundaries (!)
        if len(self.post_streaming_boundaries) > 0:

            # create NCM and NSM,
            # ...if there were no pre_ or post_boundaries that triggered
            # ...their creation in super-class
            if self.no_collision_mask is None:
                # fill masks with value of self.collision_index (= number of pre-boundaries)
                self.no_collision_mask = self.context.full_tensor(
                    flow.resolution, self.collision_index, dtype=torch.uint8)
            if self.no_streaming_mask is None:
                self.no_streaming_mask = self.context.full_tensor([flow.stencil.q,
                                                                   *flow.resolution],
                                                                  self.collision_index,
                                                                  dtype=torch.uint8)


            for i, boundary in enumerate(self.post_streaming_boundaries,
                                         start=self.collision_index + 1
                                               + len(self.post_boundaries)):
# FOR DEBUGGING: print("SIMULATION: creating no_collision_mask entries for: i, boundary = ", i, boundary)
# FOR DEBUGGING: print("collision index is:", self.collision_index)
                ncm = boundary.make_no_collision_mask(
                    [it for it in self.flow.f.shape[1:]], context=self.context)
                if ncm is not None:
                    self.no_collision_mask[ncm] = i
                nsm = boundary.make_no_streaming_mask(
                    [it for it in self.flow.f.shape], context=self.context)
                if nsm is not None:
                    self.no_streaming_mask |= nsm

        # get indices of post_streaming_boundaries, that need f_collided to be stored
        self.store_f_collided_post_streaming_boundaries_index = []
        for i, boundary in enumerate(self.post_streaming_boundaries):
            if hasattr(boundary.__class__, "store_f_collided"):
                # append boundary to lis of boundaries that need
                # ...f_collided state between collision and streaming substep
                self.store_f_collided_post_streaming_boundaries_index.append(i)
            if hasattr(boundary.__class__, "initialize_f_collided"):
                # initialize the f_collided storage in each of those boundaries
                boundary.initialize_f_collided()

        # redefine collide_and_stream() to include the storage of f_collided
        # ...for use in post_streaming_boundaries
        def collide_and_stream(*_, **__):
            self._collide()
            # pass f to post_streaming_boundaries between collision and streaming substep
            self._pass_f_collided()
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
