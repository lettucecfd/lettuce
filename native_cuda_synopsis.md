The native-cuda update introduced some major updates to the lettuce structure. Here, you find the main changes. Below, you find the steps to update your runfiles:

## Changes

1. `lt.Lattice` is deprecated.

The `device` and `dtype` fields moved to `lt.Context`. The `Stencil` object or class (either works) is passed to `Flow.__init__`.

2. `lt.Streaming` is deprecated.

Streaming is now always handled as defined in `lt.StandardStreaming`.

3. `lt.Stencil` is stored in a field of `lt.Flow` and `D` is now lower case.

So, the Dimensions can be accessed via `flow.stencil.d` instead of `flow.units.lattice.D`.

4. Further changes:
- `lt.Collision` does not neet `lattice` definition.
- `reporters` was renamed to `reporter`
- `no_collision_mask` now includes the boundary's identifier (important for custom flows).

## Updating runfiles
1. Move device and dtype declarations into lt.Context and remove all lattice declarations.
2. Move stencil declaration to lt.Flow and add context declaration.
3. Remove lt.Streaming initialization.
4. Rename reporters to reporter
