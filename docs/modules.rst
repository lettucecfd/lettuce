=============
API reference
=============

Everything listed here is available directly from the top-level package, e.g.
``lettuce.BGKCollision``, conventionally imported as ``import lettuce as lt``.

.. currentmodule:: lettuce


Core
====

.. autoclass:: Context
    :members:
    :undoc-members:

.. autoclass:: Simulation
    :members:
    :undoc-members:

.. autoclass:: BreakableSimulation
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: StreamingStrategy
    :members:
    :undoc-members:

.. autoclass:: Flow
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: UnitConversion
    :members:
    :undoc-members:


Stencils
========

.. autoclass:: Stencil
    :members:
    :undoc-members:

.. autoclass:: TorchStencil
    :members:
    :undoc-members:

.. autoclass:: D1Q3
    :show-inheritance:

.. autoclass:: D2Q9
    :show-inheritance:

.. autoclass:: D3Q15
    :show-inheritance:

.. autoclass:: D3Q19
    :show-inheritance:

.. autoclass:: D3Q27
    :show-inheritance:


Flows
=====

.. autoclass:: ExtFlow
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: TaylorGreenVortex
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: TaylorGreenVortex2D
    :show-inheritance:

.. autoclass:: TaylorGreenVortex3D
    :show-inheritance:

.. autoclass:: CouetteFlow2D
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: PoiseuilleFlow2D
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: DoublyPeriodicShear2D
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: DecayingTurbulence
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: Obstacle
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: Cavity2D
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: LambOseenVortex2D
    :members:
    :undoc-members:
    :show-inheritance:

.. data:: flow_by_name

    Maps a short name (e.g. ``'taylor2d'``, ``'couette2d'``) to a tuple of a
    flow class and its default stencil. The command line interface uses these
    names to select a flow.


Collision
=========

.. autoclass:: Collision
    :members:
    :undoc-members:

.. autoclass:: BGKCollision
    :show-inheritance:

.. autoclass:: TRTCollision
    :show-inheritance:

.. autoclass:: MRTCollision
    :show-inheritance:

.. autoclass:: RegularizedCollision
    :show-inheritance:

.. autoclass:: KBCCollision
    :show-inheritance:

.. autoclass:: KBCCollision2D
    :show-inheritance:

.. autoclass:: KBCCollision3D
    :show-inheritance:

.. autoclass:: SmagorinskyCollision
    :show-inheritance:

.. autoclass:: NoCollision
    :show-inheritance:


Equilibrium
===========

.. autoclass:: Equilibrium
    :members:
    :undoc-members:

.. autoclass:: QuadraticEquilibrium
    :show-inheritance:

.. autoclass:: QuadraticEquilibriumLessMemory
    :show-inheritance:

.. autoclass:: IncompressibleQuadraticEquilibrium
    :show-inheritance:


Boundary conditions
===================

.. autoclass:: Boundary
    :members:
    :undoc-members:

.. autoclass:: BounceBackBoundary
    :show-inheritance:

.. autoclass:: AntiBounceBackOutlet
    :show-inheritance:

.. autoclass:: EquilibriumBoundaryPU
    :show-inheritance:

.. autoclass:: EquilibriumOutletP
    :show-inheritance:

.. autoclass:: PartiallySaturatedBC
    :show-inheritance:


Forces
======

.. autoclass:: Force
    :members:
    :undoc-members:

.. autoclass:: Guo
    :show-inheritance:

.. autoclass:: ShanChen
    :show-inheritance:


Reporters and observables
=========================

.. autoclass:: Reporter
    :members:
    :undoc-members:

.. autoclass:: ObservableReporter
    :show-inheritance:

.. autoclass:: ErrorReporter
    :show-inheritance:

.. autoclass:: VTKReporter
    :show-inheritance:

.. autoclass:: HDF5Reporter
    :show-inheritance:

.. autoclass:: ProgressReporter
    :show-inheritance:

.. autoclass:: FailureReporterBase
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: NaNReporter
    :show-inheritance:

.. autoclass:: HighMaReporter
    :show-inheritance:

.. autoclass:: Observable
    :members:
    :undoc-members:

.. autoclass:: MaximumVelocity
    :show-inheritance:

.. autoclass:: IncompressibleKineticEnergy
    :show-inheritance:

.. autoclass:: Enstrophy
    :show-inheritance:

.. autoclass:: EnergySpectrum
    :show-inheritance:

.. autoclass:: Mass
    :show-inheritance:

.. autoclass:: LettuceDataset
    :members:
    :show-inheritance:

.. autofunction:: write_image

.. autofunction:: write_vtk


Utilities
=========

.. autofunction:: get_subclasses

.. autofunction:: torch_gradient

.. autofunction:: torch_jacobi

.. autofunction:: grid_fine_to_coarse

.. autofunction:: append_axes


Exceptions and warnings
=======================

.. autoexception:: LettuceException

.. autoexception:: LettuceWarning

.. autoexception:: InefficientCodeWarning

.. autoexception:: ExperimentalWarning
