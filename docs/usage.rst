=====
Usage
=====

A first simulation
==================

The following script runs a two-dimensional Taylor-Green vortex and prints the
performance in million lattice updates per second (MLUPS). It is the file
``examples/00_simplest_TGV.py`` from the repository:

.. literalinclude:: ../examples/00_simplest_TGV.py
   :language: python

Everything is available from the top-level package, which is conventionally
imported as ``lt``.

Building blocks
===============

A simulation is assembled from a few components:

``Context``
    Where and in which precision the simulation runs. ``device`` defaults to
    the first CUDA device if one is available and to the CPU otherwise;
    ``dtype`` defaults to ``torch.float32``. Pass ``device='cpu'`` or
    ``dtype=torch.float64`` to override.

Stencil
    The discrete velocity set, e.g. ``lt.D2Q9``, ``lt.D3Q19`` or ``lt.D3Q27``.

Flow
    The physical setup: domain, initial condition and boundaries, e.g.
    ``lt.TaylorGreenVortex``, ``lt.CouetteFlow2D``, ``lt.PoiseuilleFlow2D`` or
    ``lt.Obstacle``. Every flow carries a ``units`` object that converts
    between physical and lattice units (``flow.units``).

Collision
    The collision operator, e.g. ``lt.BGKCollision``. Its relaxation time is
    usually taken from the flow: ``tau=flow.units.relaxation_parameter_lu``.

``Simulation``
    Combines flow, collision and a list of reporters. Calling
    ``simulation(num_steps)`` advances the simulation and returns the
    performance in MLUPS.

Reporters
=========

Reporters are called between time steps and collect output. They are passed to
``Simulation`` as a list or appended to ``simulation.reporter`` later:

.. code-block:: python

    energy = lt.ObservableReporter(lt.IncompressibleKineticEnergy(flow),
                                   interval=10, out=None)
    simulation.reporter.append(energy)
    simulation.reporter.append(lt.VTKReporter(interval=100,
                                              filename_base='./data/tgv'))

    simulation(1000)
    print(energy.out[-1])  # [time step, time in PU, kinetic energy]

``ObservableReporter`` records scalar observables such as
``IncompressibleKineticEnergy``, ``Enstrophy`` or ``MaximumVelocity``.
``VTKReporter`` writes the velocity and pressure fields as ``.vtr`` files for
ParaView. ``examples/03_outputs_TGV.py`` shows both in a complete script.

Command line
============

Installing lettuce also installs the ``lettuce`` command:

.. code-block:: console

    $ lettuce --version
    $ lettuce --no-cuda benchmark
    $ lettuce --no-cuda convergence

``benchmark`` runs a short simulation and reports its performance;
``convergence`` checks the order of convergence on a two-dimensional
Taylor-Green vortex. Global options select the device and precision:
``--cuda/--no-cuda``, ``-i/--gpu-id`` and ``-p/--precision`` (``half``,
``single`` or ``double``). ``lettuce --help`` and ``lettuce <command> --help``
list all options.

Further examples
================

The `examples directory <https://github.com/lettucecfd/lettuce/tree/master/examples>`_
contains scripts and Jupyter notebooks for further flows, among them Couette
and Poiseuille flow, flow around an obstacle, decaying turbulence, a lid-driven
cavity and porous media.
