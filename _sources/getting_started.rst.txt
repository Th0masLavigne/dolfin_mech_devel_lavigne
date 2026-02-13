Getting Started
===============

**dolfin_mech** simplifies the implementation of complex mechanics problems. Below are the basic steps to set up a simulation.

Prerequisites
-------------
Ensure you have a working FEniCS (2019.1.0) environment. 

.. code-block:: bash

   conda activate dolfin_mech

Basic Workflow
--------------
Most simulations follow this pattern:

1. **Define Geometry**: Load a mesh or generate one using built-in scripts like ``run_Ball_Mesh.py``.
2. **Select Material**: Choose from Elastic, Inelastic, or Porous models.
3. **Define Operators**: Add physics (Inertia, Hyperelasticity, Darcy Flow).
4. **Solve**: Use the ``NonlinearSolver`` to find the solution.

First Example: Hyperelastic Ball
--------------------------------
To run your first simulation of a hyperelastic ball under loading, use:

.. code-block:: bash

   python -m dolfin_mech.run_Ball_Hyperelasticity

This example demonstrates the core ``Problem_Hyperelasticity`` class and how to apply surface pressure.