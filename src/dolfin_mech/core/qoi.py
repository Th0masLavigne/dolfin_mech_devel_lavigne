# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Quantity of Interest (QOI) management module.

This module provides the QOI class to handle scalar value extraction from
finite element simulations, supporting both global integration and point-wise
evaluation.
"""

import dolfin

################################################################################


class QOI:
	"""Class representing a Quantity of Interest (QOI).

	A QOI is a scalar value computed at the end of each time/load step. It is
	used for post-processing and analysis, often representing global aggregates
	(integrals) or specific point values.

	Common examples include:
	    - **Global Integrals:** Total volume, total strain energy, homogenized stress.
	    - **Surface Integrals:** Reaction forces, fluxes.
	    - **Point Values:** Displacement at a specific node, pressure at a sensor.

	The class handles the calculation method (assembly vs. direct evaluation)
	and optional normalization or time-derivative scaling.

	Args:
	    name (str): The identifier for the QOI (e.g., "vol", "sigma_XX").
	    expr (ufl.core.expr.Expr, optional): The UFL form or expression to evaluate.
	    expr_lst (list, optional): A list of UFL forms, one for each load step.
	    norm (float, optional): A scaling factor to divide the result by.
	        Defaults to 1.0.
	    constant (float, optional): A constant offset added to the result.
	        Defaults to 0.0.
	    divide_by_dt (bool, optional): If True, divides the result by the time
	        step ``dt``. Defaults to False.
	    form_compiler_parameters (dict, optional): Compiler options for the
	        FEniCS assembler. Defaults to {}.
	    point (tuple or list, optional): The spatial coordinate for point-wise
	        evaluation. Required if ``update_type="direct"``.
	    update_type (str, optional): The evaluation strategy. Options are:

	        * ``"assembly"``: Performs a global FEM integration (default).
	        * ``"direct"``: Evaluates a function at a specific point.
	"""

	def __init__(
		self,
		name,
		expr=None,
		expr_lst=None,
		norm=1.0,
		constant=0.0,
		divide_by_dt=False,
		form_compiler_parameters={},
		point=None,
		update_type="assembly",
	):
		"""Initializes a Quantity of Interest."""
		self.name = name
		self.expr = expr
		self.expr_lst = expr_lst
		self.norm = norm
		self.constant = constant
		self.divide_by_dt = divide_by_dt
		self.form_compiler_parameters = form_compiler_parameters
		self.point = point

		if update_type == "assembly":
			self.update = self.update_assembly
		elif update_type == "direct":
			self.update = self.update_direct

	def update_assembly(self, dt=None, k_step=None, expr=None):
		"""Computes the QOI value by assembling (integrating) over the mesh.

		Args:
		    dt (float, optional): Time step size (required if ``divide_by_dt`` is True).
		    k_step (int, optional): Current step index (used to select from ``expr_lst``).
		    expr (ufl.core.expr.Expr, optional): Optional override for the expression.
		"""
		# print(self.name)
		# print(self.expr)
		# print(self.form_compiler_parameters)
		if self.expr is not None:
			self.value = dolfin.assemble(self.expr, form_compiler_parameters=self.form_compiler_parameters)
		else:
			if k_step is None:
				self.value = dolfin.assemble(self.expr_lst[0], form_compiler_parameters=self.form_compiler_parameters)
			else:
				self.value = dolfin.assemble(
					self.expr_lst[k_step - 1], form_compiler_parameters=self.form_compiler_parameters
				)

		self.value += self.constant
		self.value /= self.norm

		if self.divide_by_dt:
			assert dt != 0, "dt (=" + str(dt) + ") should be non zero. Aborting."
			self.value /= dt

	def update_direct(self, dt=None, k_step=None):
		"""Computes the QOI value by evaluating at a specific point.

		This is a MPI-safe version of point evaluation. An error is raised if
		the point is outside the domain.

		Args:
		    dt (float, optional): Time step size (required if ``divide_by_dt`` is True).
		    k_step (int, optional): Current step index.
		"""
		# MPI safe version
		try:
			local_value = self.expr(self.point)
			found = 1.0
		except RuntimeError:
			local_value = 0.0
			found = 0.0

		comm = dolfin.MPI.comm_world
		global_value = dolfin.MPI.sum(comm, local_value)
		global_found = dolfin.MPI.sum(comm, found)

		if global_found == 0:
			if dolfin.MPI.rank(comm) == 0:
				raise ValueError(f"Erreur : Point {self.point} does not belong to any of the MPI subdomains.")
			else:
				raise ValueError("Point outside of domain (secondary rank)")

		# if the point is shared between procs (interface)
		self.value = global_value / global_found

		self.value += self.constant
		self.value /= self.norm

		if (self.divide_by_dt) and (dt is not None):
			self.value /= dt
