# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Kinematics implementation for non-linear mechanics.

This module provides the Kinematics class which handles the computation and
storage of deformation tensors, invariants, and frame transformations.
"""

import dolfin

################################################################################


class Kinematics:
	r"""Class to compute and store kinematic quantities for non-linear solid mechanics.

	This class computes standard deformation tensors, their invariants, and
	isochoric (bar) versions based on the displacement field :math:`\mathbf{U}`.
	It also supports time-stepping (old states) and local frame transformations.

	Args:
		U (dolfin.Function): Current displacement field.
		U_old (dolfin.Function, optional): Displacement field from the previous
		        time step. Defaults to None.
		Q_expr (dolfin.Expression, optional): Rotation matrix for local
		        coordinate systems. Defaults to None.

	Attributes:
	    U (dolfin.Function): The current displacement field.
	    dim (int): Spatial dimension.
	    I (dolfin.Identity): Identity tensor of dimension ``dim``.
	    F (ufl.Form): Deformation gradient tensor :math:`\mathbf{F} = \mathbf{I} + \nabla \mathbf{U}`.
	    J (ufl.Form): Determinant of the deformation gradient :math:`J = \det(\mathbf{F})`.
	    C (ufl.Form): Right Cauchy-Green deformation tensor :math:`\mathbf{C} = \mathbf{F}^T \mathbf{F}`.
	    E (ufl.Form): Green-Lagrange strain tensor :math:`\mathbf{E} = \mathbf{C} - \mathbf{I}`.
	    F_bar (ufl.Form): Isochoric deformation gradient :math:`\bar{\mathbf{F}} = J^{-1/d} \mathbf{F}`.
	    C_bar (ufl.Form): Isochoric Right Cauchy-Green tensor :math:`\bar{\mathbf{C}} = \bar{\mathbf{F}}^T \bar{\mathbf{F}}`.
	"""

	def __init__(self, U, U_old=None, Q_expr=None):
		"""Initialize the Kinematics object and compute deformation tensors.

		Args:
		    U (dolfin.Function): Current displacement field.
		    U_old (dolfin.Function, optional): Displacement field from the previous
		        time step. Defaults to None.
		    Q_expr (dolfin.Expression, optional): Rotation matrix for local
		        coordinate systems. Defaults to None.

		Notes:
		    The class automatically computes spherical and deviatoric parts of the
		    strain, as well as mid-point configurations if ``U_old`` is provided.
		"""
		self.U = U

		self.dim = self.U.ufl_shape[0]
		self.I = dolfin.Identity(self.dim)

		self.F = self.I + dolfin.grad(self.U)
		self.F = dolfin.variable(self.F)
		self.J = dolfin.det(self.F)
		self.C = self.F.T * self.F
		self.C = dolfin.variable(self.C)
		self.C_inv = dolfin.inv(self.C)
		self.IC = dolfin.tr(self.C)
		self.IIC = (dolfin.tr(self.C) * dolfin.tr(self.C) - dolfin.tr(self.C * self.C)) / 2
		self.E = (self.C - self.I) / 2
		self.E = dolfin.variable(self.E)

		self.E_sph = dolfin.tr(self.E) / self.dim * self.I
		self.E_dev = self.E - self.E_sph

		self.F_bar = self.J ** (-1 / self.dim) * self.F
		self.C_bar = self.F_bar.T * self.F_bar
		self.C_bar_inv = dolfin.inv(self.C_bar)
		self.IC_bar = dolfin.tr(self.C_bar)
		self.IIC_bar = (dolfin.tr(self.C_bar) * dolfin.tr(self.C_bar) - dolfin.tr(self.C_bar * self.C_bar)) / 2
		self.E_bar = (self.C_bar - self.I) / 2

		if U_old is not None:
			self.U_old = U_old

			self.F_old = self.I + dolfin.grad(U_old)
			self.J_old = dolfin.det(self.F_old)
			self.C_old = self.F_old.T * self.F_old
			self.E_old = (self.C_old - self.I) / 2

			self.F_bar_old = self.J_old ** (-1 / self.dim) * self.F_old
			self.C_bar_old = self.F_bar_old.T * self.F_bar_old
			self.E_bar_old = (self.C_bar_old - self.I) / 2

			self.F_mid = (self.F_old + self.F) / 2
			self.J_mid = (self.J_old + self.J) / 2
			self.C_mid = (self.C_old + self.C) / 2
			self.E_mid = (self.E_old + self.E) / 2

			self.F_bar_mid = (self.F_bar_old + self.F_bar) / 2
			self.C_bar_mid = (self.C_bar_old + self.C_bar) / 2
			self.E_bar_mid = (self.E_bar_old + self.E_bar) / 2

		if Q_expr is not None:
			self.Q_expr = Q_expr

			self.E_loc = self.Q_expr * self.E * self.Q_expr.T  # MG20211215: This should work, right?
			self.E_bar_loc = self.Q_expr * self.E_bar * self.Q_expr.T  # MG20211215: This should work, right?
			# self.E_loc     = dolfin.dot(dolfin.dot(self.Q_expr, self.E    ), self.Q_expr.T)
			# self.E_bar_loc = dolfin.dot(dolfin.dot(self.Q_expr, self.E_bar), self.Q_expr.T)

			if U_old is not None:
				self.E_old_loc = self.Q_expr * self.E_old * self.Q_expr.T  # MG20211215: This should work, right?
				self.E_bar_old_loc = (
					self.Q_expr * self.E_bar_old * self.Q_expr.T
				)  # MG20211215: This should work, right?
				# self.E_old_loc     = dolfin.dot(dolfin.dot(self.Q_expr, self.E_old    ), self.Q_expr.T)
				# self.E_bar_old_loc = dolfin.dot(dolfin.dot(self.Q_expr, self.E_bar_old), self.Q_expr.T)
