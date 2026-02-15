# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Inverse kinematics implementation for non-linear solid mechanics.

This module provides the InverseKinematics class, which computes deformation
quantities based on a displacement field defined on the deformed configuration.
"""

import dolfin

################################################################################


class InverseKinematics:
	r"""Class to compute and store kinematic quantities for inverse non-linear solid mechanics.

	In the inverse formulation, the displacement field :math:`\mathbf{u}` is defined
	on the deformed (spatial) configuration. The deformation gradient :math:`\mathbf{F}`
	is computed as the inverse of the mapping from the reference to the deformed
	configuration: :math:`\mathbf{F} = (\mathbf{I} + \nabla \mathbf{u})^{-1}`.

	Args:
	    u (dolfin.Function): Current displacement field defined on the spatial mesh.
	    u_old (dolfin.Function, optional): Displacement field from the previous
	      time step. Defaults to None.

	Attributes:
	    u (dolfin.Function): The current displacement field in the spatial configuration.
	    dim (int): Spatial dimension.
	    I (dolfin.Identity): Identity tensor of dimension ``dim``.
	    f (ufl.Form): Spatial displacement gradient :math:`\mathbf{f} = \mathbf{I} + \nabla \mathbf{u}`.
	    F (ufl.Form): Deformation gradient tensor :math:`\mathbf{F} = \mathbf{f}^{-1}`.
	    J (ufl.Form): Determinant of the deformation gradient :math:`J = \det(\mathbf{F})`.
	    C (ufl.Form): Right Cauchy-Green deformation tensor :math:`\mathbf{C} = \mathbf{F}^T \mathbf{F}`.
	    E (ufl.Form): Green-Lagrange strain tensor :math:`\mathbf{E} = \frac{1}{2}(\mathbf{C} - \mathbf{I})`.
	    F_bar (ufl.Form): Isochoric deformation gradient :math:`\bar{\mathbf{F}} = J^{-1/d} \mathbf{F}`.
	    C_bar (ufl.Form): Isochoric Right Cauchy-Green tensor :math:`\bar{\mathbf{C}} = \bar{\mathbf{F}}^T \bar{\mathbf{F}}`.
	"""

	def __init__(self, u, u_old=None):
		"""Initialize the InverseKinematics object and compute inverse deformation tensors.

		Args:
		    u (dolfin.Function): Current displacement field defined on the spatial mesh.
		    u_old (dolfin.Function, optional): Displacement field from the previous
		        time step. Defaults to None.

		Notes:
		    The class uses UFL's ``inv()`` to compute the deformation gradient
		    from the spatial gradient. This is typically used in inverse problems
		    where the reference configuration is the unknown.
		"""
		self.u = u

		self.dim = self.u.ufl_shape[0]
		self.I = dolfin.Identity(self.dim)

		self.f = self.I + dolfin.grad(self.u)
		self.F = dolfin.inv(self.f)
		self.F = dolfin.variable(self.F)
		self.J = dolfin.det(self.F)
		self.C = self.F.T * self.F
		self.C = dolfin.variable(self.C)
		self.C_inv = dolfin.inv(self.C)
		self.IC = dolfin.tr(self.C)
		self.IIC = (dolfin.tr(self.C) * dolfin.tr(self.C) - dolfin.tr(self.C * self.C)) / 2
		self.E = (self.C - self.I) / 2
		self.E = dolfin.variable(self.E)

		self.F_bar = self.J ** (-1 / self.dim) * self.F
		self.C_bar = self.F_bar.T * self.F_bar
		self.IC_bar = dolfin.tr(self.C_bar)
		self.IIC_bar = (dolfin.tr(self.C_bar) * dolfin.tr(self.C_bar) - dolfin.tr(self.C_bar * self.C_bar)) / 2
		self.E_bar = (self.C_bar - self.I) / 2

		if u_old is not None:
			self.u_old = u_old

			self.f_old = self.I + dolfin.grad(self.u_old)
			self.F_old = dolfin.inv(self.f_old)
			self.J_old = dolfin.det(self.F_old)
			self.C_old = self.F_old.T * self.F_old
			self.E_old = (self.C_old - self.I) / 2

			self.F_bar_old = self.J_old ** (-1 / self.dim) * self.F_old
			self.C_bar_old = self.F_bar_old.T * self.F_bar_old
			self.E_bar_old = (self.C_bar_old - self.I) / 2
