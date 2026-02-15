# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the SurfaceForce operators.

Implement the virtual work of external
tractions in both finite strain (accounting for Nanson's formula and surface
area changes) and small strain frameworks.
"""

import dolfin

from ...core import TimeVaryingConstant
from ..operator import Operator

################################################################################


class SurfaceForce(Operator):
	r"""Operator representing an external surface force (traction) in a large deformation (finite strain) framework.

	This operator assembles the virtual work done by an external traction vector
	prescribed on a boundary. It accounts for the change in surface area and
	orientation due to deformation using Nanson's formula.

	The residual contribution is:

	.. math::
	    \delta \Pi_{ext} = - \int_{\Gamma_0} \mathbf{T} \cdot \delta \mathbf{u} \, J \|\mathbf{F}^{-T} \mathbf{N}\| \, d\Gamma_0

	where :math:`\mathbf{F}` is the deformation gradient and :math:`\mathbf{N}`
	is the reference unit normal.

	Attributes:
	    measure (dolfin.Measure): Boundary measure (typically ``ds``).
	    tv_F (TimeVaryingConstant): Time-varying traction vector :math:`\mathbf{F}`.
	    res_form (UFL form): The resulting residual variational form.


	:param U_test: Test function (virtual displacement).
	:type U_test: dolfin.TestFunction
	:param kinematics: Kinematics object providing the deformation gradient and Jacobian.
	:type kinematics: dmech.Kinematics
	:param N: Unit normal vector in the reference configuration.
	:type N: dolfin.Constant or dolfin.Expression
	:param measure: Dolfin measure for the boundary integration.
	:type measure: dolfin.Measure
	:param F_val: Static traction value.
	:type F_val: list[float] or dolfin.Constant, optional
	:param F_ini: Initial traction value for time-varying loads.
	:type F_ini: list[float] or dolfin.Constant, optional
	:param F_fin: Final traction value for time-varying loads.
	:type F_fin: list[float] or dolfin.Constant, optional
	"""

	def __init__(self, U_test, kinematics, N, measure, F_val=None, F_ini=None, F_fin=None):
		"""Initializes the SurfaceForceLoadingOperator."""
		self.measure = measure

		self.tv_F = TimeVaryingConstant(val=F_val, val_ini=F_ini, val_fin=F_fin)
		F = self.tv_F.val

		FmTN = dolfin.dot(dolfin.inv(kinematics.F).T, N)
		T = dolfin.sqrt(dolfin.inner(FmTN, FmTN)) * F
		self.res_form = -dolfin.inner(T, U_test) * kinematics.J * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates the traction vector for the current time step.

		:param t_step: Current normalized time step (0.0 to 1.0).
		:type t_step: float
		"""
		self.tv_F.set_value_at_t_step(t_step)


################################################################################


class SurfaceForce0(Operator):
	r"""Operator representing an external surface force (traction) in a small strain (linearized) framework.

	In the small strain assumption, the geometry is assumed to be undeformed,
	meaning the current and reference surface areas and normals are identical.

	The residual contribution is:

	.. math::
	    \delta \Pi_{ext} = - \int_{\Gamma} \mathbf{F} \cdot \delta \mathbf{u} \, d\Gamma

	Attributes:
	    measure (dolfin.Measure): Boundary measure (typically ``ds``).
	    tv_F (TimeVaryingConstant): Time-varying traction vector :math:`\mathbf{F}`.
	    res_form (UFL form): The resulting residual variational form.


	:param U_test: Test function (virtual displacement).
	:type U_test: dolfin.TestFunction
	:param measure: Dolfin measure for the boundary integration.
	:type measure: dolfin.Measure
	:param F_val: Static traction value.
	:param F_ini: Initial traction value.
	:param F_fin: Final traction value.
	"""

	def __init__(self, U_test, measure, F_val=None, F_ini=None, F_fin=None):
		"""Initializes the SurfaceForce0LoadingOperator."""
		self.measure = measure

		self.tv_F = TimeVaryingConstant(val=F_val, val_ini=F_ini, val_fin=F_fin)
		F = self.tv_F.val

		self.res_form = -dolfin.inner(F, U_test) * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates the traction vector for the current time step."""
		self.tv_F.set_value_at_t_step(t_step)
