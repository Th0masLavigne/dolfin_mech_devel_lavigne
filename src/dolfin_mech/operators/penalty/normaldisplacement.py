# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the NormalDisplacement class.

Implements a
variational penalty to weakly enforce zero-displacement constraints in a
normal direction, facilitating sliding or symmetry boundary conditions
without hard Dirichlet constraints.
"""

import dolfin

from ...core import TimeVaryingConstant
from ..operator import Operator

################################################################################


class NormalDisplacement(Operator):
	r"""Operator enforcing a penalty on the displacement component normal to a given vector (typically the surface normal).

	This operator assembles a penalty potential that discourages movement
	along the direction defined by :math:`\mathbf{N}`. It is commonly used
	to weakly enforce sliding boundary conditions (symmetry conditions)
	on curved or flat boundaries.

	The penalty energy functional :math:`\Pi_{pen}` is defined as:

	.. math::
	    \Pi_{pen} = \int_{\Gamma} \frac{k_{pen}}{2} (\mathbf{u} \cdot \mathbf{N})^2 \, d\Gamma

	The resulting residual form is the directional derivative of this potential:

	.. math::
	    \delta \Pi_{pen} = \int_{\Gamma} k_{pen} (\mathbf{u} \cdot \mathbf{N}) (\delta \mathbf{u} \cdot \mathbf{N}) \, d\Gamma

	Attributes:
	    measure (dolfin.Measure): The integration measure (typically ``ds`` for boundaries).
	    tv_pen (TimeVaryingConstant): Time-varying penalty stiffness :math:`k_{pen}`.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(self, U, U_test, N, measure, pen_val=None, pen_ini=None, pen_fin=None):
		"""Initializes the NormalDisplacementPenaltyOperator.

		:param U: Current displacement field.
		:type U: dolfin.Function
		:param U_test: Test function (virtual displacement).
		:type U_test: dolfin.TestFunction
		:param N: Vector defining the normal direction (e.g., boundary normal).
		:type N: dolfin.Constant or dolfin.FacetNormal
		:param measure: Dolfin measure for integration.
		:type measure: dolfin.Measure
		:param pen_val: Static penalty stiffness value.
		:type pen_val: float, optional
		:param pen_ini: Initial penalty stiffness for time-varying loads.
		:type pen_ini: float, optional
		:param pen_fin: Final penalty stiffness for time-varying loads.
		:type pen_fin: float, optional
		"""
		self.measure = measure

		self.tv_pen = TimeVaryingConstant(val=pen_val, val_ini=pen_ini, val_fin=pen_fin)
		pen = self.tv_pen.val

		Pi = (pen / 2) * dolfin.inner(U, N) ** 2 * self.measure
		self.res_form = dolfin.derivative(Pi, U, U_test)

		# self.res_form = pen * dolfin.inner(U     , N)\
		#                     * dolfin.inner(U_test, N) * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates the penalty stiffness coefficient for the current time step.

		:param t_step: Current normalized time step (0.0 to 1.0).
		:type t_step: float
		"""
		self.tv_pen.set_value_at_t_step(t_step)
