# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2020-2024                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################
"""Defines the LagrangeMultiplierComponent class.

Applies
a variational penalty to specific components of a Lagrange multiplier tensor
to regularize the system or weakly enforce zero-value constraints.
"""

from ...core import TimeVaryingConstant
from ..operator import Operator

################################################################################


class LagrangeMultiplierComponent(Operator):
	r"""Operator representing a penalty term applied to a specific component of a Lagrange multiplier tensor.

	In mixed finite element methods, Lagrange multipliers are used to enforce
	constraints. This operator adds a penalty term to the variational form
	to regularize the solution or to weakly constrain a specific component
	:math:`\lambda_{ij}` to zero.

	The penalty contribution to the residual is defined as:

	.. math::
	    \mathcal{R}_{pen} = \int_{\Omega} k_{pen} \lambda_{ij} \delta \lambda_{ij} \, d\Omega

	where:
	    - :math:`k_{pen}` is the penalty stiffness coefficient.
	    - :math:`\lambda_{ij}` is the :math:`(i,j)` component of the Lagrange multiplier.
	    - :math:`\delta \lambda_{ij}` is the corresponding test function component.

	Attributes:
	    measure (dolfin.Measure): The integration measure (typically ``dx``).
	    tv_pen (TimeVaryingConstant): Time-varying penalty stiffness :math:`k_{pen}`.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(self, lambda_bar, lambda_bar_test, i, j, measure, pen_val=None, pen_ini=None, pen_fin=None):
		"""Initializes the LagrangeMultiplierComponentPenaltyOperator.

		:param lambda_bar: The Lagrange multiplier tensor field.
		:type lambda_bar: dolfin.Function or ufl.Coefficient
		:param lambda_bar_test: The test function associated with the Lagrange multiplier.
		:type lambda_bar_test: dolfin.TestFunction
		:param i: Row index of the component to be penalized.
		:type i: int
		:param j: Column index of the component to be penalized.
		:type j: int
		:param measure: Dolfin measure for integration.
		:type measure: dolfin.Measure
		:param pen_val: Static penalty stiffness value.
		:type pen_val: float, optional
		:param pen_ini: Initial penalty stiffness value for time-ramping.
		:type pen_ini: float, optional
		:param pen_fin: Final penalty stiffness value for time-ramping.
		:type pen_fin: float, optional
		"""
		self.measure = measure

		self.tv_pen = TimeVaryingConstant(val=pen_val, val_ini=pen_ini, val_fin=pen_fin)
		pen = self.tv_pen.val

		self.res_form = pen * lambda_bar[i, j] * lambda_bar_test[i, j] * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates the penalty stiffness for the current time step.

		:param t_step: Current normalized time step (0.0 to 1.0).
		:type t_step: float
		"""
		self.tv_pen.set_value_at_t_step(t_step)
