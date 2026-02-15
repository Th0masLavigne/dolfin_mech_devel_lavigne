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

"""Defines the MacroscopicStretchComponent class.

Weakly
enforces prescribed macroscopic displacement gradient components using a
penalty method, facilitating strain-controlled homogenization of
Representative Elementary Volumes (REVs).
"""

import dolfin

from ...core import TimeVaryingConstant
from ..operator import Operator

################################################################################


class MacroscopicStretchComponent(Operator):
	r"""Operator enforcing a component of the macroscopic displacement gradient tensor via a penalty method.

	In multiscale modeling, the macroscopic deformation is often described by
	a displacement gradient tensor :math:`\bar{\mathbf{U}}`. This operator
	penalizes the difference between a specific component :math:`\bar{U}_{ij}`
	and a target value :math:`\bar{U}_{ij}^{target}`.

	The penalty potential :math:`\Pi` is defined as:

	.. math::
	    \Pi = \int_{\Omega} \frac{k_{pen}}{2} (\bar{U}_{ij} - \bar{U}_{ij}^{target})^2 \, d\Omega

	The residual form is obtained by taking the first variation of this potential
	with respect to the macroscopic component.



	Attributes:
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    tv_U_bar_ij (TimeVaryingConstant): Time-varying target value for the
	        macroscopic component :math:`\bar{U}_{ij}`.
	    tv_pen (TimeVaryingConstant): Time-varying penalty stiffness :math:`k_{pen}`.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(
		self,
		U_bar,
		U_bar_test,
		i,
		j,
		measure,
		U_bar_ij_val=None,
		U_bar_ij_ini=None,
		U_bar_ij_fin=None,
		pen_val=None,
		pen_ini=None,
		pen_fin=None,
	):
		"""Initializes the MacroscopicStretchComponentPenaltyOperator.

		:param U_bar: Macroscopic displacement gradient tensor.
		:type U_bar: dolfin.Coefficient
		:param U_bar_test: Test function associated with the macroscopic tensor.
		:type U_bar_test: dolfin.Argument
		:param i: Row index of the component to penalize.
		:type i: int
		:param j: Column index of the component to penalize.
		:type j: int
		:param measure: Dolfin measure for integration.
		:type measure: dolfin.Measure
		:param U_bar_ij_val: Static target value for the component.
		:param U_bar_ij_ini: Initial target value for time-varying loading.
		:param U_bar_ij_fin: Final target value for time-varying loading.
		:param pen_val: Static penalty stiffness.
		:param pen_ini: Initial penalty stiffness for time-varying regularization.
		:param pen_fin: Final penalty stiffness for time-varying regularization.
		"""
		self.measure = measure

		self.tv_U_bar_ij = TimeVaryingConstant(val=U_bar_ij_val, val_ini=U_bar_ij_ini, val_fin=U_bar_ij_fin)
		U_bar_ij = self.tv_U_bar_ij.val

		self.tv_pen = TimeVaryingConstant(val=pen_val, val_ini=pen_ini, val_fin=pen_fin)
		pen = self.tv_pen.val

		Pi = (pen / 2) * (U_bar[i, j] - U_bar_ij) ** 2 * self.measure
		self.res_form = dolfin.derivative(Pi, U_bar[i, j], U_bar_test[i, j])

	def set_value_at_t_step(self, t_step):
		"""Updates the target stretch component and penalty stiffness for the current time step.

		:param t_step: Current normalized time (0.0 to 1.0).
		:type t_step: float
		"""
		self.tv_U_bar_ij.set_value_at_t_step(t_step)
		self.tv_pen.set_value_at_t_step(t_step)
