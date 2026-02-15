# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the MacroscopicStretchSymmetry class.

Weakly
enforces the symmetry of the macroscopic displacement gradient tensor using
a penalty method, effectively suppressing unconstrained macroscopic rotations
in multiscale homogenization.
"""

import dolfin

from ...core import TimeVaryingConstant
from ..operator import Operator

################################################################################


class MacroscopicStretchSymmetry(Operator):
	r"""Operator enforcing the symmetry of the macroscopic displacement gradient tensor via a penalty method.

	In homogenization theory, the macroscopic deformation gradient is often
	decomposed into a stretch and a rotation. This operator penalizes the
	skew-symmetric part of the macroscopic displacement gradient
	:math:`\bar{\mathbf{U}}`, effectively weakly enforcing
	:math:`\bar{\mathbf{U}} = \bar{\mathbf{U}}^T`.

	The penalty potential :math:`\Pi` is defined as:

	.. math::
	    \Pi = \int_{\Omega} \frac{k_{pen}}{2} \|\mathbf{\bar{U}}^T - \mathbf{\bar{U}}\|^2 \, d\Omega

	The residual form is obtained by taking the first variation of this potential
	with respect to the solution variables :math:`\mathbf{sol}`.



	Attributes:
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    tv_pen (TimeVaryingConstant): Time-varying penalty stiffness :math:`k_{pen}`.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(self, U_bar, sol, sol_test, measure, pen_val=None, pen_ini=None, pen_fin=None):
		"""Initializes the MacroscopicStretchSymmetryPenaltyOperator.

		:param U_bar: Macroscopic displacement gradient tensor.
		:type U_bar: dolfin.Coefficient
		:param sol: The solution Function containing the macroscopic variables.
		:type sol: dolfin.Function
		:param sol_test: The test Function associated with the solution.
		:type sol_test: dolfin.Argument
		:param measure: Dolfin measure for integration.
		:type measure: dolfin.Measure
		:param pen_val: Static penalty stiffness value.
		:type pen_val: float, optional
		:param pen_ini: Initial penalty stiffness for time-varying loading.
		:type pen_ini: float, optional
		:param pen_fin: Final penalty stiffness for time-varying loading.
		:type pen_fin: float, optional
		"""
		self.measure = measure

		self.tv_pen = TimeVaryingConstant(val=pen_val, val_ini=pen_ini, val_fin=pen_fin)
		pen = self.tv_pen.val

		Pi = (pen / 2) * dolfin.inner(U_bar.T - U_bar, U_bar.T - U_bar) * self.measure
		# self.res_form = dolfin.derivative(Pi, U_bar, U_bar_test) # MG20230106: Somehow this does not work… NotImplementedError("Cannot take length of non-vector expression.")
		self.res_form = dolfin.derivative(Pi, sol, sol_test)

	def set_value_at_t_step(self, t_step):
		"""Updates the penalty stiffness for the current time step.

		:param t_step: Current normalized time (0.0 to 1.0).
		:type t_step: float
		"""
		self.tv_pen.set_value_at_t_step(t_step)
