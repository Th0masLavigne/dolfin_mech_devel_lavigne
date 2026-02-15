# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the DirectionalDisplacement class.

Implements a
variational penalty term to weakly constrain or stabilize displacements along
a prescribed direction, facilitating rigid body motion suppression and
flexible Dirichlet-like boundary conditions.
"""

import dolfin

from ...core import TimeVaryingConstant
from ..operator import Operator

################################################################################


class DirectionalDisplacement(Operator):
	r"""Operator representing a directional displacement penalty term.

	This operator assembles a penalty energy functional that penalizes displacement
	along a specific direction :math:`\mathbf{N}`. It is typically used to
	weakly enforce Dirichlet-like conditions or to stabilize a model against
	rigid body translations in a specific orientation without using hard
	constraints.

	The penalty potential :math:`\Pi_{pen}` is defined as:

	.. math::
	    \Pi_{pen} = \int_{\Omega} \frac{k_{pen}}{2} (\mathbf{u} \cdot \mathbf{N})^2 \, d\Omega

	The resulting residual form is the first variation of this potential with
	respect to the displacement :math:`\mathbf{u}`:

	.. math::
	    \delta \Pi_{pen} = \int_{\Omega} k_{pen} (\mathbf{u} \cdot \mathbf{N}) (\delta \mathbf{u} \cdot \mathbf{N}) \, d\Omega

	Attributes:
	    measure (dolfin.Measure): The integration measure (typically ``dx`` or ``ds``).
	    tv_N (TimeVaryingConstant): Time-varying unit vector :math:`\mathbf{N}`
	        defining the penalty direction.
	    tv_pen (TimeVaryingConstant): Time-varying penalty stiffness :math:`k_{pen}`.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(
		self, U, U_test, measure, N_val=None, N_ini=None, N_fin=None, pen_val=None, pen_ini=None, pen_fin=None
	):
		"""Initializes the DirectionalDisplacementPenaltyOperator.

		:param U: Displacement field.
		:type U: dolfin.Function
		:param U_test: Test function (virtual displacement).
		:type U_test: dolfin.TestFunction
		:param measure: Dolfin measure for integration (domain or boundary).
		:type measure: dolfin.Measure
		:param N_val: Static penalty direction vector.
		:type N_val: list[float] or dolfin.Constant, optional
		:param N_ini: Initial penalty direction vector.
		:type N_ini: list[float] or dolfin.Constant, optional
		:param N_fin: Final penalty direction vector.
		:type N_fin: list[float] or dolfin.Constant, optional
		:param pen_val: Static penalty stiffness value.
		:type pen_val: float, optional
		:param pen_ini: Initial penalty stiffness value.
		:type pen_ini: float, optional
		:param pen_fin: Final penalty stiffness value.
		:type pen_fin: float, optional
		"""
		self.measure = measure

		self.tv_N = TimeVaryingConstant(val=N_val, val_ini=N_ini, val_fin=N_fin)
		N = self.tv_N.val

		self.tv_pen = TimeVaryingConstant(val=pen_val, val_ini=pen_ini, val_fin=pen_fin)
		pen = self.tv_pen.val

		Pi = (pen / 2) * dolfin.inner(U, N) ** 2 * self.measure
		self.res_form = dolfin.derivative(Pi, U, U_test)

		# self.res_form = pen * dolfin.inner(U     , N)\
		#                     * dolfin.inner(U_test, N) * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates the penalty direction and stiffness for the current time step.

		:param t_step: Current normalized time step (0.0 to 1.0).
		:type t_step: float
		"""
		self.tv_N.set_value_at_t_step(t_step)
		self.tv_pen.set_value_at_t_step(t_step)
