# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the Pf class.

Implements the energetic contribution
of fluid pressure to the porosity evolution equation within a coupled
poromechanical variational framework.
"""

import dolfin

from ...core import TimeVaryingConstant
from ..operator import Operator

################################################################################


class Pf(Operator):
	r"""Operator representing the fluid pressure contribution in a poromechanical formulation.

	This operator assembles the residual term related to the fluid pressure
	:math:`p_f` acting on the porosity field. In mixed formulations for
	poroelasticity or poroplasticity, this term often appears in the mass
	conservation or the porosity evolution equations.

	The residual contribution is defined as:

	.. math::
	    \mathcal{R} = \int_{\Omega} p_f \cdot \delta \phi \, d\Omega

	where:
	    - :math:`p_f` is the fluid pressure (prescribed or time-varying).
	    - :math:`\delta \phi` is the test function associated with the
	      porosity unknown.

	Attributes:
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    tv_pf (TimeVaryingConstant): Time-varying fluid pressure :math:`p_f`.
	    pf (dolfin.Constant): The current value of the fluid pressure.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(self, unknown_porosity_test, measure, pf_val=None, pf_ini=None, pf_fin=None):
		"""Initializes the PfPoroOperator.

		:param unknown_porosity_test: Test function for the porosity unknown.
		:type unknown_porosity_test: dolfin.TestFunction
		:param measure: Dolfin measure for domain integration.
		:type measure: dolfin.Measure
		:param pf_val: Static value for fluid pressure.
		:type pf_val: float, optional
		:param pf_ini: Initial value for time-varying fluid pressure.
		:type pf_ini: float, optional
		:param pf_fin: Final value for time-varying fluid pressure.
		:type pf_fin: float, optional
		"""
		self.measure = measure

		self.tv_pf = TimeVaryingConstant(val=pf_val, val_ini=pf_ini, val_fin=pf_fin)
		self.pf = self.tv_pf.val

		self.res_form = dolfin.inner(self.pf, unknown_porosity_test) * self.measure

	def set_value_at_t_step(self, t_step):
		"""Updates the time-varying fluid pressure for the current time step.

		:param t_step: Current normalized time step (0.0 to 1.0).
		:type t_step: float
		"""
		self.tv_pf.set_value_at_t_step(t_step)
