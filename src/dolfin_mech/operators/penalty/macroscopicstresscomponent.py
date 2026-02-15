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

"""Defines the MacroscopicStressComponent class.

Employs a penalty method
to weakly prescribe macroscopic stress components during multiscale
homogenization, allowing for stress-controlled Representative Elementary
Volume (REV) simulations.
"""

import dolfin

from ...core import TimeVaryingConstant
from ..operator import Operator

################################################################################


class MacroscopicStressComponent(Operator):
	r"""Operator to enforce a component of the macroscopic stress tensor via a penalty method.

	In a multiscale homogenization framework, this operator penalizes the
	deviation of a specific microscopic stress component from a prescribed
	macroscopic value. This is a "weak" alternative to using Lagrange
	multipliers for stress-controlled Representative Elementary Volume (REV)
	simulations.

	The penalty potential :math:`\Pi` is defined as:

	.. math::
	    \Pi = \int_{\Omega} \frac{k_{pen}}{2} (\sigma_{ij} - \bar{\sigma}_{ij})^2 d\Omega

	The residual form is obtained by taking the derivative of this potential
	with respect to the solution variables.



	Attributes:
	    material (Material): The material model providing the Cauchy stress tensor.
	    measure (dolfin.Measure): The integration measure (typically ``dx``).
	    tv_comp (TimeVaryingConstant): Time-varying target stress component :math:`\bar{\sigma}_{ij}`.
	    tv_pen (TimeVaryingConstant): Time-varying penalty stiffness :math:`k_{pen}`.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(
		self,
		sigma_bar,
		sigma_bar_test,
		sol,
		sol_test,
		material,
		comp_i,
		comp_j,
		measure,
		comp_val=None,
		comp_ini=None,
		comp_fin=None,
		pen_val=None,
		pen_ini=None,
		pen_fin=None,
	):
		"""Initializes the MacroscopicStressComponentPenaltyOperator.

		:param sigma_bar: Macroscopic stress tensor.
		:param sigma_bar_test: Test function for macroscopic stress.
		:param sol: The solution Function (e.g., displacement or macroscopic gradient).
		:type sol: dolfin.Function
		:param sol_test: The test Function corresponding to the solution.
		:type sol_test: dolfin.Argument
		:param material: The microscopic material model.
		:type material: dmech.Material
		:param comp_i: Row index of the stress component to penalize.
		:type comp_i: int
		:param comp_j: Column index of the stress component to penalize.
		:type comp_j: int
		:param measure: Integration measure.
		:type measure: dolfin.Measure
		:param comp_val: Static target stress value.
		:param comp_ini: Initial target stress value for time-varying loads.
		:param comp_fin: Final target stress value for time-varying loads.
		:param pen_val: Static penalty stiffness.
		:param pen_ini: Initial penalty stiffness.
		:param pen_fin: Final penalty stiffness.
		"""
		self.material = material
		self.measure = measure

		self.tv_comp = TimeVaryingConstant(val=comp_val, val_ini=comp_ini, val_fin=comp_fin)
		comp = self.tv_comp.val

		self.tv_pen = TimeVaryingConstant(val=pen_val, val_ini=pen_ini, val_fin=pen_fin)
		pen = self.tv_pen.val

		Pi = (
			(pen / 2) * (self.material.sigma[comp_i, comp_j] - comp) ** 2 * self.measure
		)  # MG20220426: Need to compute <sigma> properly, including fluid pressure
		# self.res_form = dolfin.derivative(Pi, sigma_bar[comp_i,comp_j], sigma_bar_test[comp_i,comp_j]) # MG20230106: This does not work…
		self.res_form = dolfin.derivative(Pi, sol, sol_test)  # MG20230106: This works…

		# Pi = (pen/2) * (sigma_bar[comp_i,comp_j] - comp)**2 * self.measure # MG20230106: This does not work…
		# self.res_form = dolfin.derivative(Pi, sigma_bar[comp_i,comp_j], sigma_bar_test[comp_i,comp_j])
		# self.res_form = dolfin.derivative(Pi, sol, sol_test)

	def set_value_at_t_step(self, t_step):
		"""Updates the target stress and penalty stiffness for the current time step.

		:param t_step: Current normalized time (0.0 to 1.0).
		:type t_step: float
		"""
		self.tv_comp.set_value_at_t_step(t_step)
		self.tv_pen.set_value_at_t_step(t_step)
