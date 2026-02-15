# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the Wskel and InverseWskel operators.

Model the internal virtual
work of the solid lung skeleton (fiber network) by coupling structural
elasticity with porosity-based material scaling in both finite and small
strain frameworks.
"""

import dolfin

from ...materials import elastic
from ..operator import Operator

################################################################################


class Wskel(Operator):
	r"""Operator representing the solid skeleton energy contribution in a finite strain poromechanical framework.

	This operator models the mechanical response of the solid skeleton
	(typically the elastin and collagen fiber network in lung tissue). It
	assembles the internal virtual work using the Second Piola-Kirchhoff
	stress tensor :math:`\mathbf{\Sigma}` provided by a scaled porous material
	model.

	The internal virtual work :math:`\delta \Pi_{int}` is:

	.. math::
	    \delta \Pi_{int} = \int_{\Omega_0} \mathbf{\Sigma} : \delta \mathbf{E} \, d\Omega_0

	where :math:`\mathbf{\Sigma}` is the Second Piola-Kirchhoff stress and
	:math:`\delta \mathbf{E}` is the virtual Green-Lagrange strain.

	Attributes:
	    kinematics (Kinematics): Finite strain kinematics object.
	    solid_material (WskelLungElasticMaterial): Underlying lung skeleton material.
	    material (PorousElasticMaterial): Porous wrapper handling material scaling
	        based on solid volume fraction.
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(self, kinematics, U_test, material_parameters, material_scaling, Phis0, measure):
		"""Initializes the WskelPoroOperator.

		:param kinematics: Kinematics object providing strain tensors.
		:type kinematics: dmech.Kinematics
		:param U_test: Test function (virtual displacement).
		:type U_test: dolfin.TestFunction
		:param material_parameters: Dictionary of material constants.
		:type material_parameters: dict
		:param material_scaling: Scaling factor or strategy for porosity.
		:type material_scaling: str or float
		:param Phis0: Initial solid volume fraction.
		:type Phis0: float or dolfin.Constant
		:param measure: Dolfin measure for domain integration.
		:type measure: dolfin.Measure
		"""
		self.kinematics = kinematics
		self.solid_material = elastic.WskelLung(kinematics=kinematics, parameters=material_parameters)
		self.material = elastic.Porous(solid_material=self.solid_material, scaling=material_scaling, Phis0=Phis0)
		self.measure = measure

		dE_test = dolfin.derivative(self.kinematics.E, self.kinematics.U, U_test)
		self.res_form = dolfin.inner(self.material.Sigma, dE_test) * self.measure


################################################################################


class InverseWskel(Operator):
	r"""Operator representing the solid skeleton energy contribution in a small strain or inverse poromechanical framework.

	This version is used when the configuration is assumed to be linearized or
	when volume fractions :math:`\phi_{s0}` are provided as spatial-like
	quantities requiring Jacobian scaling. It uses the Cauchy stress
	tensor :math:`\mathbf{\sigma}`.

	The internal virtual work :math:`\delta \Pi_{int}` is:

	.. math::
	    \delta \Pi_{int} = \int_{\Omega} \mathbf{\sigma} : \delta \mathbf{\epsilon} \, d\Omega

	where :math:`\delta \mathbf{\epsilon}` is the virtual infinitesimal strain.

	Attributes:
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(self, kinematics, u_test, material_parameters, material_scaling, phis0, measure):
		"""Initializes the InverseWskelPoroOperator.

		:param kinematics: Kinematics object.
		:param u_test: Test function.
		:param material_parameters: Material constants.
		:param material_scaling: Porosity scaling factor.
		:param phis0: Spatial-like reference solid volume fraction.
		:param measure: Integration measure.
		"""
		self.kinematics = kinematics
		self.solid_material = elastic.WskelLung(kinematics=kinematics, parameters=material_parameters)
		self.material = elastic.Porous(
			solid_material=self.solid_material, scaling=material_scaling, Phis0=self.kinematics.J * phis0
		)
		self.measure = measure

		epsilon_test = dolfin.sym(dolfin.grad(u_test))
		self.res_form = dolfin.inner(self.material.sigma, epsilon_test) * self.measure
