# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the HyperElasticity class.

Implements the internal virtual work
for nonlinear solid mechanics using three energy-consistent formulations:
energy-derivative, Second Piola-Kirchhoff, or First Piola-Kirchhoff.
"""

import dolfin

from ..materials import material_factory
from .operator import Operator

################################################################################


class HyperElasticity(Operator):
	r"""Operator representing the internal virtual work for hyperelastic materials.

	This class assembles the residual variational form for a hyperelastic body.
	It supports three different mathematical formulations for the internal
	virtual work, all of which are equivalent but lead to different UFL
	expressions.

	The supported formulations are:

	- **Total Potential Energy (ener):**
	    The residual is the first variation (Fréchet derivative) of the
	    strain energy density :math:`\Psi` with respect to the displacement :math:`\mathbf{u}`.

	    .. math::
	        \delta \Pi_{int} = \int_{\Omega_0} \frac{\partial \Psi}{\partial \mathbf{u}} \cdot \delta \mathbf{u} \, d\Omega_0

	- **Second Piola-Kirchhoff (PK2):**
	    The work is defined using the Second Piola-Kirchhoff stress :math:`\mathbf{\Sigma}`
	    and the variation of the Green-Lagrange strain tensor :math:`\mathbf{E}`.

	    .. math::
	        \delta \Pi_{int} = \int_{\Omega_0} \mathbf{\Sigma} : \delta \mathbf{E} \, d\Omega_0

	- **First Piola-Kirchhoff (PK1):**
	    The work is defined using the First Piola-Kirchhoff stress :math:`\mathbf{P}`
	    and the variation of the deformation gradient :math:`\mathbf{F}`.

	    .. math::
	        \delta \Pi_{int} = \int_{\Omega_0} \mathbf{P} : \delta \mathbf{F} \, d\Omega_0

	Args:
			U: The displacement trial function (or current solution).
			U_test: The test function (virtual displacement).
			kinematics: Kinematics object providing F and E.
			material_model: Name of the material model (e.g., "NeoHookean").
			material_parameters: Dictionary of material properties.
			measure: Dolfin measure for domain integration.
			formulation: Selection of the stress/strain pair ("PK1", "PK2", or "ener").

	Raises:
			ValueError: If an invalid formulation name is provided.

	Attributes:
	    kinematics (Kinematics): Kinematic variables associated with the problem.
	    material (Material): Material model instance created via the factory.
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(
		self, U, U_test, kinematics, material_model, material_parameters, measure, formulation="PK1"
	):  # PK1 or PK2 or ener
		"""Initializes the HyperElasticityOperator."""
		self.kinematics = kinematics
		self.material = material_factory(kinematics, material_model, material_parameters)
		self.measure = measure

		assert formulation in ("PK1", "PK2", "ener"), '"formulation" should be "PK1", "PK2" or "ener". Aborting.'

		if formulation == "ener":
			self.res_form = dolfin.derivative(self.material.Psi, U, U_test) * self.measure
		elif formulation == "PK2":
			dE_test = dolfin.derivative(self.kinematics.E, U, U_test)
			self.res_form = dolfin.inner(self.material.Sigma, dE_test) * self.measure
		elif formulation == "PK1":
			dF_test = dolfin.derivative(self.kinematics.F, U, U_test)
			self.res_form = dolfin.inner(self.material.P, dF_test) * self.measure
