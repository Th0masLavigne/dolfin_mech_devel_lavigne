# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the Wbulk and InverseWbulk operators.

Model the bulk energy
contributions of porous lung tissue by coupling displacement fields with
solid volume fraction evolution in both forward and inverse finite strain
frameworks.
"""

import dolfin

from ..operator import Operator

################################################################################


class Wbulk(Operator):
	r"""Operator representing the bulk energy contribution in a finite strain poromechanical framework.

	This operator couples the deformation of a porous solid (specifically lung
	elastic material) with the evolution of the solid volume fraction :math:`\Phi_s`.
	It assembles the residual contribution arising from the derivative of the
	bulk energy with respect to both displacement and porosity.

	The residual includes two main terms:
	1. The mechanical stress contribution derived from the bulk potential.
	2. The porosity-coupling term for the mixed formulation.

	Attributes:
	    kinematics (Kinematics): Finite strain kinematics (F, J, C_inv, E).
	    solid_material (WbulkLungElasticMaterial): The underlying lung material model.
	    material (PorousElasticMaterial): The porous wrapper handling scaling.
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(
		self, kinematics, U_test, Phis0, Phis, unknown_porosity_test, material_parameters, material_scaling, measure
	):
		"""Initializes the WbulkPoroOperator.

		:param kinematics: Kinematics object for finite strain.
		:type kinematics: dmech.Kinematics
		:param U_test: Test function for the displacement field.
		:type U_test: dolfin.TestFunction
		:param Phis0: Initial solid volume fraction.
		:param Phis: Current solid volume fraction.
		:param unknown_porosity_test: Test function for the porosity unknown.
		:type unknown_porosity_test: dolfin.TestFunction
		:param material_parameters: Dictionary of material constants.
		:param material_scaling: Scaling factor for the porous material.
		:param measure: Dolfin measure for integration.
		"""
		self.kinematics = kinematics
		self.solid_material = dmech.WbulkLungElasticMaterial(Phis=Phis, Phis0=Phis0, parameters=material_parameters)
		self.material = dmech.PorousElasticMaterial(
			solid_material=self.solid_material, scaling=material_scaling, Phis0=Phis0
		)
		self.measure = measure

		dE_test = dolfin.derivative(self.kinematics.E, self.kinematics.U, U_test)
		self.res_form = (
			dolfin.inner(self.material.dWbulkdPhis * self.kinematics.J * self.kinematics.C_inv, dE_test) * self.measure
		)

		self.res_form += self.material.dWbulkdPhis * unknown_porosity_test * self.measure


################################################################################


class InverseWbulk(Operator):
	r"""Operator representing the bulk energy contribution for inverse or linearized poromechanical problems.

	This operator is used when the volume fractions :math:`\phi_s` and :math:`\phi_{s0}`
	are defined in a way that includes the Jacobian :math:`J` (i.e., mapping
	quantities between reference and spatial configurations).

	The residual contribution is:

	.. math::
	    \delta \Pi = \int_{\Omega} \frac{\partial W_{bulk}}{\partial \Phi_s} \text{tr}(\delta \mathbf{\epsilon}) \, d\Omega + \int_{\Omega} \frac{\partial W_{bulk}}{\partial \Phi_s} \delta \phi \, d\Omega

	Attributes:
	    kinematics (Kinematics): Kinematics object.
	    measure (dolfin.Measure): Integration measure.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(
		self, kinematics, u_test, phis, phis0, unknown_porosity_test, material_parameters, material_scaling, measure
	):
		"""Initializes the InverseWbulkPoroOperator.

		:param phis: Spatial solid volume fraction.
		:param phis0: Reference solid volume fraction.
		:param unknown_porosity_test: Test function for the porosity field.
		"""
		self.kinematics = kinematics
		self.solid_material = dmech.WbulkLungElasticMaterial(
			Phis=self.kinematics.J * phis, Phis0=self.kinematics.J * phis0, parameters=material_parameters
		)
		self.material = dmech.PorousElasticMaterial(
			solid_material=self.solid_material, scaling=material_scaling, Phis0=self.kinematics.J * phis0
		)
		self.measure = measure

		epsilon_test = dolfin.sym(dolfin.grad(u_test))
		self.res_form = self.material.dWbulkdPhis * dolfin.tr(epsilon_test) * self.measure

		self.res_form += self.material.dWbulkdPhis * unknown_porosity_test * self.measure
