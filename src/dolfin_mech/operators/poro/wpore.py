# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the Wpore and InverseWpore operators.

Implement the variational
contribution of pore-level energy for porous lung tissue, coupling fluid
volume fraction changes with the overall poromechanical response in both
forward and inverse finite strain frameworks.
"""

from ...materials import elastic
from ..operator import Operator

################################################################################


class Wpore(Operator):
	r"""Operator representing the pore-level energy contribution in a finite strain poromechanical framework.

	This operator models the energy associated with the fluid-filled pores
	(specifically for lung tissue) and its coupling with the solid volume
	fraction :math:`\Phi_s`. It assembles the residual term derived from the
	variation of the pore energy :math:`W_{pore}` with respect to the porosity.

	The fluid volume fraction is calculated as:

	.. math::
	    \Phi_f = J - \Phi_s

	Attributes:
	    kinematics (Kinematics): Finite strain kinematics object.
	    solid_material (WporeLungElasticMaterial): Underlying pore-level lung material.
	    material (PorousElasticMaterial): Porous wrapper handling material scaling.
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(self, kinematics, Phis0, Phis, unknown_porosity_test, material_parameters, material_scaling, measure):
		"""Initializes the WporePoroOperator.

		:param kinematics: Kinematics object providing the Jacobian :math:`J`.
		:type kinematics: dmech.Kinematics
		:param Phis0: Initial solid volume fraction.
		:param Phis: Current solid volume fraction.
		:param unknown_porosity_test: Test function for the porosity unknown.
		:type unknown_porosity_test: dolfin.TestFunction
		:param material_parameters: Dictionary of material constants.
		:param material_scaling: Scaling factor for the porous material.
		:param measure: Dolfin measure for domain integration.
		"""
		self.kinematics = kinematics
		self.solid_material = elastic.WporeLung(
			Phif=self.kinematics.J - Phis, Phif0=1.0 - Phis0, parameters=material_parameters
		)
		self.material = elastic.Porous(solid_material=self.solid_material, scaling=material_scaling, Phis0=Phis0)
		self.measure = measure

		self.res_form = -self.material.dWporedPhif * unknown_porosity_test * self.measure


################################################################################


class InverseWpore(Operator):
	r"""Operator representing the pore-level energy contribution for inverse or linearized poromechanical problems.

	In this operator, the volume fractions :math:`\phi_s` and :math:`\phi_0`
	are assumed to be spatial-like quantities that require scaling by the
	Jacobian :math:`J` to recover reference volume fractions.

	The fluid fraction is mapped as:

	.. math::
	    \Phi_f = J \cdot (1 - \phi_s)

	Attributes:
	    kinematics (Kinematics): Kinematics object.
	    measure (dolfin.Measure): Integration measure.
	    res_form (UFL form): The resulting residual variational form.
	"""

	def __init__(self, kinematics, phis, phis0, unknown_porosity_test, material_parameters, material_scaling, measure):
		"""Initializes the InverseWporePoroOperator.

		:param phis: Spatial solid volume fraction.
		:param phis0: Reference solid volume fraction.
		:param unknown_porosity_test: Test function for the porosity field.
		"""
		self.kinematics = kinematics
		self.solid_material = elastic.WporeLung(
			Phif=self.kinematics.J * (1 - phis), Phif0=1 - self.kinematics.J * phis0, parameters=material_parameters
		)
		self.material = elastic.Porous(
			solid_material=self.solid_material, scaling=material_scaling, Phis0=self.kinematics.J * phis0
		)
		self.measure = measure

		self.res_form = -self.material.dWporedPhif * unknown_porosity_test * self.measure
