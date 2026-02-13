# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Composite Neo-Hookean/Mooney-Rivlin hyperelastic material.

This module provides a combined material model that sums the strain energy
densities of the Neo-Hookean and Mooney-Rivlin models.
"""

import dolfin_mech.materials as materials

from .Elastic import ElasticMaterial

################################################################################


class NeoHookeanMooneyRivlin(ElasticMaterial):
	r"""Class representing a composite Neo-Hookean/Mooney-Rivlin hyperelastic material.

	This model combines the strain energy densities of both the Neo-Hookean and
	Mooney-Rivlin models. By summing these potentials, the material response
	can be tuned using both the first invariant :math:`I_C` (via Neo-Hookean)
	and the second invariant :math:`II_C` (via Mooney-Rivlin).

	The total strain energy density is defined as:

	.. math::
	    \Psi_{total} = \Psi_{NH}(C_1) + \Psi_{MR}(C_2)

	Where:
	    - :math:`\Psi_{NH}` is the energy from :class:`NeoHookeanElasticMaterial`.
	    - :math:`\Psi_{MR}` is the energy from :class:`MooneyRivlinElasticMaterial`.

	Attributes:
	    kinematics (Kinematics): Object containing kinematic variables and invariants.
	    nh (NeoHookeanElasticMaterial): The Neo-Hookean component of the model.
	    mr (MooneyRivlinElasticMaterial): The Mooney-Rivlin component of the model.
	    Psi (UFL expression): Total strain energy density.
	    Sigma (UFL expression): Total Second Piola-Kirchhoff stress tensor.
	    P (UFL expression): Total First Piola-Kirchhoff stress tensor.
	    sigma (UFL expression): Total Cauchy stress tensor.
	"""

	def __init__(self, kinematics, parameters, decoup=False):
		"""Initializes the NeoHookeanMooneyRivlinElasticMaterial.

		:param kinematics: Kinematics object containing deformation tensors.
		:type kinematics: dmech.Kinematics
		:param parameters: Dictionary containing parameters to derive :math:`C_1` and :math:`C_2`.
		:type parameters: dict
		:param decoup: If True, uses the isochoric-volumetric decoupled formulation for both sub-models.
		:type decoup: bool, optional
		"""
		self.kinematics = kinematics

		C1, C2 = self.get_C1_and_C2_from_parameters(
			parameters
		)  # MG20220318: This is different from computing C1 & C2 separately…
		parameters["C1"] = C1
		parameters["C2"] = C2

		self.nh = materials.elastic.NeoHookean(kinematics, parameters, decoup)
		self.mr = materials.elastic.MooneyRivlin(kinematics, parameters, decoup)

		self.Psi = self.nh.Psi + self.mr.Psi
		self.Sigma = self.nh.Sigma + self.mr.Sigma
		if self.kinematics.dim == 2:
			self.Sigma_ZZ = self.nh.Sigma_ZZ + self.mr.Sigma_ZZ
		self.P = self.nh.P + self.mr.P
		self.sigma = self.nh.sigma + self.mr.sigma

	# def get_free_energy(self, *args, **kwargs):

	#     Psi_nh, Sigma_nh = self.nh.get_free_energy(*args, **kwargs)
	#     Psi_mr, Sigma_mr = self.mr.get_free_energy(*args, **kwargs)

	#     Psi   = Psi_nh   + Psi_mr
	#     Sigma = Sigma_nh + Sigma_mr

	#     return Psi, Sigma
