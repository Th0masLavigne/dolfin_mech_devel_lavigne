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

"""Composite Ciarlet-Geymonat and Neo-Hookean material model.

This module implements a hyperelastic model that additively decomposes
the strain energy density into volumetric (bulk) and shear (deviatoric) parts.
"""

import dolfin

import dolfin_mech as dmech

from .Elastic import ElasticMaterial

################################################################################


class OgdenCiarletGeymonatNeoHookean(ElasticMaterial):
	r"""Class representing a composite hyperelastic material combining the
	Ciarlet-Geymonat and Neo-Hookean models.

	This model provides a robust description of compressible hyperelasticity by
	additive decomposition of the strain energy density:

	.. math::
	    \Psi_{total} = \Psi_{bulk}(J) + \Psi_{dev}(\mathbf{C})

	Where:
	    - :math:`\Psi_{bulk}` is the Ciarlet-Geymonat potential (penalizing volume change).
	    - :math:`\Psi_{dev}` is the Neo-Hookean potential (governing shear response).

	Beyond standard stress tensors, this class computes the hydrostatic pressure
	and the Von Mises equivalent of the Second Piola-Kirchhoff stress.

	Attributes:
	    kinematics (Kinematics): Kinematic variables and deformation tensors.
	    bulk (OgdenCiarletGeymonatElasticMaterial): The volumetric component.
	    dev (NeoHookeanElasticMaterial): The deviatoric/shear component.
	    Psi (UFL expression): Total strain energy density.
	    Sigma (UFL expression): Total Second Piola-Kirchhoff stress tensor.
	    p_hydro (UFL expression): Hydrostatic pressure calculated as :math:`p = -\frac{1}{3J} tr(\mathbf{\Sigma} \mathbf{C})`.
	    Sigma_dev (UFL expression): Deviatoric part of the Second Piola-Kirchhoff stress.
	    Sigma_VM (UFL expression): Von Mises equivalent stress.
	"""

	def __init__(self, kinematics, parameters, decoup=False):
		"""Initializes the OgdenCiarletGeymonatNeoHookeanElasticMaterial.

		:param kinematics: Kinematics object containing deformation tensors.
		:type kinematics: dmech.Kinematics
		:param parameters: Dictionary of material parameters (must satisfy both sub-models).
		:type parameters: dict
		:param decoup: If True, uses the isochoric-volumetric decoupled formulation.
		:type decoup: bool, optional
		"""
		self.kinematics = kinematics

		self.bulk = dmech.OgdenCiarletGeymonat(kinematics, parameters, decoup)
		self.dev = dmech.NeoHookean(kinematics, parameters, decoup)

		self.Psi = self.bulk.Psi + self.dev.Psi
		self.Sigma = self.bulk.Sigma + self.dev.Sigma
		if self.kinematics.dim == 2:
			self.Sigma_ZZ = self.bulk.Sigma_ZZ + self.dev.Sigma_ZZ
			self.p_hydro = -(dolfin.tr(self.Sigma.T * self.kinematics.C) + self.Sigma_ZZ) / 3 / self.kinematics.J
		elif self.kinematics.dim == 3:
			self.p_hydro = -(dolfin.tr(self.Sigma.T * self.kinematics.C)) / 3 / self.kinematics.J
		self.P = self.bulk.P + self.dev.P
		self.sigma = self.bulk.sigma + self.dev.sigma
		self.Sigma_dev = self.Sigma + self.p_hydro * self.kinematics.J * self.kinematics.C_inv
		self.Sigma_VM = dolfin.sqrt(1.5 * dolfin.tr(self.Sigma_dev.T * self.Sigma_dev))

	# def get_free_energy(self, *args, **kwargs):

	#     Psi_bulk, Sigma_bulk = self.bulk.get_free_energy(*args, **kwargs)
	#     Psi_dev , Sigma_dev  = self.dev.get_free_energy(*args, **kwargs)

	#     Psi   = Psi_bulk   + Psi_dev
	#     Sigma = Sigma_bulk + Sigma_dev

	#     return Psi, Sigma

	# def get_PK2_stress(self, *args, **kwargs):

	#     Sigma_bulk = self.bulk.get_PK2_stress(*args, **kwargs)
	#     Sigma_dev  = self.dev.get_PK2_stress(*args, **kwargs)

	#     Sigma = Sigma_bulk + Sigma_dev

	#     return Sigma

	# def get_PK1_stress(self, *args, **kwargs):

	#     P_bulk = self.bulk.get_PK1_stress(*args, **kwargs)
	#     P_dev  = self.dev.get_PK1_stress(*args, **kwargs)

	#     P = P_bulk + P_dev

	#     return P
