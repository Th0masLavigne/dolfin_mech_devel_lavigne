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

"""Composite hyperelastic material combining Ciarlet-Geymonat, Neo-Hookean, and Mooney-Rivlin models.

This module provides a complex material implementation that independently
parameterizes volumetric compressibility and shear response using the first
and second invariants.
"""

import dolfin

from .elastic import ElasticMaterial
from .neohookeanmooneyrivlin import NeoHookeanMooneyRivlin
from .ogdenciarletgeymonat import OgdenCiarletGeymonat

################################################################################


class OgdenCiarletGeymonatNeoHookeanMooneyRivlin(ElasticMaterial):
	r"""Class representing a composite hyperelastic material model.

	Combines Ciarlet-Geymonat volumetric energy with a combined Neo-Hookean and
	Mooney-Rivlin deviatoric response.

	This class serves as a high-level wrapper that allows for sophisticated
	material modeling where the volumetric behavior (compressibility) and
	the shear behavior (based on the first and second invariants) are
	independently controlled but additively combined.

	The total strain energy density is:

	.. math::
	    \Psi_{total} = \Psi_{bulk}^{CG}(J) + \Psi_{dev}^{NH}(\mathbf{C}) + \Psi_{dev}^{MR}(\mathbf{C})

	Where:
	    - :math:`\Psi_{bulk}^{CG}` is handled by :class:`OgdenCiarletGeymonatElasticMaterial`.
	    - :math:`\Psi_{dev}^{NH+MR}` is handled by :class:`NeoHookeanMooneyRivlinElasticMaterial`.

	This class also provides auxiliary mechanical quantities such as the
	hydrostatic pressure and the Von Mises equivalent stress.

	Attributes:
	    kinematics (Kinematics): Kinematic variables and deformation tensors.
	    bulk (OgdenCiarletGeymonatElasticMaterial): The volumetric component.
	    dev (NeoHookeanMooneyRivlinElasticMaterial): The deviatoric component.
	    Psi (UFL expression): Total strain energy density.
	    Sigma (UFL expression): Total Second Piola-Kirchhoff stress tensor.
	    p_hydro (UFL expression): Hydrostatic pressure.
	    Sigma_dev (UFL expression): Deviatoric part of the Second Piola-Kirchhoff stress.
	    Sigma_VM (UFL expression): Von Mises equivalent stress calculated from :math:`\mathbf{\Sigma}_{dev}`.
	"""

	def __init__(self, kinematics, parameters, decoup=False):
		"""Initializes the OgdenCiarletGeymonatNeoHookeanMooneyRivlinElasticMaterial.

		:param kinematics: Kinematics object containing deformation tensors.
		:type kinematics: dolfin_mech.kinematics.Kinematics
		:param parameters: Dictionary containing material constants (C0, C1, C2, etc.).
		:type parameters: dict
		:param decoup: If True, uses the isochoric-volumetric decoupled formulation.
		:type decoup: bool, optional
		"""
		self.kinematics = kinematics

		self.bulk = OgdenCiarletGeymonat(kinematics, parameters, decoup)
		self.dev = NeoHookeanMooneyRivlin(kinematics, parameters, decoup)

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
