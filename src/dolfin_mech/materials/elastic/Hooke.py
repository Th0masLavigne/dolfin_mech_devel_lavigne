# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Implementation of linear elastic material models based on Hooke's law.

This module provides classes for standard isotropic linear elasticity, as well
as its decomposition into volumetric (bulk) and deviatoric (shear) components.
These models are designed for use with linearized kinematics.
"""

import dolfin

from .Elastic import ElasticMaterial

################################################################################


class Hooke(ElasticMaterial):
	r"""Class implementing the classical isotropic linear elastic Hooke material model.

	This model relates the Cauchy stress :math:`\boldsymbol{\sigma}` to the
	infinitesimal strain :math:`\boldsymbol{\varepsilon}` through the Lamé parameters
	:math:`\lambda` and :math:`\mu`.

	The strain energy density :math:`\psi` is defined as:

	.. math::

	    \psi = \frac{\lambda}{2} (\text{tr } \boldsymbol{\varepsilon})^2 + \mu \boldsymbol{\varepsilon} :
	    \boldsymbol{\varepsilon}

	Attributes:
	    kinematics (LinearizedKinematics): Kinematic object providing the strain tensor
	        :math:`\boldsymbol{\varepsilon}`.
	    lmbda (dolfin.Constant): First Lamé parameter :math:`\lambda`.
	    mu (dolfin.Constant): Second Lamé parameter (shear modulus) :math:`\mu`.
	    psi (ufl.Form): Strain energy density function.
	    sigma (ufl.Form): Cauchy stress tensor :math:`\boldsymbol{\sigma} = \lambda (\text{tr }
	        boldsymbol{\varepsilon}) \mathbf{I} + 2\mu\boldsymbol{\varepsilon}`.
	"""

	def __init__(self, kinematics, parameters):
		"""Initialize the Hooke material model.

		Args:
		    kinematics (LinearizedKinematics): Kinematic quantities.
		    parameters (dict): Material parameters containing ``"lambda"`` and ``"mu"``
		        or ``"E"`` and ``"nu"``.
		"""
		self.kinematics = kinematics

		self.lmbda = self.get_lambda_from_parameters(parameters)
		self.mu = self.get_mu_from_parameters(parameters)

		self.psi = (self.lmbda / 2) * dolfin.tr(self.kinematics.epsilon) ** 2 + self.mu * dolfin.inner(
			self.kinematics.epsilon, self.kinematics.epsilon
		)

		self.sigma = dolfin.diff(self.psi, self.kinematics.epsilon)
		# self.sigma = self.lmbda * dolfin.tr(self.kinematics.epsilon) * self.kinematics.I + 2 * self.mu * self.kinematics.epsilon

		self.Sigma = self.sigma
		self.P = self.sigma

		if self.kinematics.dim == 2:
			self.sigma_ZZ = self.lmbda * dolfin.tr(self.kinematics.epsilon)

	# def get_free_energy(self,
	#         U=None,
	#         epsilon=None):

	#     epsilon = self.get_epsilon_from_U_or_epsilon(
	#         U, epsilon)

	#     psi = (self.lmbda/2) * dolfin.tr(epsilon)**2 + self.mu * dolfin.inner(epsilon, epsilon)
	#     sigma = dolfin.diff(psi, epsilon)

	#     # assert (epsilon.ufl_shape[0] == epsilon.ufl_shape[1])
	#     # dim = epsilon.ufl_shape[0]
	#     # I = dolfin.Identity(dim)
	#     # sigma = self.lmbda * dolfin.tr(epsilon) * I + 2 * self.mu * epsilon

	#     return psi, sigma


################################################################################


class HookeBulk(ElasticMaterial):
	r"""Class implementing the spherical (bulk) part of the Hooke material model.

	This model captures only the volumetric response, relating the pressure
	to the trace of the strain tensor.

	The volumetric strain energy density is defined as:

	.. math::

	    \psi = \frac{d K}{2} (\text{tr } \boldsymbol{\varepsilon}_{sph})^2

	where :math:`d` is the spatial dimension and :math:`K` is the bulk modulus.
	"""

	def __init__(self, kinematics, parameters):
		r"""Initialize the Hooke Bulk material model.

		Args:
		    kinematics (LinearizedKinematics): Kinematic quantities.
		    parameters (dict): Material parameters used to derive :math:`K`.
		"""
		self.kinematics = kinematics

		# self.K = self.get_K_from_parameters(parameters)
		self.lmbda, self.mu = self.get_lambda_and_mu_from_parameters(parameters)
		self.K = (self.kinematics.dim * self.lmbda + 2 * self.mu) / self.kinematics.dim

		self.psi = (self.kinematics.dim * self.K / 2) * dolfin.tr(self.kinematics.epsilon_sph) ** 2
		self.sigma = self.kinematics.dim * self.K * self.kinematics.epsilon_sph

		self.Sigma = self.sigma
		self.P = self.sigma

		if self.kinematics.dim == 2:
			self.sigma_ZZ = self.K * dolfin.tr(self.kinematics.epsilon)

	# def get_free_energy(self,
	#         U=None,
	#         epsilon=None,
	#         epsilon_sph=None):

	#     epsilon_sph = self.get_epsilon_sph_from_U_epsilon_or_epsilon_sph(
	#         U, epsilon, epsilon_sph)
	#     assert (epsilon_sph.ufl_shape[0] == epsilon_sph.ufl_shape[1])
	#     dim = epsilon_sph.ufl_shape[0]

	#     psi   = (dim*self.K/2) * dolfin.tr(epsilon_sph)**2
	#     sigma =  dim*self.K    *           epsilon_sph

	#     return psi, sigma

	# def get_Cauchy_stress(self,
	#         U=None,
	#         epsilon=None,
	#         epsilon_sph=None):

	#     epsilon_sph = self.get_epsilon_sph_from_U_epsilon_or_epsilon_sph(
	#         U, epsilon, epsilon_sph)
	#     assert (epsilon_sph.ufl_shape[0] == epsilon_sph.ufl_shape[1])
	#     dim = epsilon_sph.ufl_shape[0]

	#     sigma = dim * self.K * epsilon_sph

	#     return sigma


################################################################################


class HookeDev(ElasticMaterial):
	r"""Class implementing the deviatoric (shear) part of the Hooke material model.

	This model captures the shape-changing response at constant volume,
	relating the deviatoric stress to the deviatoric strain.

	The deviatoric strain energy density is defined as:

	.. math::

	    \psi = G \boldsymbol{\varepsilon}_{dev} : \boldsymbol{\varepsilon}_{dev}

	where :math:`G` is the shear modulus (equivalent to :math:`\mu`).
	"""

	def __init__(self, kinematics, parameters):
		"""Initialize the Hooke Deviatoric material model.

		Args:
		    kinematics (LinearizedKinematics): Kinematic quantities.
		    parameters (dict): Material parameters used to derive :math:`G`.
		"""
		self.kinematics = kinematics

		self.G = self.get_G_from_parameters(parameters)

		self.psi = self.G * dolfin.inner(self.kinematics.epsilon_dev, self.kinematics.epsilon_dev)
		self.sigma = 2 * self.G * self.kinematics.epsilon_dev

		self.Sigma = self.sigma
		self.P = self.sigma

		if self.kinematics.dim == 2:
			self.sigma_ZZ = -2 * self.G / 3 * dolfin.tr(self.kinematics.epsilon)

	# def get_free_energy(self,
	#         U=None,
	#         epsilon=None,
	#         epsilon_dev=None):

	#     epsilon_dev = self.get_epsilon_dev_from_U_epsilon_or_epsilon_dev(
	#         U, epsilon, epsilon_dev)

	#     psi   =   self.G * dolfin.inner(epsilon_dev, epsilon_dev)
	#     sigma = 2*self.G *              epsilon_dev

	#     return psi, sigma

	# def get_Cauchy_stress(self,
	#         U=None,
	#         epsilon=None,
	#         epsilon_dev=None):

	#     epsilon_dev = self.get_epsilon_dev_from_U_epsilon_or_epsilon_dev(
	#         U, epsilon, epsilon_dev)

	#     sigma = 2*self.G * epsilon_dev

	#     return sigma
