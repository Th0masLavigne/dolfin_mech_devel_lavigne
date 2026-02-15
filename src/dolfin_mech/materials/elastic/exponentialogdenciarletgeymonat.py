# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Cécile Patte, 2019-2021                                              ###
###                                                                          ###
### INRIA, Palaiseau, France                                                 ###
###                                                                          ###
################################################################################

"""Implementation of exponential elastic material models for the Dolfin_mech library."""

import dolfin

from .elastic import ElasticMaterial

################################################################################


class ExponentialOgdenCiarletGeymonat(ElasticMaterial):
	r"""Class implementing an Exponential Ogden-Ciarlet-Geymonat elastic material model.

	This constitutive model provides a volumetric strain energy density function
	based on an exponential formulation. It is particularly effective for
	enforcing compressibility constraints or modeling the volumetric response
	of porous media like lung tissue[cite: 5, 17].

	The strain energy density :math:`\Psi` is defined as:

	.. math::

	    \Psi = \alpha \left( e^{\gamma (J^2 - 1 - 2\ln J)} - 1 \right)

	Attributes:
	    kinematics (Kinematics): Kinematic quantities (F, J, C_inv, etc.).
	    alpha (dolfin.Constant): Scaling parameter for the energy density.
	    gamma (dolfin.Constant): Exponential exponent controlling the stiffening
	        rate under volume change.
	    Psi (ufl.Form): Strain energy density function.
	    Sigma (ufl.Form): Second Piola-Kirchhoff stress tensor :math:`\mathbf{S}`.
	    P (ufl.Form): First Piola-Kirchhoff stress tensor :math:`\mathbf{P} = \mathbf{F}\mathbf{S}`.
	    sigma (ufl.Form): Cauchy stress tensor :math:`\boldsymbol{\sigma} = J^{-1} \mathbf{P}\mathbf{F}^T`.
	"""

	def __init__(self, kinematics, parameters):
		"""Initialize the Exponential Ogden-Ciarlet-Geymonat material model.

		Args:
		    kinematics (Kinematics): Kinematic object providing deformation
		        tensors and their invariants.
		    parameters (dict): Dictionary containing the material constants:
		        ``"alpha"`` and ``"gamma"``.

		Raises:
		    AssertionError: If ``alpha`` or ``gamma`` are missing from parameters.
		"""
		self.kinematics = kinematics

		if ("alpha" in parameters) and ("gamma" in parameters):
			self.alpha = dolfin.Constant(parameters["alpha"])
			self.gamma = dolfin.Constant(parameters["gamma"])
		else:
			assert 0, 'No parameter found: "+str(parameters)+". Must provide alpha & gamma. Aborting.'

		self.Psi = (self.alpha) * (
			dolfin.exp(self.gamma * (self.kinematics.J**2 - 1 - 2 * dolfin.ln(self.kinematics.J))) - 1
		)

		self.Sigma = (
			(self.alpha)
			* dolfin.exp(self.gamma * (self.kinematics.J**2 - 1 - 2 * dolfin.ln(self.kinematics.J)))
			* (2 * self.gamma)
			* (self.kinematics.J**2 - 1)
			* self.kinematics.C_inv
		)

		# self.P = dolfin.diff(self.Psi, self.kinematics.F) # MG20220426: Cannot do that for micromechanics problems
		self.P = self.kinematics.F * self.Sigma

		self.sigma = self.P * self.kinematics.F.T / self.kinematics.J
