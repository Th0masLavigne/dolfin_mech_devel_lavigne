# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Ciarlet-Geymonat volumetric elastic material implementation.

This module provides a strictly convex volumetric potential used to penalize
volume changes and prevent numerical inversion in hyperelastic formulations.
"""

import dolfin

from dolfin_mech.materials.elastic import ElasticMaterial

################################################################################


class OgdenCiarletGeymonat(ElasticMaterial):
	r"""Class representing the Ciarlet-Geymonat volumetric elastic material model.

	This model provides a strictly convex volumetric strain energy density
	functional. It is often used as the compressible part of a hyperelastic
	formulation to penalize volume changes and prevent element inversion.

	The strain energy density :math:`\Psi` is defined as:

	.. math::
	    \Psi = C_0 (J^2 - 1 - 2\ln(J))

	Where:
	    - :math:`J = \det(\mathbf{F})` is the volume ratio (Jacobian).
	    - :math:`C_0` is a material parameter related to the bulk modulus.

	The second Piola-Kirchhoff stress :math:`\mathbf{\Sigma}` is:

	.. math::
	    \mathbf{\Sigma} = 2 C_0 (J^2 - 1) \mathbf{C}^{-1}

	Attributes:
	    kinematics (Kinematics): Object containing kinematic variables (J, C_inv, etc.).
	    C0 (float): Material constant scaling the volumetric response.
	    Psi (UFL expression): Volumetric strain energy density.
	    Sigma (UFL expression): Second Piola-Kirchhoff stress tensor.
	    P (UFL expression): First Piola-Kirchhoff stress tensor.
	    sigma (UFL expression): Cauchy stress tensor.
	    checkJ (bool): If True, adds a conditional check to ensure :math:`J > 0` during stress evaluation.
	"""

	def __init__(self, kinematics, parameters, decoup=False):
		"""Initializes the OgdenCiarletGeymonatElasticMaterial.

		:param kinematics: Kinematics object containing deformation tensors and Jacobian.
		:type kinematics: dolfin_mech.kinematics.Kinematics
		:param parameters: Dictionary containing 'C0'. Can also include 'checkJ' (bool).
		:type parameters: dict
		:param decoup: If True, indicates a decoupled formulation (passed to C0 getter).
		:type decoup: bool, optional
		"""
		self.kinematics = kinematics

		self.C0 = self.get_C0_from_parameters(parameters, decoup)

		self.Psi = self.C0 * (
			self.kinematics.J**2 - 1 - 2 * dolfin.ln(self.kinematics.J)
		)  # MG20180516: In 2d, plane strain

		self.checkJ = parameters.get("checkJ", False)
		if self.checkJ:
			self.Sigma = dolfin.conditional(  # MG20230320: Otherwise Sigma is well defined for J < 0…
				dolfin.gt(self.kinematics.J, 0.0),
				2
				* self.C0
				* (self.kinematics.J**2 - 1)
				* self.kinematics.C_inv,  # MG20200206: Cannot differentiate Psi wrt to C because J is not defined as a function of C
				self.kinematics.C_inv / dolfin.Constant(0.0),
			)

		else:
			self.Sigma = (
				2 * self.C0 * (self.kinematics.J**2 - 1) * self.kinematics.C_inv
			)  # MG20200206: Cannot differentiate Psi wrt to C because J is not defined as a function of C

		if self.kinematics.dim == 2:
			self.Sigma_ZZ = 2 * self.C0 * (self.kinematics.J**2 - 1)

		# self.P = dolfin.diff(self.Psi, self.kinematics.F) # MG20220426: Cannot do that for micromechanics problems
		# self.P = 2*self.C0 * (self.kinematics.J**2 - 1) * self.kinematics.F_inv.T
		self.P = self.kinematics.F * self.Sigma

		self.sigma = self.P * self.kinematics.F.T / self.kinematics.J

	# def get_free_energy(self,
	#         U=None,
	#         C=None):

	#     C  = self.get_C_from_U_or_C(U, C)
	#     JF = dolfin.sqrt(dolfin.det(C)) # MG20200207: Watch out! This is well defined for inverted elements!

	#     Psi   = self.C0 * (JF**2 - 1 - 2*dolfin.ln(JF)) # MG20180516: in 2d, plane strain
	#     Sigma = 2*dolfin.diff(Psi, C)

	#     # C_inv = dolfin.inv(C)
	#     # Sigma = 2*self.C0 * (JF**2 - 1) * C_inv

	#     return Psi, Sigma
