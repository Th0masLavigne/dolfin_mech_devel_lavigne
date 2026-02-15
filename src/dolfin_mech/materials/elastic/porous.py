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

"""Porous elastic material scaling implementation.

This module provides a wrapper for elastic materials to scale their energy
and stress properties based on solid volume fractions, common in porous
media mechanics.
"""

import dolfin

from dolfin_mech.materials.elastic import ElasticMaterial

################################################################################


class Porous(ElasticMaterial):
	r"""Class representing a porous elastic material.

	Properties are scaled based on the solid volume fraction.
	In porous media mechanics, the apparent strain energy density and stresses of
	a representative elementary volume (REV) are often scaled versions of the
	underlying solid constituent's properties. This class facilitates that
	scaling, specifically supporting linear scaling by the initial solid
	fraction :math:`\Phi_{s0}`.

	The total energy and stress are computed as:

	.. math::
	    \Psi_{porous} = \omega \cdot \Psi_{solid}

	.. math::
	    \mathbf{\Sigma}_{porous} = \omega \cdot \mathbf{\Sigma}_{solid}

	Where :math:`\omega` is the scaling factor:

	- For ``scaling="no"``: :math:`\omega = 1`.
	- For ``scaling="linear"``: :math:`\omega = \Phi_{s0}`.

	Attributes:
	solid_material (ElasticMaterial): The base elastic material representing the solid phase.
	Psi (UFL expression): Scaled strain energy density.
	Sigma (UFL expression): Scaled Second Piola-Kirchhoff stress tensor.
	P (UFL expression): Scaled First Piola-Kirchhoff stress tensor.
	sigma (UFL expression): Scaled Cauchy stress tensor.
	dWbulkdPhis (UFL expression, optional): Scaled derivative of bulk energy with respect to solid fraction.
	dWporedPhif (UFL expression, optional): Scaled derivative of pore energy with respect to fluid fraction.
	"""

	def __init__(self, solid_material, scaling="no", Phis0=None):
		"""Initializes the PorousElasticMaterial.

		:param solid_material: An instance of an ElasticMaterial class.
		:type solid_material: dolfin_mech.materials.elastic.ElasticMaterial
		:param scaling: Scaling strategy, either "no" or "linear". Defaults to "no".
		:type scaling: str, optional
		:param Phis0: Initial solid volume fraction. Required if scaling is "linear".
		:type Phis0: dolfin.Function, dolfin.Constant, or float, optional
		:raises AssertionError: If scaling is "linear" but Phis0 is None, or if an unknown scaling string is provided.
		"""
		self.solid_material = solid_material

		if scaling == "no":
			scaling = dolfin.Constant(1)
			# self.Psi   = self.material.Psi
			# if (hasattr(self.material,       "Sigma")): self.Sigma       = self.material.Sigma
			# if (hasattr(self.material,           "P")): self.P           = self.material.P
			# if (hasattr(self.material,       "sigma")): self.sigma       = self.material.sigma
			# if (hasattr(self.material, "dWbulkdPhis")): self.dWbulkdPhis = self.material.dWbulkdPhis
		elif scaling == "linear":
			assert Phis0 is not None
			scaling = Phis0
			# self.Psi   = Phis0 * self.material.Psi
			# if (hasattr(self.material,       "Sigma")): self.Sigma       = Phis0 * self.material.Sigma
			# if (hasattr(self.material,           "P")): self.P           = Phis0 * self.material.P
			# if (hasattr(self.material,       "sigma")): self.sigma       = Phis0 * self.material.sigma
			# if (hasattr(self.material, "dWbulkdPhis")): self.dWbulkdPhis = Phis0 * self.material.dWbulkdPhis
		else:
			assert 0, 'scaling must be "no" or "linear". Aborting.'

		for attr in ("Psi", "Sigma", "P", "sigma", "dWbulkdPhis", "dWporedPhif"):
			if hasattr(self.solid_material, attr):
				setattr(self, attr, scaling * getattr(self.solid_material, attr))
