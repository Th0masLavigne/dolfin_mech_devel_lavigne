# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the LinearizedIncompressibility class.

Assembles the
linearized volumetric constraint (divergence-free condition) to enforce
material incompressibility within a small-strain mixed variational framework.
"""

import dolfin

from .operator import Operator

################################################################################


class LinearizedIncompressibility(Operator):
	r"""Operator enforcing the incompressibility constraint in a linearized (small strain) elasticity framework.

	This operator is the counterpart to the :class:`LinearizedHydrostaticPressureOperator`.
	In a mixed displacement-pressure formulation, it assembles the equation
	governing the Lagrange multiplier (pressure) to enforce that the material
	volume remains constant under small deformations.

	The residual contribution is:

	.. math::
	    \mathcal{R}_{incomp} = - \int_{\Omega} \text{tr}(\mathbf{\epsilon}) \delta p \, d\Omega

	where:
	    - :math:`\mathbf{\epsilon}` is the infinitesimal strain tensor.
	    - :math:`\text{tr}(\mathbf{\epsilon}) = \nabla \cdot \mathbf{u}` is the
	      linearized volumetric strain.
	    - :math:`\delta p` is the test function associated with the pressure field.

	The constraint :math:`\text{tr}(\mathbf{\epsilon}) = 0` effectively ensures
	that the displacement field is solenoidal (divergence-free).

	Attributes:
	    kinematics (Kinematics): Kinematic variables providing the infinitesimal strain.
	    measure (dolfin.Measure): Integration measure (typically ``dx``).
	    res_form (UFL form): The resulting residual variational form.


	:param kinematics: Kinematics object handling infinitesimal strains.
	:type kinematics: dmech.Kinematics
	:param p_test: Test function associated with the pressure field (Lagrange multiplier).
	:type p_test: dolfin.TestFunction
	:param measure: Dolfin measure for domain integration.
	:type measure: dolfin.Measure
	"""

	def __init__(self, kinematics, p_test, measure):
		"""Initializes the LinearizedIncompressibilityOperator."""
		self.kinematics = kinematics
		self.measure = measure

		self.res_form = -dolfin.tr(self.kinematics.epsilon) * p_test * self.measure
