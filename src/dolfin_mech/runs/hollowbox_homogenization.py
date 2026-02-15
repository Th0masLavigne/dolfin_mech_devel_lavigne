# coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2024                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the run_HollowBox_Homogenization driver.

Calculates the effective
(homogenized) elastic properties of a periodic unit cell. It automates the
setup of the Representative Volume Element (RVE), solves the necessary
corrector problems, and extracts macroscopic Young's modulus, Poisson's ratio,
and bulk modulus.
"""

import dolfin
import myPythonLibrary as mypy

from .. import problems
from .hollowbox_mesh import HollowBox_Mesh

################################################################################


def HollowBox_Homogenization(
	dim,
	mesh=None,
	mesh_params=None,
	mat_params={},
	res_basename="run_HollowBox_Homogenization",
	write_results_to_file=1,
	verbose=0,
):
	r"""Computes the effective elastic properties of a "Hollow Box" unit cell via homogenization.

	This function serves as a driver for the computational homogenization of a
	Representative Volume Element (RVE). The RVE consists of a solid matrix
	with a central void (the "Hollow Box").



	**Workflow:**
	1.  **Mesh Setup**: Generates or accepts a mesh defining the solid domain :math:`\Omega_s`.
	2.  **Volume Analysis**: Calculates the total RVE volume :math:`V_0` and the initial solid volume fraction :math:`\Phi_{s0} = V_{s0}/V_0`.
	3.  **Homogenization**: Instantiates a :class:`HomogenizationProblem` to solve the boundary value problems required to identify the effective stiffness tensor.
	4.  **Post-Processing**: Extracts effective Lamé parameters (:math:`\lambda_{hom}, \mu_{hom}`) and converts them to engineering constants.

	**Calculated Properties:**
	The function returns the homogenized isotropic constants:

	.. math::
	    E_{hom} = \frac{\mu_{hom}(3\lambda_{hom} + 2\mu_{hom})}{\lambda_{hom} + \mu_{hom}}, \quad
	    \nu_{hom} = \frac{\lambda_{hom}}{2(\lambda_{hom} + \mu_{hom})}

	:param dim: Spatial dimension (2 or 3).
	:type dim: int
	:param mesh: Pre-generated FEniCS mesh object (optional).
	:type mesh: dolfin.Mesh
	:param mesh_params: Dictionary of parameters to generate the mesh if ``mesh`` is None.
	:type mesh_params: dict
	:param mat_params: Material parameters for the solid phase (e.g., ``{"E": 100, "nu": 0.3}``).
	:type mat_params: dict
	:param res_basename: Base name for output files (data and logs).
	:type res_basename: str
	:param write_results_to_file: If True, writes computed properties to a ``.dat`` file.
	:type write_results_to_file: bool
	:param verbose: Verbosity level.
	:type verbose: int
	:return: A tuple containing:
	    - ``E_s``: Young's modulus of the solid phase.
	    - ``nu_s``: Poisson's ratio of the solid phase.
	    - ``E_hom``: Homogenized Young's modulus.
	    - ``nu_hom``: Homogenized Poisson's ratio.
	    - ``kappa_hom``: Homogenized Bulk modulus.
	"""
	################################################################### Mesh ###

	assert (mesh is not None) or (mesh_params is not None)
	if mesh is None:
		mesh = HollowBox_Mesh(params=mesh_params)

	coord = mesh.coordinates()
	xmax = max(coord[:, 0])
	xmin = min(coord[:, 0])
	ymax = max(coord[:, 1])
	ymin = min(coord[:, 1])
	if dim == 2:
		vol = (xmax - xmin) * (ymax - ymin)
		bbox = [xmin, xmax, ymin, ymax]
	elif dim == 3:
		zmax = max(coord[:, 2])
		zmin = min(coord[:, 2])
		vol = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
		bbox = [xmin, xmax, ymin, ymax, zmin, zmax]

	V_0 = vol
	dV = dolfin.Measure("dx", domain=mesh)
	V_s0 = dolfin.assemble(dolfin.Constant(1.0) * dV)
	Phi_s0 = V_s0 / V_0
	print("Phi_s0 = " + str(Phi_s0))

	################################################################ Problem ###

	homogenization_problem = problems.Homogenization(dim=dim, mesh=mesh, mat_params=mat_params, vol=vol, bbox=bbox)
	[lmbda_, mu_] = homogenization_problem.get_lambda_and_mu()
	kappa_ = homogenization_problem.get_kappa()

	E_ = mu_ * (3 * lmbda_ + 2 * mu_) / (lmbda_ + mu_)
	nu_ = lmbda_ / (lmbda_ + mu_) / 2

	if write_results_to_file:
		qoi_printer = mypy.DataPrinter(
			names=["E_s", "nu_s", "E_hom", "nu_hom", "kappa_hom"],
			filename=res_basename + "-qois.dat",
			limited_precision=False,
		)

		qoi_printer.write_line([mat_params["E"], mat_params["nu"], E_, nu_, kappa_])
		qoi_printer.write_line(
			[mat_params["E"], mat_params["nu"], E_, nu_, kappa_]
		)  # MG20231124: Need to write twice for some postprocessing issue

	return mat_params["E"], mat_params["nu"], E_, nu_, kappa_
