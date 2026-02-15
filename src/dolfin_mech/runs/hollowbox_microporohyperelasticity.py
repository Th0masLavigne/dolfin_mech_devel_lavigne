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

"""Defines the run_HollowBox_MicroPoroHyperelasticity driver.

Performs
finite-strain multiscale simulations on a periodic unit cell. It couples
microscopic solid hyperelasticity and surface tension with macroscopic
homogenization, allowing for complex loading paths via mixed stretch-controlled
and stress-controlled boundary conditions.
"""

import dolfin
import numpy

from .. import core, problems
from .hollowbox_mesh import HollowBox_Mesh

################################################################################


def HollowBox_MicroPoroHyperelasticity(
	dim,
	mesh=None,
	mesh_params=None,
	displacement_perturbation_degree=1,
	quadrature_degree=3,
	mat_params={},
	bcs="pbc",
	step_params={},
	load_params={},
	res_basename="run_HollowBox_MicroPoroHyperelasticity",
	add_p_hydro_and_Sigma_VM_FoI=False,
	write_qois_limited_precision=True,
	verbose=0,
):
	r"""Runs a micro-poro-hyperelastic simulation on a "Hollow Box" unit cell.

	This function simulates the coupled mechanical behavior of a porous RVE (Representative
	Volume Element) consisting of a hyperelastic solid matrix and a fluid-filled void.
	It is designed to study **micro-scale mechanisms** such as pore inflation,
	capillary effects (surface tension), and effective macroscopic response under
	mixed loading conditions (prescribed stretch or stress).



	**Key Physics:**
	1.  **Hyperelasticity**: Finite strain solid mechanics for the matrix.
	2.  **Pore Pressure**: Loading via internal fluid pressure :math:`p_f`.
	3.  **Surface Tension**: Capillary forces :math:`\gamma` acting on the pore interface :math:`\Gamma_{int}`.
	4.  **Homogenization**: Periodic Boundary Conditions (PBC) couple micro-scale fluctuations to macro-scale deformation gradients :math:`\bar{\mathbf{F}}`.

	**Loading Control:**
	The loading is defined stepwise via ``load_params``, allowing for complex paths combining:
	- **Macroscopic Stretch** (:math:`\bar{U}_{ij}`): Prescribed deformation.
	- **Macroscopic Stress** (:math:`\bar{\Sigma}_{ij}`): Prescribed average stress.
	- **Fluid Pressure** (:math:`p_f`): Pore pressure.
	- **Surface Tension** (:math:`\gamma`): Interfacial tension coefficient.

	:param dim: Dimension (2 or 3).
	:type dim: int
	:param mesh: Pre-generated mesh (optional).
	:type mesh: dolfin.Mesh
	:param mesh_params: Parameters for mesh generation if ``mesh`` is None.
	:type mesh_params: dict
	:param displacement_perturbation_degree: FE degree for displacement fluctuations.
	:type displacement_perturbation_degree: int
	:param mat_params: Material parameters for the solid skeleton.
	:type mat_params: dict
	:param bcs: Boundary condition type ("pbc" for periodic).
	:type bcs: str
	:param load_params: Dictionary defining the loading path (timelines for U_bar, sigma_bar, pf, gamma).
	:type load_params: dict
	:param res_basename: Output file prefix.
	:type res_basename: str
	:return: The fully solved ``problem`` object.
	"""
	assert (mesh is not None) or (mesh_params is not None)
	if mesh is None:
		mesh = HollowBox_Mesh(params=mesh_params)

	coord = mesh.coordinates()
	xmax = max(coord[:, 0])
	xmin = min(coord[:, 0])
	ymax = max(coord[:, 1])
	ymin = min(coord[:, 1])
	if dim == 2:
		bbox = [xmin, xmax, ymin, ymax]
		vertices = numpy.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
		a1 = vertices[1, :] - vertices[0, :]  # first vector generating periodicity
		a2 = vertices[3, :] - vertices[0, :]  # second vector generating periodicity
		tol = 1e-8
		assert (
			numpy.linalg.norm(vertices[2, :] - vertices[3, :] - a1) <= tol
		)  # check if UC vertices form indeed a parallelogram
		assert (
			numpy.linalg.norm(vertices[2, :] - vertices[1, :] - a2) <= tol
		)  # check if UC vertices form indeed a parallelogram
	elif dim == 3:
		zmax = max(coord[:, 2])
		zmin = min(coord[:, 2])
		bbox = [xmin, xmax, ymin, ymax, zmin, zmax]
		vertices = numpy.array(
			[
				[xmin, ymin, zmin],
				[xmax, ymin, zmin],
				[xmax, ymax, zmin],
				[xmin, ymax, zmin],
				[xmin, ymin, zmax],
				[xmax, ymin, zmax],
				[xmax, ymax, zmax],
				[xmin, ymax, zmax],
			]
		)

	################################################## Subdomains & Measures ###

	tol = 1e-8
	xmin_sd = dolfin.CompiledSubDomain("near(x[0], x0, tol) && on_boundary", x0=xmin, tol=tol)
	xmax_sd = dolfin.CompiledSubDomain("near(x[0], x0, tol) && on_boundary", x0=xmax, tol=tol)
	ymin_sd = dolfin.CompiledSubDomain("near(x[1], x0, tol) && on_boundary", x0=ymin, tol=tol)
	ymax_sd = dolfin.CompiledSubDomain("near(x[1], x0, tol) && on_boundary", x0=ymax, tol=tol)
	if dim == 3:
		zmin_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=zmin, tol=tol)
	if dim == 3:
		zmax_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=zmax, tol=tol)

	# if (dim==2):
	#     sint_sd = dolfin.CompiledSubDomain("near(pow(x[0] - x0, 2) + pow(x[1] - y0, 2), pow(r0, 2), 1e-2) && on_boundary", x0=x0, y0=y0, r0=r0)
	# elif (dim==3):
	#     sint_sd = dolfin.CompiledSubDomain("near(pow(x[0] - x0, 2) + pow(x[1] - y0, 2) + pow(x[2] - z0, 2), pow(r0, 2), 1e-2) && on_boundary", x0=x0, y0=y0, z0=z0, r0=r0)

	xmin_id = 1
	xmax_id = 2
	ymin_id = 3
	ymax_id = 4
	if dim == 3:
		zmin_id = 5
	if dim == 3:
		zmax_id = 6
	# sint_id = 9

	boundaries_mf = dolfin.MeshFunction(
		"size_t", mesh, mesh.topology().dim() - 1
	)  # MG20180418: size_t looks like unsigned int, but more robust wrt architecture and os
	boundaries_mf.set_all(0)

	xmin_sd.mark(boundaries_mf, xmin_id)
	xmax_sd.mark(boundaries_mf, xmax_id)
	ymin_sd.mark(boundaries_mf, ymin_id)
	ymax_sd.mark(boundaries_mf, ymax_id)
	if dim == 3:
		zmin_sd.mark(boundaries_mf, zmin_id)
	if dim == 3:
		zmax_sd.mark(boundaries_mf, zmax_id)
	# sint_sd.mark(boundaries_mf, sint_id)

	if verbose:
		xdmf_file_boundaries = dolfin.XDMFFile(res_basename + "-boundaries.xdmf")
		xdmf_file_boundaries.write(boundaries_mf)
		xdmf_file_boundaries.close()

	################################################################ Problem ###

	problem = problems.MicroPoroHyperelasticity(
		mesh=mesh,
		mesh_bbox=bbox,
		vertices=vertices,
		boundaries_mf=boundaries_mf,
		displacement_perturbation_degree=displacement_perturbation_degree,
		quadrature_degree=quadrature_degree,
		solid_behavior=mat_params,
		bcs=bcs,
	)

	################################################################ Loading ###

	n_steps = step_params.get("n_steps", 1)
	Deltat_lst = step_params.get("Deltat_lst", [step_params.get("Deltat", 1.0) / n_steps] * n_steps)
	dt_ini_lst = step_params.get("dt_ini_lst", [step_params.get("dt_ini", 1.0) / n_steps] * n_steps)
	dt_min_lst = step_params.get("dt_min_lst", [step_params.get("dt_min", 1.0) / n_steps] * n_steps)
	dt_max_lst = step_params.get("dt_max_lst", [step_params.get("dt_max", 1.0) / n_steps] * n_steps)

	# gamma = load_params.get("gamma", 0.0)

	U_bar_ij_lst = [[None for i in range(dim)] for j in range(dim)]
	sigma_bar_ij_lst = [[None for i in range(dim)] for j in range(dim)]
	for i in range(dim):
		for j in range(dim):
			U_bar_ij_lst[i][j] = load_params.get(
				"U_bar_" + str(i) + str(j) + "_lst",
				[load_params.get("U_bar_" + str(i) + str(j), None) for k_step in range(n_steps)],
			)
			sigma_bar_ij_lst[i][j] = load_params.get(
				"sigma_bar_" + str(i) + str(j) + "_lst",
				[load_params.get("sigma_bar_" + str(i) + str(j), None) for k_step in range(n_steps)],
			)
	pf_lst = load_params.get("pf_lst", [(k_step + 1) * load_params.get("pf", 0) / n_steps for k_step in range(n_steps)])
	gamma_lst = load_params.get(
		"gamma_lst", [(k_step + 1) * load_params.get("gamma", 0) / n_steps for k_step in range(n_steps)]
	)
	tension_params = load_params.get("tension_params", {})

	for k_step in range(n_steps):
		Deltat = Deltat_lst[k_step]
		dt_ini = dt_ini_lst[k_step]
		dt_min = dt_min_lst[k_step]
		dt_max = dt_max_lst[k_step]

		k_step = problem.add_step(Deltat=Deltat, dt_ini=dt_ini, dt_min=dt_min, dt_max=dt_max)

		pf = pf_lst[k_step]
		pf_old = pf_lst[k_step - 1] if (k_step > 0) else 0.0
		problem.add_surface_pressure_loading_operator(measure=problem.dS(0), P_ini=pf_old, P_fin=pf, k_step=k_step)

		for i in range(dim):
			for j in range(dim):
				U_bar_ij = U_bar_ij_lst[i][j][k_step]
				U_bar_ij_old = U_bar_ij_lst[i][j][k_step - 1] if (k_step > 0) else 0.0
				sigma_bar_ij = sigma_bar_ij_lst[i][j][k_step]
				sigma_bar_ij_old = sigma_bar_ij_lst[i][j][k_step - 1] if (k_step > 0) else 0.0
				assert (U_bar_ij is not None) or (sigma_bar_ij is not None)
				if U_bar_ij is not None:
					problem.add_macroscopic_stretch_component_penalty_operator(
						i=i, j=j, U_bar_ij_ini=U_bar_ij_old, U_bar_ij_fin=U_bar_ij, pen_val=1e6, k_step=k_step
					)
				elif sigma_bar_ij is not None:
					problem.add_macroscopic_stress_component_constraint_operator(
						i=i,
						j=j,
						sigma_bar_ij_ini=sigma_bar_ij_old,
						sigma_bar_ij_fin=sigma_bar_ij,
						pf_ini=pf_old,
						pf_fin=pf,
						k_step=k_step,
					)

		problem.add_surface_area_operator(measure=problem.dS(0), k_step=k_step)

		gamma = gamma_lst[k_step]
		gamma_old = gamma_lst[k_step - 1] if (k_step > 0) else 0.0
		problem.add_surface_tension_loading_operator(
			measure=problem.dS(0), gamma_ini=gamma_old, gamma_fin=gamma, tension_params=tension_params, k_step=k_step
		)

	################################################# Quantities of Interest ###

	problem.add_deformed_solid_volume_qoi()
	problem.add_deformed_fluid_volume_qoi()
	problem.add_deformed_volume_qoi()
	problem.add_macroscopic_stretch_qois()
	problem.add_macroscopic_solid_stress_qois()
	problem.add_macroscopic_solid_hydrostatic_pressure_qoi()
	problem.add_macroscopic_stress_qois()
	problem.add_fluid_pressure_qoi()
	problem.add_interfacial_surface_qois()

	if add_p_hydro_and_Sigma_VM_FoI:
		for operator in problem.operators:
			if hasattr(operator, "material"):
				material = operator.material
				break
		problem.add_foi(expr=material.p_hydro, fs=problem.sfoi_fs, name="p_hydro")
		problem.add_foi(expr=material.Sigma_VM, fs=problem.sfoi_fs, name="Sigma_VM")

	################################################################# Solver ###

	solver = core.NonlinearSolver(
		problem=problem,
		parameters={"sol_tol": [1e-6] * len(problem.subsols), "n_iter_max": 32},
		relax_type="constant",
		write_iter=0,
	)

	integrator = core.TimeIntegrator(
		problem=problem,
		solver=solver,
		parameters={"n_iter_for_accel": 4, "n_iter_for_decel": 16, "accel_coeff": 2, "decel_coeff": 2},
		print_out=res_basename * verbose,
		print_sta=res_basename * verbose,
		write_qois=res_basename + "-qois",
		write_qois_limited_precision=write_qois_limited_precision,
		write_sol=res_basename * verbose,
	)

	success = integrator.integrate()
	assert success, "Integration failed. Aborting."

	integrator.close()

	return problem
