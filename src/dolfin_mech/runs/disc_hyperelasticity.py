# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the Disc_Hyperelasticity driver.

Validation benchmark for 2D finite
strain problems. It orchestrates the simulation of a hyperelastic disc under
either prescribed radial expansion or uniform boundary pressure, incorporating
automated mesh generation and mixed u-p formulations for incompressibility.
"""

import math

import dolfin
import numpy

from .. import core, problems
from .disc_mesh import Disc_Mesh

################################################################################


def Disc_Hyperelasticity(
	incomp=0,
	mesh_params={},
	mat_params={},
	step_params={},
	load_params={},
	res_basename="run_Disc_Hyperelasticity",
	write_vtus_with_preserved_connectivity=False,
	verbose=0,
):
	"""Runs a 2D benchmark simulation of a hyperelastic disc.

	This function sets up and solves a finite strain problem on a circular domain.
	It supports two distinct loading scenarios:

	1.  **Radial Displacement (`type="disp"`)**: The boundary is pulled radially
	    outward by a prescribed amount :math:`dR`. This is achieved by calculating
	    the target displacement vector for every node on the boundary and applying
	    point-wise Dirichlet constraints.
	2.  **Pressure Loading (`type="pres"`)**: A uniform pressure :math:`p` is
	    applied to the outer boundary. To prevent rigid body motion in this
	    traction-controlled case, specific points on the axes are pinned.



	**Geometry:**
	Defined by center :math:`(X_0, Y_0)` and radius :math:`R`.

	:param incomp: Flag for incompressibility (0 = Compressible, 1 = Incompressible).
	:type incomp: int
	:param mesh_params: Dictionary for mesh generation (``X0``, ``Y0``, ``R``).
	:type mesh_params: dict
	:param mat_params: Dictionary for material constitutive law.
	:type mat_params: dict
	:param step_params: Dictionary for time stepping (``Deltat``, ``dt_ini``).
	:type step_params: dict
	:param load_params: Dictionary configuring the load:
	    - ``type`` (str): "disp" or "pres".
	    - ``dR`` (float): Radial displacement magnitude (for "disp").
	    - ``p`` (float): Pressure magnitude (for "pres").
	:type load_params: dict
	:param res_basename: Output file prefix.
	:type res_basename: str
	:param verbose: Verbosity level.
	:type verbose: int
	"""
	################################################################### Mesh ###

	X0 = mesh_params.get("X0", 0.5)
	Y0 = mesh_params.get("Y0", 0.5)
	R = mesh_params.get("R", 0.3)

	mesh, boundaries_mf, S_id, points_mf, x1_sd, x2_sd, x3_sd, x4_sd = Disc_Mesh(params=mesh_params)

	################################################################ Problem ###

	if incomp:
		displacement_degree = 2  # MG20211219: Incompressibility requires displacement_degree >= 2 ?!
	else:
		displacement_degree = 1

	problem = problems.Hyperelasticity(
		mesh=mesh,
		define_facet_normals=1,
		boundaries_mf=boundaries_mf,
		points_mf=points_mf,
		displacement_degree=displacement_degree,
		quadrature_degree="default",
		w_incompressibility=incomp,
		elastic_behavior=mat_params,
	)

	########################################## Boundary conditions & Loading ###

	Deltat = step_params.get("Deltat", 1.0)
	dt_ini = step_params.get("dt_ini", 1.0)
	dt_min = step_params.get("dt_min", 1.0)

	k_step = problem.add_step(Deltat=Deltat, dt_ini=dt_ini, dt_min=dt_min)

	load_type = load_params.get("type", "disp")

	if (
		load_type == "disp"
	):  # MG20220813: It would be possible to impose the spatially varying displacement directly through an expression, but this would need to be implemented within Constraint, e.g. with a TimeVaryingExpression.
		surface_nodes_coords = [
			node_coords
			for node_coords in mesh.coordinates()
			if dolfin.near((node_coords[0] - X0) ** 2 + (node_coords[1] - Y0) ** 2, R**2, eps=1e-3)
		]
		dR = load_params.get("dR", +0.1)
		for X in surface_nodes_coords:
			X_inplane = numpy.array(X) - numpy.array([X0, Y0])
			R = numpy.linalg.norm(X_inplane)
			T = math.atan2(X_inplane[1], X_inplane[0])
			r = R + dR
			t = T
			x_inplane = numpy.array([r * math.cos(t), r * math.sin(t)])
			x = numpy.array([X0, Y0]) + x_inplane
			U = x - X
			# X_sd = dolfin.AutoSubDomain(lambda x, on_boundary: dolfin.near(x[0], X[0], eps=1e-3) and dolfin.near(x[1], X[1], eps=1e-3)) # MG20220813: OMG this behaves so weird!
			X_sd = dolfin.CompiledSubDomain("near(x[0], x0) && near(x[1], y0)", x0=X[0], y0=X[1])
			problem.add_constraint(
				V=problem.displacement_subsol.fs,
				sub_domain=X_sd,
				val_ini=[0.0, 0.0],
				val_fin=U,
				k_step=k_step,
				method="pointwise",
			)
	elif load_type == "pres":
		problem.add_constraint(V=problem.displacement_subsol.fs.sub(1), sub_domain=x1_sd, val=0.0, method="pointwise")
		problem.add_constraint(V=problem.displacement_subsol.fs.sub(0), sub_domain=x2_sd, val=0.0, method="pointwise")
		problem.add_constraint(V=problem.displacement_subsol.fs.sub(1), sub_domain=x3_sd, val=0.0, method="pointwise")
		problem.add_constraint(V=problem.displacement_subsol.fs.sub(0), sub_domain=x4_sd, val=0.0, method="pointwise")
		p = load_params.get("p", -1.0)
		problem.add_surface_pressure_loading_operator(measure=problem.dS(S_id), P_ini=0, P_fin=p, k_step=k_step)

	################################################# Quantities of Interest ###

	problem.add_point_displacement_qoi(name="U", coordinates=[X0 + R, Y0], component=0)

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
		write_qois_limited_precision=1,
		write_sol=res_basename * verbose,
		write_vtus=res_basename * verbose,
		write_vtus_with_preserved_connectivity=write_vtus_with_preserved_connectivity,
	)

	success = integrator.integrate()
	assert success, "Integration failed. Aborting."

	integrator.close()
