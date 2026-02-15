# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the Ball_Mesh utility.

Provides an automated interface to the
Gmsh API to generate, mesh, and convert 3D spherical geometries into
FEniCS-compatible XDMF formats with pre-tagged boundary surfaces.
"""

import dolfin
import gmsh
import meshio

################################################################################


def Ball_Mesh(params={}):
	"""Generates a 3D spherical mesh using Gmsh and converts it for FEniCS.

	This function automates the complete meshing pipeline:
	1.  **Geometry Creation**: Uses the OpenCASCADE kernel in Gmsh to define a sphere.
	2.  **Meshing**: Generates a 3D tetrahedral mesh with specified characteristic length ``l``.
	3.  **Conversion**: Writes to VTK, converts to XDMF via `meshio`, and imports into `dolfin.Mesh`.
	4.  **Boundary Tagging**: Identifies the outer surface of the sphere and marks it in a `MeshFunction`.



	This is particularly useful for creating reproducible, high-quality meshes for
	finite element benchmarks without manually using the Gmsh GUI.

	:param params: Dictionary of mesh parameters:
	    - ``X0``, ``Y0``, ``Z0`` (float): Coordinates of the sphere center (default: 0.5).
	    - ``R`` (float): Radius of the sphere (default: 0.3).
	    - ``l`` (float): Characteristic mesh size/element length (default: 0.01).
	    - ``mesh_filebasename`` (str): Prefix for generated files (default: "mesh").
	:return: A tuple containing:
	    - ``mesh`` (dolfin.Mesh): The generated FEniCS mesh.
	    - ``boundaries_mf`` (dolfin.MeshFunction): Surface markers (ID 1 for the sphere surface).
	    - ``S_id`` (int): The integer ID used for the sphere surface (always 1).
	"""
	X0 = params.get("X0", 0.5)
	Y0 = params.get("Y0", 0.5)
	Z0 = params.get("Z0", 0.5)
	R = params.get("R", 0.3)
	l = params.get("l", 0.01)

	mesh_filebasename = params.get("mesh_filebasename", "mesh")

	################################################################### Mesh ###

	gmsh.initialize()
	gmsh.clear()
	factory = gmsh.model.occ

	sp = factory.addSphere(xc=X0, yc=Y0, zc=Z0, radius=R)

	factory.synchronize()

	ps = gmsh.model.addPhysicalGroup(dim=3, tags=[sp])

	mesh_gmsh = gmsh.model.mesh

	mesh_gmsh.setSize(gmsh.model.getEntities(0), l)
	mesh_gmsh.generate(dim=3)

	gmsh.write(mesh_filebasename + ".vtk")

	gmsh.finalize()

	mesh_meshio = meshio.read(mesh_filebasename + ".vtk")

	meshio.write(mesh_filebasename + ".xdmf", mesh_meshio)

	mesh = dolfin.Mesh()
	dolfin.XDMFFile(mesh_filebasename + ".xdmf").read(mesh)

	dolfin.File(mesh_filebasename + ".xml") << mesh

	############################################################# Boundaries ###

	boundaries_mf = dolfin.MeshFunction(value_type="size_t", mesh=mesh, dim=1)

	boundaries_mf.set_all(0)

	S_sd = dolfin.AutoSubDomain(
		lambda x, on_boundary: (
			on_boundary and dolfin.near((x[0] - X0) ** 2 + (x[1] - Y0) ** 2 + (x[2] - Z0) ** 2, R**2, eps=1e-3)
		)
	)

	S_id = 1
	S_sd.mark(boundaries_mf, S_id)

	# dolfin.XDMFFile(mesh_filebasename+"-boundaries.xdmf").write(boundaries_mf)

	################################################################# Return ###

	return mesh, boundaries_mf, S_id
