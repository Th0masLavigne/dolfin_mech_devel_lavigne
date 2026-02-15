# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the RivlinCube_Mesh utility.

Generates structured rectangular
or box meshes in FEniCS. It automates the discretization process based on
user-defined characteristic lengths and produces pre-tagged MeshFunctions
for all planar faces to facilitate easy application of boundary conditions.
"""

import dolfin

################################################################################


def RivlinCube_Mesh(dim: int = 3, params: dict = {}):
	"""Generates a structured rectangular (2D) or box (3D) mesh using FEniCS.

	This function creates the geometry for the Rivlin Cube benchmark. It supports
	arbitrary bounding box dimensions and mesh refinement levels.
	Crucially, it identifies and marks all 4 (2D) or 6 (3D) planar boundaries,
	enabling precise application of boundary conditions like symmetry planes
	or surface tractions.



	**Boundaries:**
	- **xmin** (ID 1): Left face (:math:`x = X_0`).
	- **xmax** (ID 2): Right face (:math:`x = X_1`).
	- **ymin** (ID 3): Bottom face (:math:`y = Y_0`).
	- **ymax** (ID 4): Top face (:math:`y = Y_1`).
	- **zmin** (ID 5): Back face (:math:`z = Z_0`) *[3D only]*.
	- **zmax** (ID 6): Front face (:math:`z = Z_1`) *[3D only]*.

	:param dim: Mesh dimension (2 or 3).
	:type dim: int
	:param params: Dictionary of mesh parameters:
	    - ``X0``, ``X1``, ``Y0``, ``Y1``, ``Z0``, ``Z1`` (float): Bounding box coordinates.
	    - ``l`` (float): Characteristic element size (determines number of divisions).
	    - ``refine`` (bool): If True, performs one uniform refinement step.
	    - ``mesh_filebasename`` (str): Output filename prefix (default: "mesh").
	:type params: dict
	:return: A tuple containing:
	    - ``mesh`` (dolfin.Mesh): The generated mesh.
	    - ``boundaries_mf`` (dolfin.MeshFunction): Surface markers.
	    - ``xmin_id`` ... ``zmax_id`` (int): The integer IDs for each face.
	"""
	X0 = params.get("X0", 0.0)
	X1 = params.get("X1", 1.0)
	Y0 = params.get("Y0", 0.0)
	Y1 = params.get("Y1", 1.0)
	if dim == 3:
		Z0 = params.get("Z0", 0.0)
	if dim == 3:
		Z1 = params.get("Z1", 1.0)
	l = params.get("l", 1.0)

	mesh_filebasename = params.get("mesh_filebasename", "mesh")

	LX = X1 - X0
	LY = Y1 - Y0
	if dim == 3:
		LZ = Z1 - Z0

	NX = int(LX / l)
	NY = int(LY / l)
	if dim == 3:
		NZ = int(LZ / l)

	if dim == 2:
		mesh = dolfin.RectangleMesh(dolfin.Point(X0, Y0, 0.0), dolfin.Point(X1, Y1, 0.0), NX, NY, "crossed")
	elif dim == 3:
		mesh = dolfin.BoxMesh(dolfin.Point(X0, Y0, Z0), dolfin.Point(X1, Y1, Z1), NX, NY, NZ)

	if params.get("refine", False) == True:
		mesh = dolfin.refine(mesh)

	xdmf_file_mesh = dolfin.XDMFFile(mesh_filebasename + ".xdmf")
	xdmf_file_mesh.write(mesh)
	xdmf_file_mesh.close()

	dolfin.File(mesh_filebasename + ".xml") << mesh

	################################################## Subdomains & Measures ###

	xmin_sd = dolfin.CompiledSubDomain("near(x[0], x0) && on_boundary", x0=X0)
	xmax_sd = dolfin.CompiledSubDomain("near(x[0], x0) && on_boundary", x0=X1)
	ymin_sd = dolfin.CompiledSubDomain("near(x[1], x0) && on_boundary", x0=Y0)
	ymax_sd = dolfin.CompiledSubDomain("near(x[1], x0) && on_boundary", x0=Y1)
	if dim == 3:
		zmin_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=Z0)
	if dim == 3:
		zmax_sd = dolfin.CompiledSubDomain("near(x[2], x0) && on_boundary", x0=Z1)

	xmin_id = 1
	xmax_id = 2
	ymin_id = 3
	ymax_id = 4
	if dim == 3:
		zmin_id = 5
	if dim == 3:
		zmax_id = 6

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

	# xdmf_file_boundaries = dolfin.XDMFFile("boundaries.xdmf")
	# xdmf_file_boundaries.write(boundaries_mf)
	# xdmf_file_boundaries.close()

	if dim == 2:
		return mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id
	elif dim == 3:
		return mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id
