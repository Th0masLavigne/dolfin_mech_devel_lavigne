# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the run_HeartSlice_Mesh utility.

Automates the generation of a
2D annular mesh representing a ventricular cross-section. It integrates Gmsh
for triangulation and provides pre-tagged FEniCS MeshFunctions for endocardial
and epicardial boundaries, as well as specific anatomical pinning points.
"""

import dolfin
import gmsh
import meshio

################################################################################


def HeartSlice_Mesh(params: dict = {}):
	r"""Generates a 2D annular mesh representing a heart slice using Gmsh.

	This function creates a mesh for a simplified 2D cross-section of a ventricle
	(an annulus). It handles geometry creation, meshing, conversion to FEniCS
	format, and the creation of boundary/point markers.



	**Geometry:**
	Defined by a center :math:`(X_0, Y_0)`, an inner radius :math:`R_i` (endocardium),
	and an outer radius :math:`R_e` (epicardium).

	**Markers:**

	- **Boundaries**:

	    - ID 1 (:math:`S_i`): Inner boundary (Endocardium).
	    - ID 2 (:math:`S_e`): Outer boundary (Epicardium).

	- **Points** (on the inner boundary :math:`R_i`):

	    - ID 1: Right (:math:`0` rad).
	    - ID 2: Top (:math:`\pi/2` rad).
	    - ID 3: Left (:math:`\pi` rad).
	    - ID 4: Bottom (:math:`3\pi/2` rad).

	:param params: Dictionary of mesh parameters:
	    - ``X0``, ``Y0`` (float): Center coordinates (default: 0.5, 0.5).
	    - ``Ri`` (float): Inner radius (default: 0.2).
	    - ``Re`` (float): Outer radius (default: 0.4).
	    - ``l`` (float): Characteristic element size (default: 0.1).
	    - ``mesh_filebasename`` (str): Output filename prefix (default: "mesh").
	:return: A tuple containing:
	    - ``mesh`` (dolfin.Mesh): The generated mesh.
	    - ``boundaries_mf`` (dolfin.MeshFunction): Boundary markers.
	    - ``Si_id`` (int): ID for inner boundary (1).
	    - ``Se_id`` (int): ID for outer boundary (2).
	    - ``points_mf`` (dolfin.MeshFunction): Vertex markers.
	    - ``x*_sd`` (dolfin.AutoSubDomain): Subdomains for the 4 cardinal points on the inner ring.
	"""
	X0 = params.get("X0", 0.5)
	Y0 = params.get("Y0", 0.5)
	Ri = params.get("Ri", 0.2)
	Re = params.get("Re", 0.4)
	l = params.get("l", 0.1)

	mesh_filebasename = params.get("mesh_filebasename", "mesh")

	################################################################### Mesh ###

	gmsh.initialize()
	gmsh.clear()
	factory = gmsh.model.geo

	p0 = factory.addPoint(x=X0, y=Y0, z=0, meshSize=l)
	p11 = factory.addPoint(x=X0 + Ri, y=Y0, z=0, meshSize=l)
	p12 = factory.addPoint(x=X0, y=Y0 + Ri, z=0, meshSize=l)
	p13 = factory.addPoint(x=X0 - Ri, y=Y0, z=0, meshSize=l)
	p14 = factory.addPoint(x=X0, y=Y0 - Ri, z=0, meshSize=l)
	p21 = factory.addPoint(x=X0 + Re, y=Y0, z=0, meshSize=l)
	p22 = factory.addPoint(x=X0, y=Y0 + Re, z=0, meshSize=l)
	p23 = factory.addPoint(x=X0 - Re, y=Y0, z=0, meshSize=l)
	p24 = factory.addPoint(x=X0, y=Y0 - Re, z=0, meshSize=l)

	l11 = factory.addCircleArc(p11, p0, p12)
	l12 = factory.addCircleArc(p12, p0, p13)
	l13 = factory.addCircleArc(p13, p0, p14)
	l14 = factory.addCircleArc(p14, p0, p11)
	l21 = factory.addCircleArc(p21, p0, p22)
	l22 = factory.addCircleArc(p22, p0, p23)
	l23 = factory.addCircleArc(p23, p0, p24)
	l24 = factory.addCircleArc(p24, p0, p21)

	cl = factory.addCurveLoop([l11, l12, l13, l14, l21, l22, l23, l24])

	s = factory.addPlaneSurface([cl])

	factory.synchronize()

	ps = gmsh.model.addPhysicalGroup(dim=2, tags=[s])

	mesh_gmsh = gmsh.model.mesh

	mesh_gmsh.generate(dim=2)

	gmsh.write(mesh_filebasename + ".vtk")

	gmsh.finalize()

	mesh_meshio = meshio.read(mesh_filebasename + ".vtk")

	mesh_meshio.points = mesh_meshio.points[:, :2]

	meshio.write(mesh_filebasename + ".xdmf", mesh_meshio)

	mesh = dolfin.Mesh()
	dolfin.XDMFFile(mesh_filebasename + ".xdmf").read(mesh)

	dolfin.File(mesh_filebasename + ".xml") << mesh

	############################################################# Boundaries ###

	boundaries_mf = dolfin.MeshFunction(value_type="size_t", mesh=mesh, dim=1)

	boundaries_mf.set_all(0)

	Si_sd = dolfin.AutoSubDomain(
		lambda x, on_boundary: on_boundary and dolfin.near((x[0] - X0) ** 2 + (x[1] - Y0) ** 2, Ri**2, eps=1e-3)
	)

	Se_sd = dolfin.AutoSubDomain(
		lambda x, on_boundary: on_boundary and dolfin.near((x[0] - X0) ** 2 + (x[1] - Y0) ** 2, Re**2, eps=1e-3)
	)

	Si_id = 1
	Si_sd.mark(boundaries_mf, Si_id)
	Se_id = 2
	Se_sd.mark(boundaries_mf, Se_id)

	# dolfin.XDMFFile(mesh_filebasename+"-boundaries.xdmf").write(boundaries_mf)

	################################################################# Points ###

	points_mf = dolfin.MeshFunction(value_type="size_t", mesh=mesh, dim=0)

	points_mf.set_all(0)

	x1 = [X0 + Ri, Y0]
	x1_sd = dolfin.AutoSubDomain(
		lambda x, on_boundary: dolfin.near(x[0], x1[0], eps=1e-3) and dolfin.near(x[1], x1[1], eps=1e-3)
	)
	x2 = [X0, Y0 + Ri]
	x2_sd = dolfin.AutoSubDomain(
		lambda x, on_boundary: dolfin.near(x[0], x2[0], eps=1e-3) and dolfin.near(x[1], x2[1], eps=1e-3)
	)
	x3 = [X0 - Ri, Y0]
	x3_sd = dolfin.AutoSubDomain(
		lambda x, on_boundary: dolfin.near(x[0], x3[0], eps=1e-3) and dolfin.near(x[1], x3[1], eps=1e-3)
	)
	x4 = [X0, Y0 - Ri]
	x4_sd = dolfin.AutoSubDomain(
		lambda x, on_boundary: dolfin.near(x[0], x4[0], eps=1e-3) and dolfin.near(x[1], x4[1], eps=1e-3)
	)

	x1_id = 1
	x1_sd.mark(points_mf, x1_id)
	x2_id = 2
	x2_sd.mark(points_mf, x2_id)
	x3_id = 3
	x3_sd.mark(points_mf, x3_id)
	x4_id = 4
	x4_sd.mark(points_mf, x4_id)

	# dolfin.XDMFFile(mesh_filebasename+"-points.xdmf").write(points_mf)

	return mesh, boundaries_mf, Si_id, Se_id, points_mf, x1_sd, x2_sd, x3_sd, x4_sd
