# coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2024                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2025                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Defines the HollowBox_Mesh utility.

Automates the creation of a periodic
unit cell (RVE) containing a centered void. It implements geometric boolean
subtractions and enforces matching node distributions on opposite faces to
facilitate periodic homogenization in multiscale mechanics.
"""

import dolfin
import gmsh
import meshio
import numpy

################################################################################


def setPeriodic(dim, coord, xmin, ymin, zmin, xmax, ymax, zmax, e=1e-6):
	# From https://gitlab.onelab.info/gmsh/gmsh/-/issues/744
	r"""Applies periodic meshing constraints to matching entities on opposite faces of a bounding box.

	This function identifies geometric entities (curves in 2D, surfaces in 3D) on opposite
	sides of the domain and enforces a periodic mesh mapping between them in Gmsh.
	This ensures that nodes on the master face :math:`\Gamma_{master}` match exactly with
	nodes on the slave face :math:`\Gamma_{slave}`.



	:param dim: The topological dimension of the entities to match (2 for curves/edges, 3 for surfaces).
	:param coord: The index of the coordinate direction for periodicity (0 for X, 1 for Y, 2 for Z).
	:param xmin, ymin, zmin: Coordinates of the minimum corner of the bounding box.
	:param xmax, ymax, zmax: Coordinates of the maximum corner of the bounding box.
	:param e: Geometric tolerance for entity detection (default: 1e-6).
	"""
	dx = (xmax - xmin) if (coord == 0) else 0.0
	dy = (ymax - ymin) if (coord == 1) else 0.0
	dz = (zmax - zmin) if (coord == 2) else 0.0
	d = max(dx, dy, dz)
	e *= d

	smin = gmsh.model.getEntitiesInBoundingBox(
		xmin - e, ymin - e, zmin - e, xmax - dx + e, ymax - dy + e, zmax - dz + e, dim - 1
	)
	# print ("smin:",smin)
	for i in smin:
		bb = gmsh.model.getBoundingBox(*i)
		bbe = [bb[0] + dx, bb[1] + dy, bb[2] + dz, bb[3] + dx, bb[4] + dy, bb[5] + dz]
		smax = gmsh.model.getEntitiesInBoundingBox(
			bbe[0] - e, bbe[1] - e, bbe[2] - e, bbe[3] + e, bbe[4] + e, bbe[5] + e, dim - 1
		)
		# print ("smax:",smax)
		for j in smax:
			bb2 = gmsh.model.getBoundingBox(*j)
			bb2e = [bb2[0] - dx, bb2[1] - dy, bb2[2] - dz, bb2[3] - dx, bb2[4] - dy, bb2[5] - dz]
			if numpy.linalg.norm(numpy.asarray(bb2e) - numpy.asarray(bb)) < e:
				gmsh.model.mesh.setPeriodic(
					dim - 1, [j[1]], [i[1]], [1, 0, 0, dx, 0, 1, 0, dy, 0, 0, 1, dz, 0, 0, 0, 1]
				)


################################################################################


def HollowBox_Mesh(params: dict = {}):
	"""Generates a periodic mesh for a "Hollow Box" Representative Volume Element (RVE).

	This function creates a unit cell with a central void by boolean subtraction.
	To create a centered void in a periodic cell :math:`[0,1]^d`, it subtracts
	parts of a sphere (or circle) from the 4 corners (2D) or 8 corners (3D).
	Due to periodicity, these corner cutouts form a single central hole in the
	tiled lattice.



	**Steps:**
	1.  **Geometry**: Creates a base rectangle/box and subtracts disks/spheres at corners.
	2.  **Periodicity**: Calls ``setPeriodic`` for each spatial direction to link opposite boundaries.
	3.  **Meshing**: Generates the mesh and exports it to XDMF format.

	:param params: Dictionary of parameters:
	    - ``dim`` (int): Dimension (2 or 3).
	    - ``xmin``, ``xmax``, etc.: Bounding box coordinates.
	    - ``r0`` (float): Radius of the void.
	    - ``l`` (float): Characteristic element size.
	    - ``mesh_filebasename`` (str): Output filename.
	:return: The generated FEniCS mesh.
	"""
	dim = params.get("dim")
	assert dim in (2, 3)
	xmin = params.get("xmin", 0.0)
	ymin = params.get("ymin", 0.0)
	zmin = params.get("zmin", 0.0)
	xmax = params.get("xmax", 1.0)
	ymax = params.get("ymax", 1.0)
	zmax = params.get("zmax", 1.0)
	xshift = params.get("xshift", 0.0)
	yshift = params.get("yshift", 0.0)
	zshift = params.get("zshift", 0.0)
	r0 = params.get("r0", 0.2)
	l = params.get("l", 0.1)

	mesh_filebasename = params.get("mesh_filebasename", "mesh")

	################################################################### Mesh ###

	gmsh.initialize()

	if dim == 2:
		box_tag = 1
		hole1_tag = 2
		hole2_tag = 3
		hole3_tag = 4
		hole4_tag = 5
		rve_tag = 6

		gmsh.model.occ.addRectangle(
			x=xmin + xshift, y=ymin + yshift, z=0.0, dx=xmax - xmin, dy=ymax - ymin, tag=box_tag
		)
		gmsh.model.occ.addDisk(xc=xmin, yc=ymin, zc=0.0, rx=r0, ry=r0, tag=hole1_tag)
		gmsh.model.occ.addDisk(xc=xmax, yc=ymin, zc=0.0, rx=r0, ry=r0, tag=hole2_tag)
		gmsh.model.occ.addDisk(xc=xmax, yc=ymax, zc=0.0, rx=r0, ry=r0, tag=hole3_tag)
		gmsh.model.occ.addDisk(xc=xmin, yc=ymax, zc=0.0, rx=r0, ry=r0, tag=hole4_tag)
		gmsh.model.occ.cut(
			objectDimTags=[(2, box_tag)],
			toolDimTags=[(2, hole1_tag), (2, hole2_tag), (2, hole3_tag), (2, hole4_tag)],
			tag=rve_tag,
		)
		gmsh.model.occ.synchronize()
		gmsh.model.addPhysicalGroup(dim=2, tags=[rve_tag])
		setPeriodic(
			dim=2,
			coord=0,
			xmin=xmin + xshift,
			ymin=ymin + yshift,
			zmin=0.0,
			xmax=xmax + xshift,
			ymax=ymax + yshift,
			zmax=0.0,
		)
		setPeriodic(
			dim=2,
			coord=1,
			xmin=xmin + xshift,
			ymin=ymin + yshift,
			zmin=0.0,
			xmax=xmax + xshift,
			ymax=ymax + yshift,
			zmax=0.0,
		)
		gmsh.model.mesh.setSize(dimTags=gmsh.model.getEntities(0), size=l)
		gmsh.model.mesh.generate(dim=2)
	if dim == 3:
		box_tag = 1
		hole1_tag = 2
		hole2_tag = 3
		hole3_tag = 4
		hole4_tag = 5
		hole5_tag = 6
		hole6_tag = 7
		hole7_tag = 8
		hole8_tag = 9
		rve_tag = 10

		gmsh.model.occ.addBox(
			x=xmin + xshift,
			y=ymin + yshift,
			z=zmin + zshift,
			dx=xmax - xmin,
			dy=ymax - ymin,
			dz=zmax - zmin,
			tag=box_tag,
		)
		gmsh.model.occ.addSphere(xc=xmin, yc=ymin, zc=zmin, radius=r0, tag=hole1_tag)
		gmsh.model.occ.addSphere(xc=xmax, yc=ymin, zc=zmin, radius=r0, tag=hole2_tag)
		gmsh.model.occ.addSphere(xc=xmax, yc=ymax, zc=zmin, radius=r0, tag=hole3_tag)
		gmsh.model.occ.addSphere(xc=xmin, yc=ymax, zc=zmin, radius=r0, tag=hole4_tag)
		gmsh.model.occ.addSphere(xc=xmin, yc=ymin, zc=zmax, radius=r0, tag=hole5_tag)
		gmsh.model.occ.addSphere(xc=xmax, yc=ymin, zc=zmax, radius=r0, tag=hole6_tag)
		gmsh.model.occ.addSphere(xc=xmax, yc=ymax, zc=zmax, radius=r0, tag=hole7_tag)
		gmsh.model.occ.addSphere(xc=xmin, yc=ymax, zc=zmax, radius=r0, tag=hole8_tag)
		gmsh.model.occ.cut(
			objectDimTags=[(3, box_tag)],
			toolDimTags=[
				(3, hole1_tag),
				(3, hole2_tag),
				(3, hole3_tag),
				(3, hole4_tag),
				(3, hole5_tag),
				(3, hole6_tag),
				(3, hole7_tag),
				(3, hole8_tag),
			],
			tag=rve_tag,
		)
		gmsh.model.occ.synchronize()
		gmsh.model.addPhysicalGroup(dim=3, tags=[rve_tag])
		setPeriodic(
			dim=3,
			coord=0,
			xmin=xmin + xshift,
			ymin=ymin + yshift,
			zmin=zmin + zshift,
			xmax=xmax + xshift,
			ymax=ymax + yshift,
			zmax=zmax + zshift,
		)
		setPeriodic(
			dim=3,
			coord=1,
			xmin=xmin + xshift,
			ymin=ymin + yshift,
			zmin=zmin + zshift,
			xmax=xmax + xshift,
			ymax=ymax + yshift,
			zmax=zmax + zshift,
		)
		setPeriodic(
			dim=3,
			coord=2,
			xmin=xmin + xshift,
			ymin=ymin + yshift,
			zmin=zmin + zshift,
			xmax=xmax + xshift,
			ymax=ymax + yshift,
			zmax=zmax + zshift,
		)
		gmsh.model.mesh.setSize(dimTags=gmsh.model.getEntities(0), size=l)
		gmsh.model.mesh.generate(dim=3)

	gmsh.write(mesh_filebasename + ".vtk")
	gmsh.finalize()

	mesh = meshio.read(mesh_filebasename + ".vtk")
	if dim == 2:
		mesh.points = mesh.points[:, :2]
	meshio.write(mesh_filebasename + ".xdmf", mesh)

	mesh = dolfin.Mesh()
	dolfin.XDMFFile(mesh_filebasename + ".xdmf").read(mesh)

	return mesh
