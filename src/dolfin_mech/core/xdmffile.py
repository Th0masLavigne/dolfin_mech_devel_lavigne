# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""Utility for streamlined XDMF output management."""

import dolfin

################################################################################


class XDMFFile:
	"""Wrapper for dolfin.XDMFFile to manage multi-field output.

	This class simplifies the process of writing multiple FEniCS functions to
	a single XDMF file at each time step. It configures standard settings for
	performance and file size optimization.



	**Configuration:**
	- ``flush_output=True``: Ensures data is written to disk immediately (useful for monitoring).
	- ``functions_share_mesh=True``: Optimizes file size by storing the mesh topology only once for all functions (assuming a fixed mesh).

	:param filename: Path to the output file (e.g., "results.xdmf").
	:type filename: str
	:param functions: List of FEniCS Function objects to be saved.
	:type functions: list
	"""

	def __init__(self, filename, functions):
		"""Initializes the XDMFFile wrapper."""
		self.xdmf_file = dolfin.XDMFFile(filename)
		self.xdmf_file.parameters["flush_output"] = True
		self.xdmf_file.parameters["functions_share_mesh"] = True
		# self.xdmf_file.parameters["rewrite_function_mesh"] = False

		self.functions = functions

	def close(self):
		"""Closes the underlying XDMF file handle."""
		self.xdmf_file.close()

	def write(self, time=0):
		"""Writes the current state of all registered functions to the file.

		Iterates through the list of functions provided at initialization and
		appends their current values to the XDMF file, associated with the
		given simulation time.

		:param time: The current simulation time.
		:type time: float
		"""
		for function in self.functions:
			self.xdmf_file.write(function, float(time))
