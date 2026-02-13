# coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

"""summary"""

import dolfin
import numpy

################################################################################


class TimeVaryingConstant:
	r"""A wrapper for dolfin.Constant that evolves linearly over a time step.

	This class manages a FEniCS constant whose value needs to change during
	the simulation (e.g., ramping up a load from 0 to 100 over a load step).
	It supports both scalar and vector constants.



	**Mechanism:**
	The value at any normalized time step :math:`\tau \in [0, 1]` is computed as:

	.. math::
	    v(\tau) = (1 - \tau) v_{ini} + \tau v_{fin}

	It provides methods to update the underlying ``dolfin.Constant`` automatically
	when driven by a time integrator.

	:param val: A single value (if the constant is fixed).
	:param val_ini: The initial value at the start of the step.
	:param val_fin: The final value at the end of the step.
	"""

	def __init__(self, val=None, val_ini=None, val_fin=None):
		"""Initializes the TimeVaryingConstant."""
		if (val is not None) and (val_ini is None) and (val_fin is None):
			val_ini = val
			val_fin = val
		elif (val is None) and (val_ini is not None) and (val_fin is not None):
			pass
		else:
			assert 0, "Must provide val or val_ini & val_fin. Aborting."

		assert type(val_ini) in (int, float, list, numpy.ndarray), (
			"type(val_ini) (=" + str(type(val_ini)) + ") should be int, float, list or numpy.ndarray. Aborting."
		)
		if type(val_ini) in (int, float):
			assert type(val_fin) in (int, float), (
				"type(val_fin) (=" + str(type(val_fin)) + ") should be int or float. Aborting."
			)
			self.val_ini = numpy.array([val_ini], dtype=float)
			self.val_fin = numpy.array([val_fin], dtype=float)
			self.val_cur = numpy.array([val_ini], dtype=float)
			self.val_old = numpy.array([val_ini], dtype=float)
			self.set_value = self.set_value_sca
		elif type(val_ini) in (list, numpy.ndarray):
			assert type(val_fin) in (list, numpy.ndarray), (
				"type(val_fin) (=" + str(type(val_fin)) + ") should be list or numpy.ndarray. Aborting."
			)
			self.val_ini = numpy.array(val_ini, dtype=float)
			self.val_fin = numpy.array(val_fin, dtype=float)
			self.val_cur = numpy.array(val_ini, dtype=float)
			self.val_old = numpy.array(val_ini, dtype=float)
			self.set_value = self.set_value_vec
		self.val = dolfin.Constant(val_ini)
		# print("ini", self.val_ini)
		# print("fin", self.val_fin)

	def set_value_sca(self, val):
		"""Updates the value for a scalar constant."""
		if type(val) in (int, float):
			self.val.assign(dolfin.Constant(val))
		elif type(val) in (list, numpy.ndarray):
			self.val.assign(dolfin.Constant(val[0]))

	def set_value_vec(self, val):
		"""Updates the value for a vector constant."""
		# print(val)
		self.val.assign(dolfin.Constant(val))

	def set_value_at_t_step(self, t_step):
		r"""Interpolates and sets the value based on the normalized step time.

		:param t_step: Normalized time :math:`\tau \in [0,1]`.
		"""
		# print("set_value_at_t_step")
		# print("t_step", t_step)
		# print("ini", self.val_ini)
		# print("fin", self.val_fin)
		# print("cur", self.val_ini * (1. - t_step) + self.val_fin * t_step)
		self.set_value(self.val_ini * (1.0 - t_step) + self.val_fin * t_step)

	def set_dvalue_at_t_step(self, t_step):
		r"""Sets the value to the *increment* :math:`v(\tau) - v_{old}`.

		This is useful for incremental formulations where the unknown is the increment.
		"""
		# print("set_dvalue_at_t_step")
		self.val_old[:] = self.val_cur[:]
		# print("old", self.val_old)
		# print("t_step", t_step)
		# print("ini", self.val_ini)
		# print("fin", self.val_fin)
		self.val_cur[:] = self.val_ini * (1.0 - t_step) + self.val_fin * t_step
		# print("cur", self.val_cur)
		# print("dvalue", self.val_cur - self.val_old)
		# print(self.val.str(1))
		self.set_value(self.val_cur - self.val_old)
		# print(self.val.str(1))

	def restore_old_value(self):
		"""Restores the internal state to the previous value.

		Used when a time step fails and needs to be retried.
		"""
		self.val_cur[:] = self.val_old[:]

	def homogenize(self):
		"""Sets the value to zero (homogeneous condition)."""
		# print("homogenize")
		# print(self.val.str(1))
		self.set_value(0 * self.val_ini)
		# print(self.val.str(1))
