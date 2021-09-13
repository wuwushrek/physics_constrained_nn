# -*- coding: utf-8 -*-
# __all__ = ["inv_mass_prod_vector", "initialize_problem_var", "quaternion_mapping", "compute_qpos_der"]

import sys
import os 

sys.path.append(os.path.dirname(os.path.realpath(__file__))) # Append this reposirtory to path to be found as module
# To import and compile c++ codes from within python
import cppimport
# Register the CPU XLA custom calls
mujoco_prim = cppimport.imp("mujoco_prim")

from functools import partial
import numpy as np

from jax import jit
from jax import numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes, lax
from jax.interpreters import ad, batching, xla
from jax.abstract_arrays import ShapedArray

for _name, _value in mujoco_prim.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

import functools
import traceback

_indentation = 0
def _trace(msg=None):
    """Print a message at current indentation."""
    # if msg is not None:
    #     print("  " * _indentation + msg)

def _trace_indent(msg=None):
    """Print a message and then indent the rest."""
    global _indentation
    _trace(msg)
    _indentation = 1 + _indentation

def _trace_unindent(msg=None):
    """Unindent then print a message."""
    global _indentation
    _indentation = _indentation - 1
    _trace(msg)

def trace(name):
  """A decorator for functions to trace arguments and results."""

  def trace_func(func):  # pylint: disable=missing-docstring
    def pp(v):
        """Print certain values more succinctly"""
        vtype = str(type(v))
        if "jax.lib.xla_bridge._JaxComputationBuilder" in vtype:
            return "<JaxComputationBuilder>"
        elif "jaxlib.xla_extension.XlaOp" in vtype:
            return "<XlaOp at 0x{:x}>".format(id(v))
        elif ("partial_eval.JaxprTracer" in vtype or
              "batching.BatchTracer" in vtype or
              "ad.JVPTracer" in vtype):
            return "Traced<{}>".format(v.aval)
        elif isinstance(v, tuple):
            return "({})".format(pp_values(v))
        else:
            return str(v)
    def pp_values(args):
        return ", ".join([pp(arg) for arg in args])
    
    @functools.wraps(func)
    def func_wrapper(*args):
      _trace_indent("call {}({})".format(name, pp_values(args)))
      res = func(*args)
      _trace_unindent("|<- {} = {}".format(name, pp(res)))
      return res

    return func_wrapper

  return trace_func

class expectNotImplementedError(object):
  """Context manager to check for NotImplementedError."""
  def __enter__(self): pass
  def __exit__(self, type, value, tb):
    global _indentation
    _indentation = 0
    if type is NotImplementedError:
      print("\nFound expected exception:")
      traceback.print_exc(limit=3)
      return True
    elif type is None:  # No exception
      assert False, "Expected NotImplementedError"
    else:
      return False

xops = xla_client.ops
EPS_DIFF = 1e-6 # The epsilon value used in the forward differentiation of the function

# This function exposes the C++ code to initiliaze the model given the xml path
def initialize_problem_var(modelpath, eps_diff=EPS_DIFF):
    """ This function initialize the model parameters. It must always be called before any of the
        function below
        :param modelpath : Path the the xml model file containing the parameters of the model of interest
    """
    status = mujoco_prim.init_model(modelpath, 500, eps_diff)
    print("Loading status : ", status)

# This function exposes the C++ code to compute information on the quaternion variables of the system
def quaternion_mapping(nq):
    """ This function returns the quaternion mappping as a tuple of jax array
        :param nq : Number of positgional coordinate of the model
    """
    mL1, mL2 = list(), list()
    for i in range(nq):
        r1, r2 = mujoco_prim.joint_mapping(i)
        assert r2 >= 0, 'Model is not well defined --> Joint with no address for velocity'
        mL1.append(r1)
        mL2.append(r2)
    return (tuple(mL1), tuple(mL2))

# This function computes \dot{q} = f(q,w) if q is a quaternion and \dot{q} = w if not a quaternion
# @partial(jit, static_argnums=(2,3))
def compute_qpos_der(qpos, qvel, quat_arr):
    """ This function computes \dot{q} = f(q,w) if q is a quaternion and \dot{q} = w if not a quaternion
        :param qpos : The positional coordinate of the model can be batches
        :param qvel : The velocity coordinate of the model can be batches
        :param quat_arr : A tuple (arr1,arr2) of jax array specifying which coordinates are quaternion or not
    """
    quat_ind, vel_ind = quat_arr
    count_n = 0
    while count_n < qpos.shape[-1]:
        quat_addr, dof_addr = quat_ind[count_n], vel_ind[count_n]
        if quat_addr >= 0:
            a1, b1, c1, d1 = jnp.take(qpos, quat_addr, axis=-1), jnp.take(qpos, quat_addr+1, axis=-1), jnp.take(qpos, quat_addr+2, axis=-1), jnp.take(qpos, quat_addr+3, axis=-1)
            a2, b2, c2, d2 = 0, jnp.take(qvel, dof_addr, axis=-1), jnp.take(qvel, dof_addr+1, axis=-1), jnp.take(qvel, dof_addr+2, axis=-1)
            q0 = 0.5*(a1*a2 - b1*b2 - c1*c2 - d1*d2);
            q1 = 0.5*(a1*b2 + b1*a2 + c1*d2 - d1*c2);
            q2 = 0.5*(a1*c2 - b1*d2 + c1*a2 + d1*b2);
            q3 = 0.5*(a1*d2 + b1*c2 - c1*b2 + d1*a2);
            res_val = jnp.hstack((q0,q1,q2,q3)) if count_n == 0 else jnp.hstack((res_val,q0,q1,q2,q3))
            count_n += 4
        else:
            res_val = jnp.take(qvel, dof_addr, axis=-1) if count_n == 0 else jnp.hstack((res_val, jnp.take(qvel, dof_addr, axis=-1)))
            count_n += 1
    return res_val


# **********************************************
# *  UTILITY FUNCTIONS FOR DIMENSION CHECKING  *
# **********************************************

@trace('check_dim_imputs')
def check_dim_imputs(f_in, xla_c=None):
    """ Given inputs of a primitive ( and if available, the xla compiler translation ruler), check if the inputs dimensions 
        has the same types, dimensions and return structures needed for abstract and translation interpretation
    """
    shape_f_in = [ v_in.shape if xla_c is None else xla_c.get_shape(v_in).dimensions() for v_in in f_in]
    type_f_in = [ dtypes.canonicalize_dtype(v_in.dtype) if xla_c is None else xla_c.get_shape(v_in).element_type() for v_in in f_in]
    # Pick a shape in the list
    shape_0 = shape_f_in[0]
    type_0 = type_f_in[0]
    assert len(shape_0) <= 2, 'The C++ implemetation works only with (batch_size, dim) or (dim,)'
    assert all(shape_elem[:-1] == shape_0[:-1] for shape_elem in shape_f_in), 'No matches in batch dimensions of {}'.format(shape_f_in)
    assert all(type_elem == type_0 for type_elem in type_f_in), 'No matches in types of {}'.format(type_f_in)
    if xla_c is not None:
        assert type_0 == np.float32 or type_0 == np.float64, 'The types of the inputs should be either float64 or float32. Got {}'.format(type_0)
        dims_spec = [xla_client.Shape.array_shape(np.dtype(np.int64), (), ()), 
                    *[xla_client.Shape.array_shape(np.dtype(type_0), dims_v, tuple(range(len(dims_v) - 1, -1, -1))) for dims_v in shape_f_in]
                    ]
        return type_0, xops.ConstantLiteral(xla_c, np.prod(shape_0[:-1]).astype(np.int64)), tuple(dims_spec)
    else :
        return tuple(ShapedArray(s, type_0) for s in shape_f_in)

@trace('check_batch_inputs')
def check_batch_inputs(args, axes):
    """ This function checks the batching dimension and which inputs in args are actually batched.
        We assume the batch data is always only following the first axe (to be modified later)
    """
    assert all( ax is None or ax == 0 for ax in axes ), 'Batching is allowed only on axes 0 : {}'.format(axes)
    batch_sizes = [ inp.shape[0] for inp, ax in zip(args, axes) if ax is not None ]
    size_batch = batch_sizes[0]
    assert all(b_size == size_batch for b_size in batch_sizes), 'The batch dimension mustS match'
    return tuple(lax.broadcast(q, (size_batch,)) if axes_q is None else q for q, axes_q in zip(args, axes)), size_batch


# *********************************
# *     SUPPORT FOR M(q)^-1 v     *
# *********************************

@trace('iM_product_vect')
def iM_product_vect(q, vect):
    """ Compute M^-1(q) vect, given the current configuration position and a vector.
        :param q    : The configuration position (can be euler angle or quaternions) -> dimension should be model->nq
        :param vect : A vector of dimension model->nq : number of degrees of freedom
    """
    return _iM_product_vect_prim.bind(q, vect)

# *  SUPPORT FOR ABSTRACT COMPILATION  *
@trace('iM_product_vect_abstract')
def _iM_product_vect_abstract(q, vect):
    """ Abstract evaluation of the primitive _iM_product_vect_prim
    """
    # Check inputs dimension
    shapearray_q, shapearray_vect = check_dim_imputs((q, vect)) 
    return shapearray_vect

# *  SUPPORT FOR JIT COMPILATION  *
@trace('iM_product_vect_translation')
def _iM_product_vect_translation(c, q, vect):
    """ XLA translation of _iM_product_vect_prim
    """
    type_in, size_xla, dims_spec = check_dim_imputs((q, vect), c)
    op_name = b"iM_prod_vect_wrapper_f32" if type_in == np.float32 else b"iM_prod_vect_wrapper_f64"
    return xops.CustomCallWithLayout(c, op_name, operands=(size_xla, q, vect), 
                                        operand_shapes_with_layout=dims_spec, shape_with_layout=dims_spec[2])

# *  SUPPORT FOR BATCHING WITH VMAP  *
@trace('iM_product_vect_batch')
def _iM_product_vect_batch(args, axes):
    """ Batch rules for the primitive _iM_product_vect_prim
    """
    (q, vect), size_batch = check_batch_inputs(args, axes)
    if len(q.shape) <= 2:
        return _iM_product_vect_prim.bind(q, vect), 0
    for i in range(size_batch):
        mCompute = _iM_product_vect_prim.bind(q[i], vect[i])[None]
        batch_res = mCompute if i == 0 else jnp.vstack((batch_res, mCompute))
    return batch_res, 0

# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# Compute the product of the inverse of the mass inertia matrix and a given vector and store the result inside
_iM_product_vect_prim = core.Primitive("iM_product_vect")
_iM_product_vect_prim.def_impl(partial(xla.apply_primitive, _iM_product_vect_prim))
_iM_product_vect_prim.def_abstract_eval(_iM_product_vect_abstract)
batching.primitive_batchers[_iM_product_vect_prim] = _iM_product_vect_batch
xla.backend_specific_translations["cpu"][_iM_product_vect_prim] = _iM_product_vect_translation


# **************************************
# *     SUPPORT FOR JVP(M(q)^-1 v)     *
# **************************************

# *  SUPPORT FOR ABSTRACT COMPILATION  *
@trace('iM_product_vect_jvp_abstract')
def _iM_product_vect_jvp_abstract(q, vect, q_tan, vect_tan):
    """ Abstract rule for _iM_product_vect_jvp_prim
    """
    shapearray_q, shapearray_vect, shapearray_q_tan, shapearray_vect_tan = check_dim_imputs((q, vect, q_tan, vect_tan)) 
    return shapearray_vect

# *  SUPPORT FOR JIT COMPILATION  *
@trace('iM_product_vect_jvp_translation')
def _iM_product_vect_jvp_translation(c, q, vect, q_tan, vect_tan):
    """ XLA translation of _iM_product_vect_jvp_prim
    """
    type_in, size_xla, dims_spec = check_dim_imputs((q, vect, q_tan, vect_tan), c)
    op_name = b"iM_prod_vect_jvp_wrapper_f32" if type_in == np.float32 else b"iM_prod_vect_jvp_wrapper_f64"
    return xops.CustomCallWithLayout(c, op_name, operands=(size_xla, q, vect, q_tan, vect_tan), 
                                        operand_shapes_with_layout=dims_spec, shape_with_layout=dims_spec[2])

# *  SUPPORT FOR BATCHING WITH VMAP  *
@trace('iM_product_vect_jvp_batch')
def _iM_product_vect_jvp_batch(args, axes):
    """ Batch rule for _iM_product_vect_jvp_prim
    """
    (q, vect, q_tan, vect_tan), size_batch = check_batch_inputs(args, axes)
    if len(q.shape) <= 2:
        return _iM_product_vect_jvp_prim.bind(q, vect, q_tan, vect_tan), 0
    for i in range(size_batch):
        mCompute = _iM_product_vect_jvp_prim.bind(q[i], vect[i], q_tan[i], vect_tan[i])[None]
        jac_res = mCompute if i == 0 else jnp.vstack((jac_res, mCompute))
    return jac_res, 0

# *  SUPPORT FOR FORWARD AUTODIFF  OF iM_product_vect*
@trace('iM_product_vect_jvp_tan')
def iM_product_vect_jvp(args, tangents):
    """ JVP rule for _iM_product_vect_prim
    """
    def make_zero(tan, val):
        return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan
    q, vect = args
    q_tan, vect_tan = tangents
    q_tan_arr, vect_tan_arr = make_zero(q_tan, q), make_zero(vect_tan, vect)
    f_res = iM_product_vect(q, vect)
    derf_res = _iM_product_vect_jvp_prim.bind(q, vect, q_tan_arr, vect_tan_arr)
    return f_res, derf_res

# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
_iM_product_vect_jvp_prim = core.Primitive("iM_product_vect_jvp")
_iM_product_vect_jvp_prim.def_impl(partial(xla.apply_primitive, _iM_product_vect_jvp_prim))
_iM_product_vect_jvp_prim.def_abstract_eval(_iM_product_vect_jvp_abstract)
batching.primitive_batchers[_iM_product_vect_jvp_prim] = _iM_product_vect_jvp_batch
xla.backend_specific_translations["cpu"][_iM_product_vect_jvp_prim] = _iM_product_vect_jvp_translation
ad.primitive_jvps[_iM_product_vect_prim] = iM_product_vect_jvp # *  SUPPORT FOR FORWARD AUTODIFF  *


# **************************************
# *     SUPPORT FOR VJP(M(q)^-1 v)     *
# **************************************

# *  SUPPORT FOR ABSTRACT COMPILATION  *
@trace('iM_product_vect_vjp_abstract')
def _iM_product_vect_vjp_abstract(cotan, q, vect):
    """ Abstract evaluation of the primitive _iM_product_vect_vjp_prim
    """
    # Check inputs dimension
    shapearray_cotan, shapearray_q, shapearray_vect = check_dim_imputs((cotan, q, vect)) 
    return (shapearray_q, shapearray_vect)

# *  SUPPORT FOR JIT COMPILATION  *
@trace('iM_product_vect_vjp_translation')
def _iM_product_vect_vjp_translation(c, cotan, q, vect):
    """ XLA translation of _iM_product_vect_vjp_prim
    """
    type_in, size_xla, dims_spec = check_dim_imputs((cotan, q, vect), c)
    op_name = b"iM_prod_vect_vjp_wrapper_f32" if type_in == np.float32 else b"iM_prod_vect_vjp_wrapper_f64"
    return xops.CustomCallWithLayout(c, op_name, operands=(size_xla, cotan, q, vect), operand_shapes_with_layout=dims_spec, 
                                        shape_with_layout=xla_client.Shape.tuple_shape((dims_spec[2],dims_spec[3])))

# *  SUPPORT FOR BATCHING WITH VMAP  *
@trace('iM_product_vect_vjp_batch')
def _iM_product_vect_vjp_batch(args, axes):
    """ Batch rule for _iM_product_vect_vjp_prim
    """
    (cotan, q, vect), size_batch = check_batch_inputs(args, axes)
    if len(q.shape) <= 2:
        return _iM_product_vect_vjp_prim.bind(cotan, q, vect), (0, 0)
    for i in range(size_batch):
        cotan_q_t, cotan_vect_t = _iM_product_vect_vjp_prim.bind(cotan[i], q[i], vect[i])
        cotan_q = cotan_q_t[None] if i == 0 else jnp.vstack((cotan_q, cotan_q_t[None]))
        cotan_vect = cotan_vect_t[None] if i == 0 else jnp.vstack((cotan_vect, cotan_vect_t[None]))
    return (cotan_q, cotan_vect), (0, 0)

# *  SUPPORT FOR TRANSPOSE RULE  *
@trace('_iM_product_vect_vjp_transpose')
def iM_product_vect_vjp_transpose(ct, q, vect, q_tan, vect_tan):
    """ Transpos for the primitive _iM_product_vect_prim used inside the Jacobian Vector product of _iM_product_vect_prim
    """
    assert not ad.is_undefined_primal(q) and not ad.is_undefined_primal(vect), 'The two first arguments should be constant in vjp formulation'
    assert ad.is_undefined_primal(q_tan) or ad.is_undefined_primal(vect_tan), 'Both variables can not be undefined'
    if type(ct) is ad.Zero:
        return None, None, ad.Zero(q_tan.aval) if ad.is_undefined_primal(q_tan) else None, ad.Zero(vect_tan.aval) if ad.is_undefined_primal(q_tan) else None
    cotan_q, cotan_vect = _iM_product_vect_vjp_prim.bind(ct, q, vect)
    return None, None, cotan_q if ad.is_undefined_primal(q_tan) else None, cotan_vect if ad.is_undefined_primal(vect_tan) else None

# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
_iM_product_vect_vjp_prim = core.Primitive("iM_product_vect_vjp")
_iM_product_vect_vjp_prim.multiple_results = True
_iM_product_vect_vjp_prim.def_impl(partial(xla.apply_primitive, _iM_product_vect_vjp_prim))
_iM_product_vect_vjp_prim.def_abstract_eval(_iM_product_vect_vjp_abstract)
batching.primitive_batchers[_iM_product_vect_vjp_prim] = _iM_product_vect_vjp_batch
xla.backend_specific_translations["cpu"][_iM_product_vect_vjp_prim] = _iM_product_vect_vjp_translation
ad.primitive_transposes[_iM_product_vect_jvp_prim] = iM_product_vect_vjp_transpose # *  SUPPORT FOR TRANSPOSE RULE  *

# ********************************************
# *     SUPPORT FOR JVP(JVP((M(q)^-1 v))     *
# ********************************************

# *  SUPPORT FOR ABSTRACT COMPILATION  *
@trace('iM_product_vect_jvp_jvp_abstract')
def _iM_product_vect_jvp_jvp_abstract(q, vect, q_tan, vect_tan, q_t, vect_t, q_tan_t, vect_tan_t):
    """ Abstract rule for _iM_product_vect_jvp_jvp_prim
    """
    shapes_arr = check_dim_imputs((q, vect, q_tan, vect_tan, q_t, vect_t, q_tan_t, vect_tan_t)) 
    return shapes_arr[1]

# *  SUPPORT FOR JIT COMPILATION  *
@trace('iM_product_vect_jvp_jvp_translation')
def _iM_product_vect_jvp_jvp_translation(c, q, vect, q_tan, vect_tan, q_t, vect_t, q_tan_t, vect_tan_t):
    """ XLA translation of _iM_product_vect_jvp_jvp_prim
    """
    type_in, size_xla, dims_spec = check_dim_imputs((q, vect, q_tan, vect_tan, q_t, vect_t, q_tan_t, vect_tan_t), c)
    op_name = b"iM_prod_vect_jvp_jvp_wrapper_f32" if type_in == np.float32 else b"iM_prod_vect_jvp_jvp_wrapper_f64"
    return xops.CustomCallWithLayout(c, op_name, operands=(size_xla, q, vect, q_tan, vect_tan, q_t, vect_t, q_tan_t, vect_tan_t), 
                                        operand_shapes_with_layout=dims_spec, shape_with_layout=dims_spec[2])

# *  SUPPORT FOR BATCHING WITH VMAP  *
@trace('iM_product_vect_jvp_jvp_batch')
def _iM_product_vect_jvp_jvp_batch(args, axes):
    """ Batch rule for _iM_product_vect_jvp_jvp_prim
    """
    (q, vect, q_tan, vect_tan, q_t, vect_t, q_tan_t, vect_tan_t), size_batch = check_batch_inputs(args, axes)
    if len(q.shape) <= 2:
        return _iM_product_vect_jvp_jvp_prim.bind(q, vect, q_tan, vect_tan, q_t, vect_t, q_tan_t, vect_tan_t), 0
    for i in range(size_batch):
        mCompute = _iM_product_vect_jvp_jvp_prim.bind(q[i], vect[i], q_tan[i], vect_tan[i], q_t[i], vect_t[i], q_tan_t[i], vect_tan_t[i])[None]
        jac_res = mCompute if i == 0 else jnp.vstack((jac_res, mCompute))
    return jac_res, 0

# *  SUPPORT FOR FORWARD AUTODIFF  OF iM_product_vect_JVP*
@trace('iM_product_vect_jvp_jvp_tan')
def iM_product_vect_jvp_jvp(args, tangents):
    """ JVP rule for _iM_product_vect_jvp_prim
    """
    def make_zero(tan, val):
        return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan
    q, vect, q_tan, vect_tan = args
    q_t, vect_t, q_tan_t, vect_tan_t = tangents
    q_t_arr, vect_t_arr,  q_tan_t_arr, vect_tan_t_arr = make_zero(q_t, q), make_zero(vect_t, vect), make_zero(q_tan_t, q_tan), make_zero(vect_tan_t, vect_tan)
    f_res = _iM_product_vect_jvp_prim.bind(q, vect, q_tan, vect_tan)
    derf_res = _iM_product_vect_jvp_jvp_prim.bind(q, vect, q_tan, vect_tan, q_t_arr, vect_t_arr,  q_tan_t_arr, vect_tan_t_arr)
    return f_res, derf_res

# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
_iM_product_vect_jvp_jvp_prim = core.Primitive("iM_product_vect_jvp_jvp")
_iM_product_vect_jvp_jvp_prim.def_impl(partial(xla.apply_primitive, _iM_product_vect_jvp_jvp_prim))
_iM_product_vect_jvp_jvp_prim.def_abstract_eval(_iM_product_vect_jvp_jvp_abstract)
batching.primitive_batchers[_iM_product_vect_jvp_jvp_prim] = _iM_product_vect_jvp_jvp_batch
xla.backend_specific_translations["cpu"][_iM_product_vect_jvp_jvp_prim] = _iM_product_vect_jvp_jvp_translation
ad.primitive_jvps[_iM_product_vect_jvp_prim] = iM_product_vect_jvp_jvp # *  SUPPORT FOR FORWARD AUTODIFF  OF iM_product_vect_JVP*


# ********************************************
# *     SUPPORT FOR VJP(JVP((M(q)^-1 v))     *
# ********************************************

@trace('iM_product_vect_jvp_vjp_abstract')
def _iM_product_vect_jvp_vjp_abstract(cotan, q, vect, q_tan, vect_tan):
    """ Abstract evaluation of the primitive _iM_product_vect_jvp_vjp_prim
    """
    # Check inputs dimension
    shapearray_cotan, shapearray_q, shapearray_vect,shapearray_q_tan, shapearray_vect_tan  = check_dim_imputs((cotan, q, vect, q_tan, vect_tan)) 
    return (shapearray_q, shapearray_vect, shapearray_q_tan, shapearray_vect_tan)

@trace('iM_product_vect_jvp_vjp_translation')
def _iM_product_vect_jvp_vjp_translation(c, cotan, q, vect, q_tan, vect_tan):
    """ XLA translation of _iM_product_vect_jvp_vjp_prim
    """
    type_in, size_xla, dims_spec = check_dim_imputs((cotan, q, vect, q_tan, vect_tan), c)
    op_name = b"iM_prod_vect_jvp_vjp_wrapper_f32" if type_in == np.float32 else b"iM_prod_vect_jvp_vjp_wrapper_f64"
    return xops.CustomCallWithLayout(c, op_name, operands=(size_xla, cotan, q, vect, q_tan, vect_tan), operand_shapes_with_layout=dims_spec, 
                                        shape_with_layout=xla_client.Shape.tuple_shape((dims_spec[2],dims_spec[3], dims_spec[4],dims_spec[5])))

@trace('iM_product_vect_jvp_vjp_batch')
def _iM_product_vect_jvp_vjp_batch(args, axes):
    """ Batch rule for _iM_product_vect_jvp_vjp_prim
    """
    (cotan, q, vect, q_tan, vect_tan), size_batch = check_batch_inputs(args, axes)
    if len(q.shape) <= 2:
        return _iM_product_vect_jvp_vjp_prim.bind(cotan, q, vect, q_tan, vect_tan), (0, 0, 0, 0)
    for i in range(size_batch):
        cotan_q_t, cotan_vect_t, cotan_q_tan_t, cotan_vect_tan_t  = _iM_product_vect_jvp_vjp_prim.bind(cotan[i], q[i], vect[i], q_tan[i], vect_tan[i])
        cotan_q = cotan_q_t[None] if i == 0 else jnp.vstack((cotan_q, cotan_q_t[None]))
        cotan_vect = cotan_vect_t[None] if i == 0 else jnp.vstack((cotan_vect, cotan_vect_t[None]))
        cotan_q_tan = cotan_q_tan_t[None] if i == 0 else jnp.vstack((cotan_q_tan, cotan_q_tan_t[None]))
        cotan_vect_tan = cotan_vect_tan_t[None] if i == 0 else jnp.vstack((cotan_vect_tan, cotan_vect_tan_t[None]))
    return (cotan_q, cotan_vect, cotan_q_tan, cotan_vect_tan), (0, 0, 0, 0)

@trace('_iM_product_vect_jvp_vjp_transpose')
def iM_product_vect_jvp_vjp_transpose(ct, q, vect, q_tan, vect_tan, q_t, vect_t, q_tan_t, vect_tan_t):
    """ Transpose for the primitive _iM_product_vect_jvp_jvp_prim used inside the Jacobian Vector product of _iM_product_vect_prim
    """
    assert not ad.is_undefined_primal(q) and not ad.is_undefined_primal(vect) and not ad.is_undefined_primal(q_tan) and not ad.is_undefined_primal(vect_tan), \
            'The four first arguments should be constant in vjp formulation'
    assert ad.is_undefined_primal(q_t) or ad.is_undefined_primal(vect_t) or ad.is_undefined_primal(q_tan_t) or ad.is_undefined_primal(vect_tan_t),\
            'Variables can not be undefined'
    if type(ct) is ad.Zero:
        return None, None, None, None, ad.Zero(q_tan.aval) if ad.is_undefined_primal(q_tan) else None, ad.Zero(vect_tan.aval) if ad.is_undefined_primal(q_tan) else None
    cotan_q, cotan_vect, cotan_q_tan, cotan_vect_tan = _iM_product_vect_jvp_vjp_prim.bind(ct, q, vect, q_tan, vect_tan)
    return None, None, None, None, cotan_q if ad.is_undefined_primal(q_t) else None, cotan_vect if ad.is_undefined_primal(vect_t) else None,\
            cotan_q_tan if ad.is_undefined_primal(q_tan_t) else None, cotan_vect_tan if ad.is_undefined_primal(vect_tan_t) else None

# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *       
_iM_product_vect_jvp_vjp_prim = core.Primitive("iM_product_vect_jvp_vjp")
_iM_product_vect_jvp_vjp_prim.multiple_results = True
_iM_product_vect_jvp_vjp_prim.def_impl(partial(xla.apply_primitive, _iM_product_vect_jvp_vjp_prim))
_iM_product_vect_jvp_vjp_prim.def_abstract_eval(_iM_product_vect_jvp_vjp_abstract)
batching.primitive_batchers[_iM_product_vect_jvp_vjp_prim] = _iM_product_vect_jvp_vjp_batch
xla.backend_specific_translations["cpu"][_iM_product_vect_jvp_vjp_prim] = _iM_product_vect_jvp_vjp_translation
ad.primitive_transposes[_iM_product_vect_jvp_jvp_prim] = iM_product_vect_jvp_vjp_transpose

# @trace('iM_product_vect_vjp_transpose_jvp')
# def iM_product_vect_vjp_transpose_jvp(args, tangents):
#     c, q, vect = args
#     c_tan, q_tan, vect_tan = tangents
#     def make_zero(tan, val):
#         return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan
#     c_tan_arr, q_tan_arr, vect_tan_arr = make_zero(c_tan, c), make_zero(q_tan, q), make_zero(vect_tan, vect)
#     cotan_q, cotan_vect = _iM_product_vect_vjp_prim.bind(c, q, vect)

#     return (cotan_q, cotan_vect), (q_tan_arr, vect_tan_arr)