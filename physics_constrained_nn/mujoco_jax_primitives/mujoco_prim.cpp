// cppimport

// The import bove is essential for this module to be recognized for python import

// This file defines the Python interface to the XLA custom call implemented on the CPU.
// It is exposed as a standard pybind11 module defining "capsule" objects containing our
// method. For simplicity, we export a separate capsule for each supported dtype.

// export OMP_NUM_THREADS=1

#include "mujoco_utils.h"
#include "pybind11_kernel_helpers.h"
#include <algorithm>

// enable compilation with and without OpenMP support
#if defined(_OPENMP)
    #include <omp.h>
#endif

using namespace mujoco_prim; // Mujoco primitive

namespace {

template <typename T>
void iM_prod_vect_wrapper(void *out, const void **in) {
  // Parse the inputs
  const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]); // Specify the number of batches for each input
  const T *q = reinterpret_cast<const T *>(in[1]); // Read the q vector for which to evaluate the intertia matrix
  const T *vect = reinterpret_cast<const T *>(in[2]); // Read the constant vector for which to do inv(inertia Matrix) vector
  T *iMq_v = reinterpret_cast<T *>(out); // The output is stored as a pointer as we have a single output

  // Compute the output here
  #pragma omp parallel for schedule(static)
  for (std::int64_t n = 0; n < size; ++n) {
    double qval[mModel->nq];
    double vTerm[mModel->nv];
    double resVect[mModel->nv];
    // Copy and convert the inputs to be double variables
    std::copy(&q[n*mModel->nq], &q[n*mModel->nq]+mModel->nq, qval);
    std::copy(&vect[n*mModel->nv], &vect[n*mModel->nv]+mModel->nv, vTerm);
    // Find the product by calling MuJoCo
    iM_prod_vect(qval, vTerm, resVect);
    // Save the output with the Template argument
    std::copy(resVect, resVect+mModel->nv, &iMq_v[n*mModel->nv]);
  }
}

template <typename T>
void iM_prod_vect_jvp_wrapper(void *out, const void **in) {
  // Parse the inputs
  const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]); // Specify the number of batches for each input
  const T *q = reinterpret_cast<const T *>(in[1]); // Read the q vector for which to evaluate the intertia matrix
  const T *vect = reinterpret_cast<const T *>(in[2]); // Read the constant vector for which to do inv(inertia Matrix) vector
  const T *q_tan = reinterpret_cast<const T *>(in[3]); // Read the q vector for which to evaluate the intertia matrix
  const T *vect_tan = reinterpret_cast<const T *>(in[4]); // Read the constant vector for which to do inv(inertia Matrix) vector
  T *iMq_v = reinterpret_cast<T *>(out); // The output is stored as a pointer as we have a single output

  // Compute the output here
  #pragma omp parallel for schedule(static)
  for (std::int64_t n = 0; n < size; ++n) {
    double qval[mModel->nq];
    double vTerm[mModel->nv];
    double qval_tan[mModel->nq];
    double vTerm_tan[mModel->nv];
    double resVect[mModel->nv];
    // Copy and convert the inputs to be double variables
    std::copy(&q[n*mModel->nq], &q[n*mModel->nq]+mModel->nq, qval);
    std::copy(&vect[n*mModel->nv], &vect[n*mModel->nv]+mModel->nv, vTerm);
    std::copy(&q_tan[n*mModel->nq], &q_tan[n*mModel->nq]+mModel->nq, qval_tan);
    std::copy(&vect_tan[n*mModel->nv], &vect_tan[n*mModel->nv]+mModel->nv, vTerm_tan);
    // Find the product by calling MuJoCo
    iM_prod_vect_jvp(qval, vTerm, qval_tan, vTerm_tan, resVect);
    // Save the output with the Template argument
    std::copy(resVect, resVect+mModel->nv, &iMq_v[n*mModel->nv]);
  }
}

template <typename T>
void iM_prod_vect_jvp_jvp_wrapper(void *out, const void **in) {
  // Parse the inputs
  const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]); // Specify the number of batches for each input
  const T *q = reinterpret_cast<const T *>(in[1]); // Read the q vector for which to evaluate the intertia matrix
  const T *vect = reinterpret_cast<const T *>(in[2]); // Read the constant vector for which to do inv(inertia Matrix) vector
  const T *q_tan = reinterpret_cast<const T *>(in[3]); // Read the q vector for which to evaluate the intertia matrix
  const T *vect_tan = reinterpret_cast<const T *>(in[4]); // Read the constant vector for which to do inv(inertia Matrix) vector
  const T *q_t = reinterpret_cast<const T *>(in[5]); // Read the q vector for which to evaluate the intertia matrix
  const T *vect_t = reinterpret_cast<const T *>(in[6]); // Read the constant vector for which to do inv(inertia Matrix) vector
  const T *q_tan_t = reinterpret_cast<const T *>(in[7]); // Read the q vector for which to evaluate the intertia matrix
  const T *vect_tan_t = reinterpret_cast<const T *>(in[8]); // Read the constant vector for which to do inv(inertia Matrix) vector
  T *iMq_v = reinterpret_cast<T *>(out); // The output is stored as a pointer as we have a single output

  // Compute the output here
  #pragma omp parallel for schedule(static)
  for (std::int64_t n = 0; n < size; ++n) {
    double qval[mModel->nq];
    double vTerm[mModel->nv];
    double qval_tan[mModel->nq];
    double vTerm_tan[mModel->nv];
    double qval_t[mModel->nq];
    double vTerm_t[mModel->nv];
    double qval_tan_t[mModel->nq];
    double vTerm_tan_t[mModel->nv];
    double resVect[mModel->nv];
    // Copy and convert the inputs to be double variables
    std::copy(&q[n*mModel->nq], &q[n*mModel->nq]+mModel->nq, qval);
    std::copy(&vect[n*mModel->nv], &vect[n*mModel->nv]+mModel->nv, vTerm);
    std::copy(&q_tan[n*mModel->nq], &q_tan[n*mModel->nq]+mModel->nq, qval_tan);
    std::copy(&vect_tan[n*mModel->nv], &vect_tan[n*mModel->nv]+mModel->nv, vTerm_tan);
    std::copy(&q_t[n*mModel->nq], &q_t[n*mModel->nq]+mModel->nq, qval_t);
    std::copy(&vect_t[n*mModel->nv], &vect_t[n*mModel->nv]+mModel->nv, vTerm_t);
    std::copy(&q_tan_t[n*mModel->nq], &q_tan_t[n*mModel->nq]+mModel->nq, qval_tan_t);
    std::copy(&vect_tan_t[n*mModel->nv], &vect_tan_t[n*mModel->nv]+mModel->nv, vTerm_tan_t);
    // Find the product by calling MuJoCo
    iM_prod_vect_jvp_jvp(qval, vTerm, qval_tan, vTerm_tan, qval_t, vTerm_t, qval_tan_t, vTerm_tan_t, resVect);
    // Save the output with the Template argument
    std::copy(resVect, resVect+mModel->nv, &iMq_v[n*mModel->nv]);
  }
}

template <typename T>
void iM_prod_vect_vjp_wrapper(void *out_tuple, const void **in) {
  // Parse the inputs
  const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]); // Specify the number of batches for each input
  const T *cotan = reinterpret_cast<const T *>(in[1]); // Read the q vector for which to evaluate the intertia matrix
  const T *q = reinterpret_cast<const T *>(in[2]); // Read the constant vector for which to do inv(inertia Matrix) vector
  const T *vect = reinterpret_cast<const T *>(in[3]); // Read the q vector for which to evaluate the intertia matrix
  
  T **out = reinterpret_cast<T **>(out_tuple); // The output is stored as a pointer as we have a single output
  T *q_cotan = reinterpret_cast<T *>(out[0]); // The output M(q)^-1 x v is stored as a pointer as we have a single output
  T *vect_cotan = reinterpret_cast<T *>(out[1]); 
  // Compute the output here
  #pragma omp parallel for schedule(static)
  for (std::int64_t n = 0; n < size; ++n) {
    double qval[mModel->nq];
    double vTerm[mModel->nv];
    double cotan_val[mModel->nv];
    double q_cotan_res[mModel->nq];
    double vect_cotan_res[mModel->nv];
    // Copy and convert the inputs to be double variables
    std::copy(&q[n*mModel->nq], &q[n*mModel->nq]+mModel->nq, qval);
    std::copy(&vect[n*mModel->nv], &vect[n*mModel->nv]+mModel->nv, vTerm);
    std::copy(&cotan[n*mModel->nv], &cotan[n*mModel->nv]+mModel->nv, cotan_val);
    // Find the product by calling MuJoCo
    iM_prod_vect_vjp(cotan_val, qval, vTerm, q_cotan_res, vect_cotan_res);
    // Save the output with the Template argument
    std::copy(q_cotan_res, q_cotan_res+mModel->nq, &q_cotan[n*mModel->nq]);
    std::copy(vect_cotan_res, vect_cotan_res+mModel->nv, &vect_cotan[n*mModel->nv]);
  }
}

template <typename T>
void iM_prod_vect_jvp_vjp_wrapper(void *out_tuple, const void **in) {
  // Parse the inputs
  const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]); // Specify the number of batches for each input
  const T *cotan = reinterpret_cast<const T *>(in[1]); // Read the q vector for which to evaluate the intertia matrix
  const T *q = reinterpret_cast<const T *>(in[2]); // Read the constant vector for which to do inv(inertia Matrix) vector
  const T *vect = reinterpret_cast<const T *>(in[3]); // Read the q vector for which to evaluate the intertia matrix
  const T *q_tan = reinterpret_cast<const T *>(in[4]); // Read the constant vector for which to do inv(inertia Matrix) vector
  const T *vect_tan = reinterpret_cast<const T *>(in[5]);
  
  T **out = reinterpret_cast<T **>(out_tuple); // The output is stored as a pointer as we have a single output
  T *q_cotan = reinterpret_cast<T *>(out[0]); // The output M(q)^-1 x v is stored as a pointer as we have a single output
  T *vect_cotan = reinterpret_cast<T *>(out[1]);
  T *q_tan_cotan = reinterpret_cast<T *>(out[2]); // The output M(q)^-1 x v is stored as a pointer as we have a single output
  T *vect_tan_cotan = reinterpret_cast<T *>(out[3]);
  // Compute the output here
  #pragma omp parallel for schedule(static)
  for (std::int64_t n = 0; n < size; ++n) {
    double qval[mModel->nq];
    double vTerm[mModel->nv];
    double qval_tan[mModel->nq];
    double vTerm_tan[mModel->nv];
    double cotan_val[mModel->nv];

    double q_cotan_res[mModel->nq];
    double vect_cotan_res[mModel->nv];
    double q_tan_cotan_res[mModel->nq];
    double vect_tan_cotan_res[mModel->nv];

    // Copy and convert the inputs to be double variables
    std::copy(&q[n*mModel->nq], &q[n*mModel->nq]+mModel->nq, qval);
    std::copy(&vect[n*mModel->nv], &vect[n*mModel->nv]+mModel->nv, vTerm);
    std::copy(&q_tan[n*mModel->nq], &q_tan[n*mModel->nq]+mModel->nq, qval_tan);
    std::copy(&vect_tan[n*mModel->nv], &vect_tan[n*mModel->nv]+mModel->nv, vTerm_tan);
    std::copy(&cotan[n*mModel->nv], &cotan[n*mModel->nv]+mModel->nv, cotan_val);
    // Find the product by calling MuJoCo
    iM_prod_vect_jvp_vjp(cotan_val, qval, vTerm, qval_tan, vTerm_tan, q_cotan_res, vect_cotan_res, q_tan_cotan_res, vect_tan_cotan_res);
    // Save the output with the Template argument
    std::copy(q_cotan_res, q_cotan_res+mModel->nq, &q_cotan[n*mModel->nq]);
    std::copy(vect_cotan_res, vect_cotan_res+mModel->nv, &vect_cotan[n*mModel->nv]);
    std::copy(q_tan_cotan_res, q_tan_cotan_res+mModel->nq, &q_tan_cotan[n*mModel->nq]);
    std::copy(vect_tan_cotan_res, vect_tan_cotan_res+mModel->nv, &vect_tan_cotan[n*mModel->nv]);
  }
}

pybind11::dict Registrations() {
  pybind11::dict dict;

  dict["iM_prod_vect_wrapper_f32"] = EncapsulateFunction(iM_prod_vect_wrapper<float>);
  dict["iM_prod_vect_wrapper_f64"] = EncapsulateFunction(iM_prod_vect_wrapper<double>);

  dict["iM_prod_vect_jvp_wrapper_f32"] = EncapsulateFunction(iM_prod_vect_jvp_wrapper<float>);
  dict["iM_prod_vect_jvp_wrapper_f64"] = EncapsulateFunction(iM_prod_vect_jvp_wrapper<double>);

  dict["iM_prod_vect_vjp_wrapper_f32"] = EncapsulateFunction(iM_prod_vect_vjp_wrapper<float>);
  dict["iM_prod_vect_vjp_wrapper_f64"] = EncapsulateFunction(iM_prod_vect_vjp_wrapper<double>);

  dict["iM_prod_vect_jvp_jvp_wrapper_f32"] = EncapsulateFunction(iM_prod_vect_jvp_jvp_wrapper<float>);
  dict["iM_prod_vect_jvp_jvp_wrapper_f64"] = EncapsulateFunction(iM_prod_vect_jvp_jvp_wrapper<double>);

  dict["iM_prod_vect_jvp_vjp_wrapper_f32"] = EncapsulateFunction(iM_prod_vect_jvp_vjp_wrapper<float>);
  dict["iM_prod_vect_jvp_vjp_wrapper_f64"] = EncapsulateFunction(iM_prod_vect_jvp_vjp_wrapper<double>);

  return dict;
}

PYBIND11_MODULE(mujoco_prim, m) { 
  m.doc() = "An interface of primitive function to interact with MuJoCo"; // optional module docstring
  m.def("registrations", &Registrations); 
  m.def("init_model", &init_model, "A function to initialize the global variable containing the model of interest");
  m.def("joint_mapping", &joint_mapping, "A function to return if the qpos[id] belongs to quaternion");
}  

}// namespace

/*
<%
import os
cfg['extra_compile_args'] = ['-std=c++11', '-fopenmp', '-mavx']
cfg['extra_link_args'] = ['-fopenmp']
cfg['include_dirs'] = ['{}/include'.format(os.environ['MUJOCO_PY_MUJOCO_PATH'])]
cfg['library_dirs'] = ['{}/bin'.format(os.environ['MUJOCO_PY_MUJOCO_PATH'])]
cfg['libraries'] = ['mujoco200nogl']
cfg['dependencies'] = ['mujoco_utils.h', 'pybind11_kernel_helpers.h', 'kernel_helpers.h']
cfg['sources'] = ['mujoco_utils.cpp']
setup_pybind11(cfg)
%>
*/