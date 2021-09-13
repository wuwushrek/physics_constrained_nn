#ifndef _MUJOCO_UTILS_JAX_KERNEL_HELPERS_H_
#define _MUJOCO_UTILS_JAX_KERNEL_HELPERS_H_

#include "mujoco.h"
#include <tuple>

extern mjModel* mModel;

// Initialize the global variable storing the model parameters
int init_model(const char* model_path, const int nstep, const double eps_diff);

// Given  the id_q of a qpos, return a tuple (i,j) of indexes, where i specifies
// -1 if the qpos is not a quaternion, and i>=0 the starting index of the quaternion id_q belongs to.
// j represents the address/index of the corresponding speed \dot{q}
std::tuple<int, int> joint_mapping(const int id_q);

// Compute the vector Jacobian product of iM_prod_vect givem the cotan of the output
void iM_prod_vect_vjp(const double *cotan, const double *q, const double *vect, double *q_cotan, double *vect_cotan);

// Compute The jacobian vector product of iM_prod_vect given tan inputs q_tan, vect_tan
// Result : jac(M(q)^-1 vect) q_tan + M(q)^-1 vect_tan
void iM_prod_vect_jvp(const double *q, const double *vect, const double *q_tan, const double *vect_tan, double *out_tan);

// Compute the product of the inverse of the mass inertia matrix and a given vector
// and store the result inside the vector out: M(q)^-1 out
void iM_prod_vect(const double *q, const double *vect, double *out);

// Compute the jacobian vector product of the jacobian vector product of of iM_prod_vect_jvp
void iM_prod_vect_jvp_jvp(const double* q,  const double* v, const double* q1,  const double* v1,
						const double* q_t,  const double* v_t, const double* q1_t,  const double* v1_t,  
						double *out);

// Compute the vjp rules of iM_prod_vect_jvp
void iM_prod_vect_jvp_vjp(const double* cotan, const double* q,  const double* v, const double* q1,  const double* v1,
							double* cotan_q,  double* cotan_v, double* cotan_q1, double* cotan_v1);

#endif