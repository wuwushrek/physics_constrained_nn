#include "mujoco_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Load the model that will be used for derivative computation
mjModel* mModel = 0;
double _eps_diff = 1e-6;
double _eps_diff_horder = 1e-6;
// We assume a model will not have a state space of size greater than 1000
int qpos_type[1000];
int qvel_type[1000];

// Initialize the global variable storing the model parameters
int init_model(const char* model_path, const int nstep, const double eps_diff){
	// Activate the mujoco environment using the key
	mj_activate(std::getenv("MUJOCO_PY_MJKEY_PATH"));
	// Load the model and save it into the global variable
	mModel = mj_loadXML(model_path, NULL, NULL, 0);
	// The epsilon used in the finite difference computation
	_eps_diff = eps_diff; 
	_eps_diff_horder = pow(eps_diff, 0.75);
	printf("Error second term : %f \n", _eps_diff_horder);
	// Check if the model was actually loaded correctly
	if (!mModel){
		printf("Could not load modelfile '%s'\n", model_path);
		return 0;
	}
	// Initialize qpos_type and qvel_type with -1
	for (int j=0; j<1000; j++){
		qpos_type[j] = -1;
		qvel_type[j] = -1;
	}
	// Store for each qpos index the type (quaternion or not) of the corresponding joint
	for (int i = 0; i < mModel->njnt; i++){
		if (mModel->jnt_type[i] == mjJNT_BALL){ // Quaternion
			// printf("Joint %d, size = 4, start address : %d , dof : %d \n", i, mModel->jnt_qposadr[i], mModel->jnt_dofadr[i]);
			for (int j=mModel->jnt_qposadr[i]; j < mModel->jnt_qposadr[i]+4; j++){
				qpos_type[j] = mModel->jnt_qposadr[i];
				qvel_type[j] = mModel->jnt_dofadr[i];
			}
		}else if (mModel->jnt_type[i] == mjJNT_FREE){ // Free joint is [position, quaterion]
			// printf("Joint %d, size = 7, start address : %d, quat address : %d, dof : %d \n", i, mModel->jnt_qposadr[i], mModel->jnt_qposadr[i]+3, mModel->jnt_dofadr[i]);
			for (int j=0; j <3; j++)
				qvel_type[mModel->jnt_qposadr[i]+j] = mModel->jnt_dofadr[i] + j;
			for (int j=mModel->jnt_qposadr[i]+3; j < mModel->jnt_qposadr[i]+7; j++){
				qpos_type[j] = mModel->jnt_qposadr[i] + 3;
				qvel_type[j] = mModel->jnt_dofadr[i]  + 3;
			}
		} else{ // Otherwise, not a quaternion
			// printf("Joint %d, size = 1, start address : %d, dof : %d \n", i, mModel->jnt_qposadr[i], mModel->jnt_dofadr[i]);
			qvel_type[mModel->jnt_qposadr[i]] = mModel->jnt_dofadr[i];
		}
	}
	// Update the mumber of iterations required by the optimzer
	mModel->opt.iterations = nstep;
	// Set the error tolerance to be 0
	mModel->opt.tolerance = 0;
	// Some printing for user
	printf("Number of q : %d \n", mModel->nq);
	printf("Number of v : %d \n", mModel->nv);
	printf("Number of u : %d \n", mModel->nu);
	return 1;
}

// Given  the id_q of a qpos, return a tuple (i,j) of indexes, where i specifies
// -1 if the qpos is not a quaternion, and i>=0 the starting index of the quaternion id_q belongs to.
// j represents the address/index of the corresponding speed \dot{q}
std::tuple<int, int> joint_mapping(const int id_q){
	return std::make_tuple(qpos_type[id_q], qvel_type[id_q]);
}

// Compute the product of the inverse of the mass inertia matrix and a given vector
// and store the result inside the vector out: M(q)^-1 vect
void iM_prod_vect(const double *q, const double *vect, double *out){
	// Load the data structure of the model ino mjData
	mjData* d = mj_makeData(mModel);
	// Copy the input inside the data of the Mujoco Model
	mju_copy(d->qpos, q, mModel->nq);
	// Forward dynamics and position-related parameters to compute M(q)
	mj_fwdPosition(mModel, d);
	// mju_printMat(d->qM, 1, mModel->nM);
	// Now used the computed M(q) to efficiently compute the product M(q)^-1 vect
	mj_solveM(mModel, d, out, vect, 1);
	// Free and delete the created/allocated data
	mj_deleteData(d);
}

// Compute The jacobian vector product of iM_prod_vect given tan inputs q_tan, vect_tan
// Result : jac(M(q)^-1 vect) q_tan + M(q)^-1 vect_tan
void iM_prod_vect_jvp(const double *q, const double *vect, const double *q_tan, const double *vect_tan, double *out_tan){
	// Load the data structure of the model ino mjData
	mjData* d = mj_makeData(mModel);
	// Copy qpos in the data d
	mju_copy(d->qpos, q, mModel->nq);
	// Now compute the center value and aux result
	mj_fwdPosition(mModel, d); // Compute first M(q)
	mjtNum centerV[mModel->nv]; // Store the product M(q)^-1 vect
	mj_solveM(mModel, d, centerV, vect, 1); // Compute the product M(q)^-1 vect
	mj_solveM(mModel, d, out_tan, vect_tan, 1); // Compute M(q)^-1 vect_tan
	mjtNum diff_q[mModel->nq*mModel->nv];
	for (int i=0; i<mModel->nq; i++){
		// get the quaternion id (starting address) for this dof
		int quatadr = qpos_type[i]; // (-1: not a quaternion)
		d->qpos[i] += _eps_diff; // Finite q[i] + h
		if (quatadr >= 0)
			mju_normalize4(&d->qpos[quatadr]); // Re-normalize quaternion after perturbation
		mj_fwdPosition(mModel, d); // Forward dynamics and position-related parameters to compute M(q+h)
		mj_solveM(mModel, d, &diff_q[mModel->nv*i], vect, 1); //  Now used the computed M(q+h) to efficiently compute the product inv(M(q+h))vterm
		mju_subFrom(&diff_q[mModel->nv*i], centerV, mModel->nv); // finite difference
		if (quatadr >= 0)
			mju_copy4(&d->qpos[quatadr], &q[quatadr]);
		else
			d->qpos[i] = q[i];
	}
	// Do the product of jac(M(q)^-1 vect) q_tan
	mju_mulMatTVec(centerV, diff_q, q_tan, mModel->nq, mModel->nv);
	// Append the result to jvp_tan
	mju_addToScl(out_tan, centerV, 1.0/_eps_diff, mModel->nv);
	mj_deleteData(d);
}

// Compute the jacobian vector product of mult_minv_v and a tangent vector (qpos_t, vTerm_t)
void iM_prod_vect_jvp_jvp(const double* q,  const double* v, const double* q1,  const double* v1,
							const double* q_t,  const double* v_t, const double* q1_t,  const double* v1_t,  
							double *out){
	// Load the data structure of the model into mjData
	mjData* d = mj_makeData(mModel);
	// Copy qpos in the data d
	mju_copy(d->qpos, q, mModel->nq);
	// Now compute the center value and aux result
	mj_fwdPosition(mModel, d); // Compute first M(q)
	mjtNum iMq_v[mModel->nv]; // Store the product M(q)^-1 v
	mj_solveM(mModel, d, iMq_v, v, 1); // Compute the product M(q)^-1 v
	mjtNum iMq_v1[mModel->nv]; // Store the product M(q)^-1 v1
	mj_solveM(mModel, d, iMq_v1, v1, 1); // Compute the product M(q)^-1 v1
	// mjtNum iMq_v1_t[mModel->nv];
	mj_solveM(mModel, d, out, v1_t, 1); // Compute M(q)^-1 v1_t
	mjtNum iMq_v_t[mModel->nv];
	mj_solveM(mModel, d, iMq_v_t, v_t, 1); // Compute M(q)^-1 v_t

	mjtNum diff_q_v[mModel->nq*mModel->nv];
	mjtNum diff_q_v_t[mModel->nq*mModel->nv];
	mjtNum diff_q_v1[mModel->nq*mModel->nv];

	for (int i=0; i<mModel->nq; i++){
		// get the quaternion id (starting address) for this dof
		int quatadr = qpos_type[i]; // (-1: not a quaternion)
		// printf("Address quat : %d \n", quatadr);
		d->qpos[i] += _eps_diff; // Finite qpos + h
		if (quatadr >= 0)
			mju_normalize4(&d->qpos[quatadr]); // Re-normalize quaternion after perturbation
		mj_fwdPosition(mModel, d); // Forward dynamics and position-related parameters to compute M(q+h)

		// mju_printMat(d->qM, 1, mModel->nM);
		mj_solveM(mModel, d, &diff_q_v[mModel->nv*i], v, 1); //  (der(M(q)^-1 v)/der(q))_{:,i}
		mju_subFrom(&diff_q_v[mModel->nv*i], iMq_v, mModel->nv); // finite difference (der(M(q)^-1 v)/der(q))_{:,i}

		mj_solveM(mModel, d, &diff_q_v1[mModel->nv*i], v1, 1); //  (der(M(q)^-1 v1)/der(q))_{:,i}
		mju_subFrom(&diff_q_v1[mModel->nv*i], iMq_v1, mModel->nv); // finite difference (der(M(q)^-1 v1)/der(q))_{:,i}

		mj_solveM(mModel, d, &diff_q_v_t[mModel->nv*i], v_t, 1); //  (der(M(q)^-1 v_t)/der(q))_{:,i}
		mju_subFrom(&diff_q_v_t[mModel->nv*i], iMq_v_t, mModel->nv); // finite difference (der(M(q)^-1 v_t)/der(q))_{:,i}

		if (quatadr >= 0)
			mju_copy4(&d->qpos[quatadr], &q[quatadr]);
		else
			d->qpos[i] = q[i];
	}

	// Compute the four partial derivatives term
	mju_mulMatTVec(iMq_v_t, diff_q_v, q1_t, mModel->nq, mModel->nv);
	mju_addToScl(out, iMq_v_t, 1.0/_eps_diff, mModel->nv);

	mju_mulMatTVec(iMq_v_t, diff_q_v_t, q1, mModel->nq, mModel->nv);
	mju_addToScl(out, iMq_v_t, 1.0/_eps_diff, mModel->nv);

	mju_mulMatTVec(iMq_v_t, diff_q_v1, q_t, mModel->nq, mModel->nv);
	mju_addToScl(out, iMq_v_t, 1.0/_eps_diff, mModel->nv);

	// Copy the partial derivative product with q1 in iMq_v1
	mju_mulMatTVec(iMq_v_t, diff_q_v, q1, mModel->nq, mModel->nv);
	mju_scl(iMq_v1, iMq_v_t, 1.0/_eps_diff, mModel->nv); // Copy the partial derivative product with q1 in iMq_v1

	// Compute the second order partial derivative with respect to q
	mjtNum qpos_curr[mModel->nq];
	for (int i=0; i<mModel->nq; i++){
		int quatadr_i = qpos_type[i]; // (-1: not a quaternion)
		d->qpos[i] += _eps_diff_horder; // Finite qpos + h
		if (quatadr_i >= 0)
			mju_normalize4(&d->qpos[quatadr_i]); // Re-normalize quaternion after perturbation
		mju_copy(qpos_curr, d->qpos, mModel->nq);
		mj_fwdPosition(mModel, d);
		mj_solveM(mModel, d, iMq_v, v, 1);
		for (int j=0; j<mModel->nq; j++){
			int quatadr_j = qpos_type[j]; // (-1: not a quaternion)
			d->qpos[j] += _eps_diff; // Finite qpos + h
			if (quatadr_j >= 0)
				mju_normalize4(&d->qpos[quatadr_j]); // Re-normalize quaternion after perturbation
			mj_fwdPosition(mModel, d);
			mj_solveM(mModel, d, &diff_q_v[mModel->nv*j], v, 1);
			mju_subFrom(&diff_q_v[mModel->nv*j], iMq_v, mModel->nv);
			if (quatadr_j >= 0)
				mju_copy4(&d->qpos[quatadr_j], &qpos_curr[quatadr_j]);
			else
				d->qpos[j] = qpos_curr[j];
		}
		mju_mulMatTVec(iMq_v_t, diff_q_v, q1, mModel->nq, mModel->nv);
		mju_scl(&diff_q_v_t[mModel->nv*i], iMq_v_t, 1.0/_eps_diff, mModel->nv);
		mju_subFrom(&diff_q_v_t[mModel->nv*i], iMq_v1, mModel->nv);
		if (quatadr_i >= 0)
			mju_copy4(&d->qpos[quatadr_i], &q[quatadr_i]);
		else
			d->qpos[i] = q[i];
	}
	mju_mulMatTVec(iMq_v_t, diff_q_v_t, q_t, mModel->nq, mModel->nv);
	mju_addToScl(out, iMq_v_t, 1.0/_eps_diff_horder, mModel->nv);

	mj_deleteData(d);
}

// Compute the jacobian vector product of mult_minv_v and a tangent vector (qpos_t, vTerm_t)
void iM_prod_vect_jvp_vjp(const double* cotan, const double* q,  const double* v, const double* q1,  const double* v1,
							double* cotan_q,  double* cotan_v, double* cotan_q1, double* cotan_v1){
	// Load the data structure of the model into mjData
	mjData* d = mj_makeData(mModel);
	// Copy qpos in the data d
	mju_copy(d->qpos, q, mModel->nq);
	// Now compute the center value and aux result
	mj_fwdPosition(mModel, d); // Compute first M(q)
	mjtNum iMq_v[mModel->nv]; // Store the product M(q)^-1 v
	mj_solveM(mModel, d, iMq_v, v, 1); // Compute the product M(q)^-1 v
	mjtNum iMq_v1[mModel->nv]; // Store the product M(q)^-1 v1
	mj_solveM(mModel, d, iMq_v1, v1, 1); // Compute the product M(q)^-1 v1
	// Fill cotan of v1: cotan with respect to v1 also name iMq_v_cotan
	mj_solveM(mModel, d, cotan_v1, cotan, 1); // Compute cotan * M(q)^-1

	mjtNum diff_q_v[mModel->nq*mModel->nv];
	mjtNum diff_q_v_t[mModel->nq*mModel->nv]; // Also named diff_q_cotan 
	mjtNum diff_q_v1[mModel->nq*mModel->nv];

	for (int i=0; i<mModel->nq; i++){
		// get the quaternion id (starting address) for this dof
		int quatadr = qpos_type[i]; // (-1: not a quaternion)
		// printf("Address quat : %d \n", quatadr);
		d->qpos[i] += _eps_diff; // Finite qpos + h
		if (quatadr >= 0)
			mju_normalize4(&d->qpos[quatadr]); // Re-normalize quaternion after perturbation
		mj_fwdPosition(mModel, d); // Forward dynamics and position-related parameters to compute M(q+h)

		// mju_printMat(d->qM, 1, mModel->nM);
		mj_solveM(mModel, d, &diff_q_v[mModel->nv*i], v, 1); //  (der(M(q)^-1 v)/der(q))_{:,i}
		mju_subFrom(&diff_q_v[mModel->nv*i], iMq_v, mModel->nv); // finite difference (der(M(q)^-1 v)/der(q))_{:,i}

		mj_solveM(mModel, d, &diff_q_v1[mModel->nv*i], v1, 1); //  (der(M(q)^-1 v1)/der(q))_{:,i}
		mju_subFrom(&diff_q_v1[mModel->nv*i], iMq_v1, mModel->nv); // finite difference (der(M(q)^-1 v1)/der(q))_{:,i}

		mj_solveM(mModel, d, &diff_q_v_t[mModel->nv*i], cotan, 1); //  (der(M(q)^-1 v_t)/der(q))_{:,i}
		mju_subFrom(&diff_q_v_t[mModel->nv*i], cotan_v1, mModel->nv); // finite difference (der(M(q)^-1 v_t)/der(q))_{:,i}

		if (quatadr >= 0)
			mju_copy4(&d->qpos[quatadr], &q[quatadr]);
		else
			d->qpos[i] = q[i];
	}

	mjtNum temp[mModel->nq]; 

	//  Fill cotan of q1
	mju_mulMatVec(temp, diff_q_v, cotan, mModel->nq, mModel->nv);
	mju_scl(cotan_q1, temp, 1.0/_eps_diff, mModel->nq);

	//  Fill cotan of v
	mju_mulMatTVec(iMq_v, diff_q_v_t, q1, mModel->nq, mModel->nv);
	mju_scl(cotan_v, iMq_v, 1.0/_eps_diff, mModel->nv);

	//  Fill cotan of 1
	mju_mulMatVec(temp, diff_q_v1, cotan, mModel->nq, mModel->nv);
	mju_scl(cotan_q, temp, 1.0/_eps_diff, mModel->nq);

	// Copy the partial derivative product with q1 in iMq_v1
	mju_mulMatTVec(iMq_v, diff_q_v, q1, mModel->nq, mModel->nv);
	mju_scl(iMq_v1, iMq_v, 1.0/_eps_diff, mModel->nv); // Copy the partial derivative product with q1 in iMq_v1

	// Compute the second order partial derivative with respect to q
	mjtNum qpos_curr[mModel->nq];
	for (int i=0; i<mModel->nq; i++){
		int quatadr_i = qpos_type[i]; // (-1: not a quaternion)
		d->qpos[i] += _eps_diff_horder; // Finite qpos + h
		if (quatadr_i >= 0)
			mju_normalize4(&d->qpos[quatadr_i]); // Re-normalize quaternion after perturbation
		mju_copy(qpos_curr, d->qpos, mModel->nq);
		mj_fwdPosition(mModel, d);
		mj_solveM(mModel, d, iMq_v, v, 1);
		for (int j=0; j<mModel->nq; j++){
			int quatadr_j = qpos_type[j]; // (-1: not a quaternion)
			d->qpos[j] += _eps_diff; // Finite qpos + h
			if (quatadr_j >= 0)
				mju_normalize4(&d->qpos[quatadr_j]); // Re-normalize quaternion after perturbation
			mj_fwdPosition(mModel, d);
			mj_solveM(mModel, d, &diff_q_v[mModel->nv*j], v, 1);
			mju_subFrom(&diff_q_v[mModel->nv*j], iMq_v, mModel->nv);
			if (quatadr_j >= 0)
				mju_copy4(&d->qpos[quatadr_j], &qpos_curr[quatadr_j]);
			else
				d->qpos[j] = qpos_curr[j];
		}
		mju_mulMatTVec(iMq_v, diff_q_v, q1, mModel->nq, mModel->nv);
		mju_scl(&diff_q_v_t[mModel->nv*i], iMq_v, 1.0/_eps_diff, mModel->nv);
		mju_subFrom(&diff_q_v_t[mModel->nv*i], iMq_v1, mModel->nv);
		if (quatadr_i >= 0)
			mju_copy4(&d->qpos[quatadr_i], &q[quatadr_i]);
		else
			d->qpos[i] = q[i];
	}
	mju_mulMatVec(temp, diff_q_v_t, cotan, mModel->nq, mModel->nv);
	mju_addToScl(cotan_q, temp, 1.0/_eps_diff_horder, mModel->nq);

	mj_deleteData(d);
}

// Compute the vector Jacobian product of iM_prod_vect givem the cotan of the output
void iM_prod_vect_vjp(const double *cotan, const double *q, const double *vect, double *q_cotan, double *vect_cotan){
	// Load the data structure of the model ino mjData
	mjData* d = mj_makeData(mModel);
	// Copy qpos in the data d
	mju_copy(d->qpos, q, mModel->nq);
	// Now compute the center value and aux result
	mj_fwdPosition(mModel, d); // Compute first M(q)
	mjtNum centerV[mModel->nv]; // Store the product M(q)^-1 vect
	mj_solveM(mModel, d, centerV, vect, 1); // Compute the product M(q)^-1 vect
	mj_solveM(mModel, d, vect_cotan, cotan, 1);
	mjtNum diff_q[mModel->nq*mModel->nv];
	for (int i=0; i<mModel->nq; i++){
		// get the quaternion id (starting address) for this dof
		int quatadr = qpos_type[i]; // (-1: not a quaternion)
		d->qpos[i] += _eps_diff; // Finite q[i] + h
		if (quatadr >= 0)
			mju_normalize4(&d->qpos[quatadr]); // Re-normalize quaternion after perturbation
		mj_fwdPosition(mModel, d); // Forward dynamics and position-related parameters to compute M(q+h)
		mj_solveM(mModel, d, &diff_q[mModel->nv*i], vect, 1); //  Now used the computed M(q+h) to efficiently compute the product inv(M(q+h))vterm
		mju_subFrom(&diff_q[mModel->nv*i], centerV, mModel->nv); // finite difference
		if (quatadr >= 0)
			mju_copy4(&d->qpos[quatadr], &q[quatadr]);
		else
			d->qpos[i] = q[i];
	}
	mjtNum temp[mModel->nq]; // Store the product M(q)^-1 vect
	mju_mulMatVec(temp, diff_q, cotan, mModel->nq, mModel->nv);
	mju_scl(q_cotan, temp, 1.0/_eps_diff, mModel->nq);
	mj_deleteData(d);
}

// void position_der(const float* qpos, const float* qvel, )
void mul_quat(double res[4], const double q1[4], const double q2[4]){
	// a1, b1, c1, d1 = q1[0], q1[1], q1[2], q1[3];
	// a2, b2, c2, d2 = q2[0], q2[1], q2[2], q2[3];
	double a1 = q1[0]; double b1 = q1[1]; double c1 = q1[2]; double d1 = q1[3];
	double a2 = q2[0]; double b2 = q2[1]; double c2 = q2[2]; double d2 = q2[3]; 
	res[0] = a1*a2 - b1*b2 - c1*c2 - d1*d2;
	res[1] = a1*b2 + b1*a2 + c1*d2 - d1*c2;
	res[2] = a1*c2 - b1*d2 + c1*a2 + d1*b2;
	res[3] = a1*d2 + b1*c2 - c1*b2 + d1*a2;
}