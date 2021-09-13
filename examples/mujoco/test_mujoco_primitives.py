# from jax.config import config
# config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from physics_constrained_nn.mujoco_jax_primitives import iM_product_vect, initialize_problem_var, quaternion_mapping, compute_qpos_der

import time
from functools import partial

# Import the environment for test
from os import path
from dm_control import suite

from dm_control import viewer
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import mjlib

import argparse
import numpy as np

import time

import numdifftools as nd

# File path 
import os
current_file_path = os.path.dirname(os.path.realpath(__file__))
xml_path = current_file_path + '/../../../dm_control/dm_control/suite/'

# Command line argument
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='reacher')
parser.add_argument('--task_name', type=str, default='hard')
args = parser.parse_args()

# Define the environment
domain_name = args.env_name
task_name = args.task_name

seed = 201 # 703
np.random.seed(seed)

# Launch the environment
env = suite.load(domain_name, task_name, task_kwargs={'random': seed})

# Time step setup, Set this to only do discrete step according to the actual time step
# env.physics.model.opt.timestep = 1e-3
env._n_sub_steps = 1
print('DM time step: ', env._n_sub_steps*env.physics.model.opt.timestep)
actual_dt = env._n_sub_steps*env.physics.model.opt.timestep
# Reset the environment
env.reset()

# Set this to only do discrete step according to the actual time step
# env._n_sub_steps = 1

actSpec = env.action_spec()
lowU = actSpec.minimum
highU = actSpec.maximum
obsSpec = env.observation_spec()
print('Action limits: ', lowU, highU)
print('Observation space: ', obsSpec)

# Launch the viewer
# viewer.launch(env)

# Import the model
model = env.physics.model
# model.opt.integrator = 1

# # Set the model to use elliptic model for contact force
# print('Model cone: ', model.opt.cone)
# print('Model impratio: ', model.opt.impratio)
# model.opt.cone = 1
# model.opt.impratio = 10

# Build the sim
sim = env.physics
print(sim.named.data.qpos)

# Get the data from the sim
data = env.physics.data

print('Pos space: ', model.nq)
print('Vel space: ', model.nv)
print('Action space: ', model.nu)
print('Activation: ', model.na)
print('Time step: ', model.opt.timestep)

n_pos = model.nq
n_vel = model.nv
m_vel = model.nu

# Initialize the parameters for the model
initialize_problem_var(xml_path+domain_name+'.xml', eps_diff=1e-6)
# Find the quaternion distribution over the model
quat_arr = quaternion_mapping(n_pos)
print(quat_arr)

def py_iMv_q_num(x):
	global sim, n_pos, n_vel
	assert x.shape[0] == sim.data.qpos.shape[0] + sim.data.qvel.shape[0]
	sim.data.qpos[:] = x[:n_pos]
	sim.data.ctrl[:] = np.zeros(m_vel, dtype=np.float32)
	sim.forward()
	res_array = np.ndarray(shape=(n_vel,),dtype=np.float64, order='C')
	mjlib.mj_solveM(sim.model.ptr, sim.data.ptr, res_array, x[n_pos:], 1)
	return res_array

def py_iMq_v_eval(q, v):
	x_input = np.hstack((q,v))
	res_py = np.zeros((q.shape[0],n_vel))
	for i in range(q.shape[0]):
		res_py[i,:] = py_iMv_q_num(x_input[i,:])
	return res_py

def py_iMq_v_jvp(q, v, q_tan, v_tan):
	global n_pos, n_vel
	x_input = np.hstack((q,v))
	jacFun = nd.Jacobian(py_iMv_q_num, full_output=True)
	if len(x_input.shape) == 1:
		val, info = jacFun(x_input)
		return info.error_estimate, val @ np.hstack((q_tan, v_tan))

	res_value = np.zeros((x_input.shape[0], n_vel))
	err_est = -1
	for i in range(x_input.shape[0]):
		val, info = jacFun(x_input[i,:])
		err_est = np.maximum(err_est, np.max(info.error_estimate))
		res_value[i,:] = val @ np.hstack((q_tan[i,:], v_tan[i,:]))
	return err_est, res_value

def py_iMq_v_jac(q, v):
	global n_pos, n_vel
	x_input = np.hstack((q,v))
	jacFun = nd.Jacobian(py_iMv_q_num, full_output=True)
	res_value = np.zeros((x_input.shape[0], n_vel, x_input.shape[1]))
	err_est = -1
	for i in range(x_input.shape[0]):
		val, info = jacFun(x_input[i,:])
		err_est = np.maximum(err_est, np.max(info.error_estimate))
		res_value[i] = val
	return err_est, res_value[:,:,:n_pos], res_value[:,:,n_pos:]

def py_iMq_v_jvp_jvp(q, v, qtan, vtan, q_t, v_t, qtan_t, vtan_t):
	global n_pos, n_vel
	jacFun = nd.Jacobian(py_iMv_q_num, full_output=True)

	# q, v, qtan, vtan, q_t, v_t, qtan_t, vtan_t
	def double_jvp(x_vect):
		assert x_vect.shape[0] == 2*(n_pos+n_vel)
		val, info = jacFun(x_vect[:(n_pos+n_vel)])
		return val @ np.hstack(x_vect[(n_pos+n_vel):])

	jacjacFun = nd.Jacobian(double_jvp, full_output=True)

	x_input = np.hstack((q,v,qtan,vtan))
	x_input_tan = np.hstack((q_t,v_t,qtan_t,vtan_t))
	res_value = np.zeros((x_input.shape[0], n_vel))
	err_est = -1
	for i in range(x_input.shape[0]):
		val, info = jacjacFun(x_input[i,:])
		res_value[i,:] = val @ np.hstack(x_input_tan[i,:])
		err_est = np.maximum(err_est, np.max(info.error_estimate))
	return err_est, res_value


print('########################## Collect data from SIM #############################')
# Collect data from the simulator
n_data = 5
env.reset()
qpos_arr = np.zeros((n_data, n_pos))
qvel_arr = np.zeros((n_data, n_vel))

for i in range(n_data):
	qpos_arr[i,:] = sim.data.qpos[:]
	qvel_arr[i,:] = sim.data.qvel[:]
	nextAction = np.random.uniform(low=lowU, high=highU)
	env.step(nextAction)

# Convert the data to jax array
qpos_arr_jnp = jnp.asarray(qpos_arr)
qvel_arr_jnp = jnp.asarray(qvel_arr)

# Collect data for tangent vectors
env.reset()
qpos_arr_tan = np.zeros((n_data, n_pos))
qvel_arr_tan = np.zeros((n_data, n_vel))

for i in range(n_data):
	qpos_arr_tan[i,:] = sim.data.qpos[:]
	qvel_arr_tan[i,:] = sim.data.qvel[:]
	nextAction = np.random.uniform(low=lowU, high=highU)
	env.step(nextAction)

# Convert the data to jax array
qpos_arr_tan_jnp = jnp.asarray(qpos_arr_tan)
qvel_arr_tan_jnp = jnp.asarray(qvel_arr_tan)


print('############################### Test M(q)^-1 v ###############################')
curr_time = time.time()
res_py = py_iMq_v_eval(qpos_arr, qvel_arr)
exec_time_py = time.time() - curr_time
print('Exec time PY        			: ', exec_time_py)

# Jit the function iM_product_vect, then execute it once for 
inv_mass_prod_vector_jit = jax.jit(iM_product_vect, device=jax.devices('cpu')[0])
inv_mass_prod_vector_jit(qpos_arr_jnp, qvel_arr_jnp)

curr_time = time.time()
res_jax = inv_mass_prod_vector_jit(qpos_arr_jnp, qvel_arr_jnp)
exec_time_jax = time.time() - curr_time
print('Exec time JAX       			: ', exec_time_jax)
# print(res_py)
print(res_py.shape, res_jax.shape)
print('Error M^-1(q) v     			: ', np.max(np.abs(np.asarray(res_jax)-res_py))/np.max(np.abs(res_jax)))
assert np.max(np.abs(np.asarray(res_jax)-res_py))/np.max(np.abs(res_jax)) <= 1e-5, 'Inv(M(q)) v doesnt work well'


print('############################ Test JVP(M(q)^-1 v) #############################')
curr_time = time.time()
err_est_jvp, res_py_der = py_iMq_v_jvp(qpos_arr, qvel_arr, qpos_arr_tan, qvel_arr_tan)
exec_time_py = time.time() - curr_time
print('Error Estimate JVP 			: {}'.format(err_est_jvp))
print('Exec time PY        			: ', exec_time_py)

inv_mass_prod_vector_jit_jvp = jax.jit(jax.vmap(lambda x, y, xt, yt : jax.jvp(inv_mass_prod_vector_jit, (x, y), (xt, yt))), device=jax.devices('cpu')[0])
inv_mass_prod_vector_jit_jvp(qpos_arr_jnp, qvel_arr_jnp, qpos_arr_tan, qvel_arr_tan)

curr_time = time.time()
(res_eval, res_jax_der) = inv_mass_prod_vector_jit_jvp(qpos_arr_jnp, qvel_arr_jnp, qpos_arr_tan_jnp, qvel_arr_tan_jnp)
exec_time_jax = time.time() - curr_time
print('Exec time JAX       			: ', exec_time_jax)

print('Error jac(M^-1(q)v) 			: ', np.max(np.abs(np.asarray(res_jax_der)-res_py_der))/np.max(np.abs(res_jax_der)))
assert np.max(np.abs(np.asarray(res_jax_der)-res_py_der))/np.max(np.abs(res_jax_der)) <= 1e-5, 'jvp(Inv(M(q)) v, qtam, vtan)'


print('############################ Test VJP(M(q)^-1 v) #############################')
curr_time = time.time()
err_est_jac, res_py_jacq, res_py_jacv = py_iMq_v_jac(qpos_arr, qvel_arr) 
exec_time_py = time.time() - curr_time
print('Error Estimate JAC 			: {}'.format(err_est_jac))
print('Exec time PY        			: ', exec_time_py)

# Test of VJP by computing and comparing the jacobian of the function
jfwd_v = jax.jit(jax.vmap(jax.jacfwd(inv_mass_prod_vector_jit, argnums=1)), device=jax.devices('cpu')[0])
jfwd_q = jax.jit(jax.vmap(jax.jacfwd(inv_mass_prod_vector_jit, argnums=0)), device=jax.devices('cpu')[0])
jbwd_v = jax.jit(jax.vmap(jax.jacrev(inv_mass_prod_vector_jit, argnums=1)), device=jax.devices('cpu')[0])
jbwd_q = jax.jit(jax.vmap(jax.jacrev(inv_mass_prod_vector_jit, argnums=0)), device=jax.devices('cpu')[0])
jfwd_q(qpos_arr_jnp, qvel_arr_jnp)
jfwd_v(qpos_arr_jnp, qvel_arr_jnp)
jbwd_q(qpos_arr_jnp, qvel_arr_jnp)
jbwd_v(qpos_arr_jnp, qvel_arr_jnp)


curr_time = time.time()
res_jax_jacq = jfwd_q(qpos_arr_jnp, qvel_arr_jnp)
res_jax_jacv = jfwd_v(qpos_arr_jnp, qvel_arr_jnp)
exec_time_jax = time.time() - curr_time
print('Exec time PY FWD 			: ', exec_time_jax)

curr_time = time.time()
res_jax_jacbq = jbwd_q(qpos_arr_jnp, qvel_arr_jnp)
res_jax_jacbv = jbwd_v(qpos_arr_jnp, qvel_arr_jnp)
exec_time_jax = time.time() - curr_time
print('Exec time PY BWD 			: ', exec_time_jax)

print('JACFWDq - JACREVq			: ', np.max(np.abs(res_jax_jacq-res_jax_jacbq))/ np.max(np.abs(res_jax_jacq)))
print('JACFWDv - JACREVv			: ', np.max(np.abs(res_jax_jacv-res_jax_jacbv))/ np.max(np.abs(res_jax_jacv)))
print('JACFWDq - JACPYq			: ', np.max(np.abs(res_jax_jacq-res_py_jacq))/np.max(np.abs(res_jax_jacq))  )
print('JACFWDv - JACPYv			: ', np.max(np.abs(res_jax_jacv-res_py_jacv))/np.max(np.abs(res_jax_jacv))  )

# assert np.max(np.abs(res_jax_jacq-res_jax_jacbq))/ np.max(np.abs(res_jax_jacq)) <= 1e-6, 'Jacobian wrt q using fwd and rev does not match'
# assert np.max(np.abs(res_jax_jacv-res_jax_jacbv))/ np.max(np.abs(res_jax_jacv)) <= 1e-6, 'Jacobian wrt v using fwd and rev does not match'
# assert np.max(np.abs(res_jax_jacq-res_py_jacq))/np.max(np.abs(res_jax_jacq)) <= 1e-5, 'Jacobian wrt q does not match finite difference'
# assert np.max(np.abs(res_jax_jacv-res_py_jacv))/np.max(np.abs(res_jax_jacv)) <= 1e-5, 'Jacobian wrt v does not match finite difference'


print('########################## Test JVP(JVP(M(q)^-1 v)) ##########################')
zero_nq = np.zeros((n_data, n_pos))
zero_nv = np.zeros((n_data, n_vel))
zero_nq_jnp = jnp.zeros((n_data, n_pos))
zero_nv_jnp = jnp.zeros((n_data, n_vel))
curr_time = time.time()
err_est_jac, res_py_jacjac = py_iMq_v_jvp_jvp(qpos_arr, qvel_arr, qpos_arr_tan, qvel_arr_tan, qpos_arr, qvel_arr, qpos_arr_tan, qvel_arr_tan) 
exec_time_py = time.time() - curr_time
print('Error Estimate 2JVP			: {}'.format(err_est_jac))
print('Exec time PY        			: ', exec_time_py)

inv_mass_prod_vector_jit_jvp_jvp = jax.jit(jax.vmap(lambda x,y,xt,yt,x_tan,y_tan,xt_tan,yt_tan : jax.jvp(inv_mass_prod_vector_jit_jvp, (x,y,xt,yt), (x_tan,y_tan,xt_tan,yt_tan))[1]), device=jax.devices('cpu')[0])
f_val, res_jax_jacjac = inv_mass_prod_vector_jit_jvp_jvp(qpos_arr_jnp, qvel_arr_jnp, qpos_arr_tan_jnp, qvel_arr_tan_jnp, qpos_arr_tan_jnp, zero_nv_jnp, zero_nq_jnp, zero_nv_jnp)
curr_time = time.time()
f_val, res_jax_jacjac = inv_mass_prod_vector_jit_jvp_jvp(qpos_arr_jnp, qvel_arr_jnp, qpos_arr_tan_jnp, qvel_arr_tan_jnp, qpos_arr_jnp, qvel_arr_jnp, qpos_arr_tan_jnp, qvel_arr_tan_jnp)
exec_time_jax = time.time() - curr_time
print('Exec time JAX 				: ', exec_time_jax)
print('Error 2JVP((M^-1(q)v))			: ', np.max(np.abs(np.asarray(res_py_jacjac)-res_jax_jacjac))/np.max(np.abs(res_py_jacjac)))


print('######################### Test GRAD(JVP(M(q)^-1 v)) ##########################')

def random_try(qpos, qvel, qpos_t, qvel_t):
	res_eval, res_jax_der = inv_mass_prod_vector_jit_jvp(qpos, qvel, qpos_t, qvel_t)
	return jnp.sum(res_jax_der)

def py_random_try(qpos, qvel, qpos_t, qvel_t):
	def temp_f (x_input):
		res_arr = np.zeros((qpos.shape[0], n_vel))
		for i in range(qpos.shape[0]):
			x = x_input[i*(n_pos+n_vel)*2:(i+1)*(n_pos+n_vel)*2]
			err_1, res = py_iMq_v_jvp(x[:n_pos], x[n_pos:(n_pos+n_vel)], x[(n_pos+n_vel):(2*n_pos+n_vel)], x[(2*n_pos+n_vel):] )
			res_arr[i,:] = res
		return np.sum(res_arr)
	grad_f = nd.Gradient(temp_f, full_output=True)
	x_input = np.hstack((qpos, qvel, qpos_t, qvel_t)).flatten()
	val, err = grad_f(x_input)
	return np.max(err.error_estimate), val.reshape((qpos.shape[0],(n_pos+n_vel)*2))

curr_time = time.time()
err, res_py_grad = py_random_try(qpos_arr, qvel_arr, qpos_arr_tan, qvel_arr_tan)
exec_time_py = time.time() - curr_time
print('Error Estimate JVP  			: {}'.format(err))
print('Exec time PY        			: ', exec_time_py)

res_py_gradq = res_py_grad[:, :n_pos]
res_py_gradv = res_py_grad[:, n_pos:(n_pos+n_vel)]
res_py_gradq1 = res_py_grad[:, (n_pos+n_vel):(2*n_pos+n_vel)]
res_py_gradv1 = res_py_grad[:, (2*n_pos+n_vel):]

grad_fun_jvp = jax.jit(jax.grad(random_try, argnums=(0,1,2,3)), device=jax.devices('cpu')[0])
res_jax_gradq, res_jax_gradv, res_jax_gradq1, res_jax_gradv1 = grad_fun_jvp(qpos_arr_jnp, qvel_arr_jnp, qpos_arr_tan_jnp, qvel_arr_tan_jnp)
curr_time =  time.time()
res_jax_gradq, res_jax_gradv, res_jax_gradq1, res_jax_gradv1 = grad_fun_jvp(qpos_arr_jnp, qvel_arr_jnp, qpos_arr_tan_jnp, qvel_arr_tan_jnp)
exec_time_jax = time.time() - curr_time
print('Exec time JAX 				: ', exec_time_jax)

print('Error grad_q  jac (M^-1(q)v)		: ', np.max(np.abs(np.asarray(res_jax_gradq)-res_py_gradq))/np.max(np.abs(res_py_gradq)))
print('Error grad_v  jac (M^-1(q)v)		: ', np.max(np.abs(np.asarray(res_jax_gradv)-res_py_gradv))/np.max(np.abs(res_py_gradv)))
print('Error grad_q1 jac (M^-1(q)v)		: ', np.max(np.abs(np.asarray(res_jax_gradq1)-res_py_gradq1))/np.max(np.abs(res_py_gradq1)))
print('Error grad_v1 jac (M^-1(q)v)		: ', np.max(np.abs(np.asarray(res_jax_gradv1)-res_py_gradv1))/np.max(np.abs(res_py_gradv1)))

print('################################ Test dq/dt ##################################')
compute_qpos_der_aux = jax.jit(jax.vmap(partial(compute_qpos_der, quat_arr=quat_arr)))
res_jax_der_vel = compute_qpos_der_aux(qpos_arr_jnp, qvel_arr_jnp)
curr_time = time.time()
compute_qpos_der_aux(qpos_arr_jnp, qvel_arr_jnp)
der_time = time.time() - curr_time
print('Exec time JAX       			: ', der_time)
qpos_arr_next = qpos_arr_jnp[:-1,:] + res_jax_der_vel[1:,:] * actual_dt

print('Error dq/dt         			: ', np.max(np.abs(np.asarray(qpos_arr_next)-qpos_arr_jnp[1:])) / np.max(np.abs(qpos_arr_next)))
# assert np.max(np.abs(np.asarray(qpos_arr_next)-qpos_arr_jnp[1:])) / np.max(np.abs(qpos_arr_next)) <= 1e-5, 'Derivative of q is uncertain'
print('##############################################################################')