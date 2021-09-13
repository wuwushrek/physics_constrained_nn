import functools

# Import relative to JAX
import jax
import jax.numpy as jnp

# Import relative to BRAX
# import brax
from brax import envs
from brax.physics.integrators import kinetic, potential
from brax.physics.base import QP, P, vec_to_np
from brax.physics.system import System
from brax.physics.math import ang_to_quat

from flax import struct
import jax
from jax import ops
import jax.numpy as jnp

from brax.physics import bodies
from brax.physics import config_pb2
from brax.physics import math
from brax.physics.base import P, QP, euler_to_quat, vec_to_np, take
from brax.physics.math import safe_norm

from brax.io import html
from tqdm.auto import tqdm
import pickle

def index_active_posrot(brax_sys):
	""" This function computes the indexes of the states of the system that are active during the simulation
		:param brax_sys : A brax object of class System representing a dynamical system
	"""
	pos_indx, quat_indx, rot_indx = [], [], []
	pos_indx = brax_sys.active_pos.astype(bool)
	rot_indx = brax_sys.active_rot.astype(bool)
	quat_indx = jnp.sum(brax_sys.active_rot, axis=1) >= 1
	quat_indx = jnp.hstack((quat_indx.reshape(-1,1), brax_sys.active_rot.astype(bool)))
	return pos_indx, quat_indx, rot_indx

# def kinetic_new(qp, dt, active_pos, active_rot):
# 	@jax.vmap
# 	def op(qp: QP, active_pos: jnp.ndarray, active_rot: jnp.ndarray) -> QP:
# 		pos = qp.pos + qp.vel * dt * active_pos
# 		rot_at_ang_quat = ang_to_quat(qp.ang * active_rot)
# 		rot = jnp.matmul(
# 			jnp.matmul(jnp.eye(4) + .5 * dt * rot_at_ang_quat, qp.rot),
# 			jnp.eye(4) - .5 * dt * rot_at_ang_quat)
# 		# rot = rot / jnp.linalg.norm(rot)
# 		return QP(pos=pos, rot=rot, vel=qp.vel, ang=qp.ang)
# 	return op(qp, active_pos, active_rot)

# Old and partially wrong version of side information
# @functools.partial(jax.jit, static_argnums=(1,))
def qdot_sideinfo(qp : QP, env_sys : System):
	""" This function computes \dot(pos, rot) given the current configuration and the system information
		:param qp : The current state of the system
		:param env_sys : The brax System object encoding the dynamical system
	"""
	fn = lambda dt : kinetic(env_sys.config, qp, dt, env_sys.active_pos, env_sys.active_rot)
	# fn = lambda dt : kinetic_new(qp, dt, env_sys.active_pos, env_sys.active_rot)
	return jax.jvp(fn, (0.0,), (1.0,))[1]

# @functools.partial(jax.jit, static_argnums=(1,))
def coriolis_sideinfo(qp : QP, env_sys : System):
	""" This function computes -M(q)^-1 c(q,p) where c is the coriolis force --> We should always check if this function is compatible with latex BRAX
		:param qp : The current state of the system
		:param env_sys : The brax System object encoding the dynamical system
	"""
	dp_j = env_sys.joint_revolute.apply(qp)
	dp_j += env_sys.joint_universal.apply(qp)
	dp_j += env_sys.joint_spherical.apply(qp)
	fn = lambda dt : potential(env_sys.config, qp, dp_j, dt, env_sys.active_pos, env_sys.active_rot)
	return jax.jvp(fn, (0.0,), (1.0,))[1]

def qdot_sideinfo_simple(qp : QP, env_sys : System):
	""" This function computes \dot(pos, rot) given the current configuration and the system information
		:param qp : The current state of the system
		:param env_sys : The brax System object encoding the dynamical system
	"""
	dp_pos = qp.vel * env_sys.active_pos
	@jax.vmap
	def op(qp_ang : jnp.ndarray, qp_rot : jnp.ndarray, active_rot : jnp.ndarray):
		return jnp.matmul(ang_to_quat(qp_ang * active_rot), qp_rot)
	return dp_pos, op(qp.ang, qp.rot,  env_sys.active_rot)

def coriolis_sideinfo_simple(qp : QP, env_sys : System):
	""" This function computes -M(q)^-1 c(q,p) where c is the coriolis force --> We should always check if this function is compatible with latex BRAX
		:param qp : The current state of the system
		:param env_sys : The brax System object encoding the dynamical system
	"""
	dp_j = env_sys.joint_revolute.apply(qp)
	dp_j += env_sys.joint_universal.apply(qp)
	dp_j += env_sys.joint_spherical.apply(qp)
	dp_vel = (env_sys.config.velocity_damping * qp.vel + (dp_j.vel + vec_to_np(env_sys.config.gravity))) * env_sys.active_pos
	dp_ang = (env_sys.config.angular_damping * qp.ang + dp_j.ang) * env_sys.active_rot
	return dp_vel, dp_ang


# @functools.partial(jax.jit, static_argnums=(2,))
def actuator_sideinfo(qp : QP, act : jnp.ndarray, env_sys : System):
	""" This function computes M(q)^-1 tau(q,act) where tau denotes the effect of the control input
		:param qp : The current state of the system
		:param act : The action (angle or torque) to be applied
		:param env_sys : The brax System object encoding the dynamical system
	"""
	dp_a = env_sys.angle_1d.apply(qp, act)
	dp_a += env_sys.angle_2d.apply(qp, act)
	dp_a += env_sys.angle_3d.apply(qp, act)
	dp_a += env_sys.torque_1d.apply(qp, act)
	dp_a += env_sys.torque_2d.apply(qp, act)
	dp_a += env_sys.torque_3d.apply(qp, act)
	fn = lambda dt : potential(env_sys.config, qp, dp_a, dt, env_sys.active_pos, env_sys.active_rot)
	return jax.jvp(fn, (0.0,), (1.0,))[1]

def state2qp_aux(state_aux, bool_indx):
	""" Compute a vector of size bool_indx where the element specified
		by the True index of bool_indx should be set to state_aux and the rest to zero
		:param state_aux : A vector
		:param bool_indx : A boolean vector with the number of True = shape(state_aux)
	"""
	mRes = jnp.zeros(shape=(bool_indx.shape[0],))
	return jax.ops.index_update(mRes, bool_indx, state_aux)

def fast_qp2state(m_qp : QP, pos_indx : jnp.ndarray, quat_indx : jnp.ndarray, ang_indx : jnp.ndarray):
	"""This function provides a translation from a QP (not batched) active state representation to a 1D arary.
		The returned 1D array provides ONLY the active states of the dynamical system.
		Warning : Care must be taken in batch setting where batch should only be done on the first argument : jax.vmap(qp2state, in_axes=(0,None))
		:param m_qp : The QP state of the system
		:param pos_indx : The boolean of the active position in QP.pos or QP.vel
		:param quat_indx : The boolean of the active rotation in QP.rot
		:param ang_indx : The boolean of the active angular velocity in QP.ang
	"""
	return jnp.concatenate((m_qp.pos[pos_indx].ravel(), m_qp.rot[quat_indx].ravel(), m_qp.vel[pos_indx].ravel(), m_qp.ang[ang_indx].ravel()))

def qp2state(m_qp : QP, qp_indx : jnp.ndarray):
	"""This function provides a translation from a QP (not batched) active state representation to a 1D arary.
		The returned 1D array provides ONLY the active states of the dynamical system.
		Warning : Care must be taken in batch setting where batch should only be done on the first argument : jax.vmap(qp2state, in_axes=(0,None))
		:param m_qp : The QP state of the system
		:param qp_indx : A 1D boolean array specifying the active or inactive states
	"""
	return jnp.concatenate((m_qp.pos.ravel(), m_qp.rot.ravel(), m_qp.vel.ravel(), m_qp.ang.ravel()))[qp_indx]

def state2qp_merge(state : jnp.ndarray, arg_extra_state : jnp.ndarray, qp_indx : jnp.ndarray, qp_indx_neg : jnp.ndarray, qp_base : QP):
	state_n = jnp.zeros(qp_indx.shape)
	state_n = jax.ops.index_update(state_n, qp_indx, state)
	state_n = jax.ops.index_update(state_n, qp_indx_neg, arg_extra_state)
	return QP(pos=state_n[:qp_base.pos.size].reshape(qp_base.pos.shape), 
			  rot=state_n[qp_base.pos.size:(qp_base.pos.size+qp_base.rot.size)].reshape(qp_base.rot.shape), 
			  vel=state_n[(qp_base.pos.size+qp_base.rot.size):(qp_base.pos.size+qp_base.rot.size+qp_base.vel.size)].reshape(qp_base.vel.shape), 
			  ang=state_n[(qp_base.pos.size+qp_base.rot.size+qp_base.vel.size):].reshape(qp_base.ang.shape))


def fast_state2qp(x_pos: jnp.ndarray, x_rot : jnp.ndarray, x_vel: jnp.ndarray, x_ang: jnp.ndarray,
		qp_base : jnp.ndarray, pos_indx : jnp.ndarray, quat_indx : jnp.ndarray, ang_indx : jnp.ndarray):
	""" This function provides a translation from a 1D active state representation to a QP object.
		It assumes that qp_base provides the value for the inactive state of the dynamical system.
		Warning : Care must be taken in batch setting where batch should only be done on the first argument : jax.vmap(qp2state, in_axes=(0,None, None))
		:param state : The active state of the system as a 1D array
		:param qp_base : The base QP object with value for inactive state
		:param qp_indx : A 1D boolean array specifying the active or inactive states
	"""
	mPos = jax.ops.index_update(qp_base.pos.ravel(), pos_indx, x_pos).reshape(qp_base.pos.shape)
	mVel = jax.ops.index_update(qp_base.vel.ravel(), pos_indx, x_vel).reshape(qp_base.vel.shape)
	mQuat = jax.ops.index_update(qp_base.rot.ravel(), quat_indx, x_rot).reshape(qp_base.rot.shape)
	mAng = jax.ops.index_update(qp_base.ang.ravel(), ang_indx, x_ang).reshape(qp_base.ang.shape)
	return QP(pos=mPos, rot=mQuat, vel=mVel, ang=mAng)


def state2qp(state : jnp.ndarray,  qp_base : QP, qp_indx : jnp.ndarray):
	""" This function provides a translation from a 1D active state representation to a QP object.
		It assumes that qp_base provides the value for the inactive state of the dynamical system.
		Warning : Care must be taken in batch setting where batch should only be done on the first argument : jax.vmap(qp2state, in_axes=(0,None, None))
		:param state : The active state of the system as a 1D array
		:param qp_base : The base QP object with value for inactive state
		:param qp_indx : A 1D boolean array specifying the active or inactive states
	"""
	state_n = jnp.concatenate((qp_base.pos.ravel(), qp_base.rot.ravel(), qp_base.vel.ravel(), qp_base.ang.ravel()))
	state_n = jax.ops.index_update(state_n, qp_indx, state)
	return QP(pos=state_n[:qp_base.pos.size].reshape(qp_base.pos.shape), 
			  rot=state_n[qp_base.pos.size:(qp_base.pos.size+qp_base.rot.size)].reshape(qp_base.rot.shape), 
			  vel=state_n[(qp_base.pos.size+qp_base.rot.size):(qp_base.pos.size+qp_base.rot.size+qp_base.vel.size)].reshape(qp_base.vel.shape), 
			  ang=state_n[(qp_base.pos.size+qp_base.rot.size+qp_base.vel.size):].reshape(qp_base.ang.shape))

def generate_data(m_rng, act_lb, act_ub, jit_env_reset, jit_env_step, qp_indx, 
					num_data=100, max_length=1000, repeat_u = 1, control_policy=None, 
					n_rollout=1, merge_traj=True):
	""" Generate trajectories from a uniform distribution of the control input and output the trajectories
		:param m_rng : A random key for the random key generator module
		:param act_lb : The array of lower bound on the control inputs
		:param act_ub : The array of upper bound on the control input
		:param jit_env_reset : A function to reset the environment
		:param jit_env_step : A function to apply one time step in the environment
		:param qp_indx : The 1D array of active indexes in the states returned by the environment
		:param num_data : The total number of trajectories
		:param max_length : The length of the trajectory before resetting the environment
		:param repeat_u : Number of time to repeat the control value u (in case of no subset)
		:param control_policy : The policy from which to generate the action
		:param n_rollout : The number of rollout in the trajectories
	"""
	# Lower bound and upper bound on the control input should be arrays
	try:
		shape_u1, shape_u2 = act_lb.shape, act_ub.shape
		assert shape_u1 == shape_u2, 'Upper and lower bound of the control input must be the same shape'
	except:
		print('The bounds on the control should be array with a shape attribute')
		raise

	# Jit the function that converts a QP to a 1d array --> Require the position of active state
	jit_qp2state = jax.jit(lambda x : qp2state(x, qp_indx))

	# Jit the function that converts a QP to a nonactive 1d array
	qp_indx_neg = ~qp_indx
	jit_qp2nonactivestate = jax.jit(lambda x : qp2state(x, qp_indx_neg))

	# Load the policy to generate the trajectories
	if control_policy is not None:
		# Parameters of the policy network
		num_observation = control_policy['n_obs']
		normalize = control_policy['normalize']
		num_action = control_policy['n_act']
		# Open the file containing the weight of the NN
		f = open(control_policy['file'], 'rb')
		params = pickle.load(f)
		f.close()
		# Import the function to make the adequate Neural network
		from brax.training import ppo, sac
		if control_policy['policy'] == 'ppo':
			_, inference_fn = ppo.make_params_and_inference_fn(num_observation, num_action, normalize)
		else:
			assert control_policy['policy'] == 'sac'
			_, inference_fn = sac.make_params_and_inference_fn(num_observation, num_action, normalize)
		# Jit the inference function
		inference_fn = jax.jit(inference_fn)

	# Store the active and the inactive states
	res_x, res_x_extra_args = [], []
	res_xnext, res_u = [[] for r in range(n_rollout)], [[] for r in range(n_rollout)]
	
	# Iterate over the number of training/testing data
	for i in tqdm(range(num_data)):
		# Split the random number generator
		m_rng, subkey = jax.random.split(m_rng)
		# Reset the environment with the new subkey
		state = jit_env_reset(subkey)
		# Get the active state from the current QP
		state_repr = jit_qp2state(state.qp)
		# Get the inactive state from the current QP
		state_past_nactive = jit_qp2nonactivestate(state.qp)
		temp_resx = [state_repr]
		temp_resx_extra = [state_past_nactive]
		temp_resu = []
		for j in tqdm(range(max_length+n_rollout-1), leave=False):
			if j % repeat_u == 0: # Generate a new control input every repeat_u iteration
				m_rng, subkey = jax.random.split(m_rng)
				if control_policy is not None: # If the policy is given, use it
					noise_val = jax.random.uniform(subkey, act_lb.shape, minval=act_lb, maxval=act_ub)
					m_act = inference_fn(params, state.obs, m_rng) + noise_val
				else: # If the policy is not given, generate bounded random control inputs
					m_act = jax.random.uniform(subkey, shape=act_lb.shape, minval=act_lb, maxval=act_ub)
			# Do one step in the environment
			state = jit_env_step(state, m_act)
			# Save the active state from the next QP
			state_repr_next = jit_qp2state(state.qp)
			# Save the current state and the current control applied
			temp_resu.append(m_act)
			# Save the activate state 
			temp_resx.append(state_repr_next)
			# Save the inactive state from the next QP
			state_next_nactive = jit_qp2nonactivestate(state.qp)
			temp_resx_extra.append(state_next_nactive)
			# # Check if the inactive state are actually inactive
			# print(qp_indx)
			# print(state_past_nactive)
			# print(state_next_nactive)
			# print()
			# assert jnp.max(jnp.abs(state_past_nactive-state_next_nactive)) <= 1e-7, 'Non active state changes too much : {}'.format(jnp.max(jnp.abs(state_past_nactive-state_next_nactive)))
			# state_past_nactive = state_next_nactive
		# Append the trajectory into the result
		# temp_resx = jnp.array(temp_resx)
		# temp_resu = jnp.array(temp_resu)
		# temp_resx_extra = jnp.array(temp_resx_extra)
		if merge_traj:
			res_x.extend(temp_resx[:max_length])
			res_x_extra_args.extend(temp_resx_extra[:max_length])
		else:
			res_x.append(temp_resx[:max_length])
			res_x_extra_args.append(temp_resx_extra[:max_length])
		# Fill and save the rollout data
		for j, r in enumerate(res_u):
			if merge_traj:
				r.extend(temp_resu[j:(j+max_length)])
			else:
				r.append(temp_resu[j:(j+max_length)])
		for j, r in enumerate(res_xnext):
			if merge_traj:
				r.extend(temp_resx[(j+1):(j+1+max_length)])
			else:
				r.append(temp_resx[(j+1):(j+1+max_length)])
	return res_x, res_x_extra_args, res_u, res_xnext, m_rng

def visualize_traj(env, batch_state, qp_base, qp_indx, filename : str = ''):
	""" A function to visualize the evolution of the states in the environment
		:param env : A brax environment
		:param batch_state : (The trajectory to visualize, Extra trajectory information)
		:param qp_base : The base QP object with value for inactive state
		:param qp_indx : A 1D boolean array specifying the active or inactive states
		:param filename : The directory to save the HTML file denonstrating the trajectory
	"""
	# Jit the function that converts a QP to a nonactive 1d array
	qp_indx_neg = ~qp_indx
	# Extract the state and the extra trajectory
	xstate, xargsstate = batch_state
	assert type(xstate) == list, 'The state should be a list'
	# List to store the result
	qp_trajs = []
	# Trasnform a state and the extra state information
	jit_state2qp = jax.jit(lambda x, y : state2qp_merge(x, y, qp_indx, qp_indx_neg, qp_base))
	for i, state, stateargs in tqdm(zip(range(len(xstate)), xstate, xargsstate), total=len(xstate)):
		qp_trajs.append(jit_state2qp(state, stateargs))
	if not filename == '':
		html.save_html(filename, env.sys, qp_trajs)
	return html.render(env.sys, qp_trajs)

def visualize(sys, qps):
	"""Renders a 3D visualization of the environment."""
	return HTML(html.render(sys, qps))

# ant -- 5 capsule plane
# humanoid -- 4 capsule plane
# reacher -- 0 contact
# fetch -- 4 box plane
# grasp -- 5 capsule capsule, 1 capsule plane
# cheetah -- 4, capsule plane
# ur5e -- 10 capsule capsule, 5 capsule plane

################################### Collision utils ############################################ (Brax verison dependent)

def capsule_plane_sideinfo(qp : QP, env_sys : System, forceContact : jnp.ndarray) -> P:
	my_self = env_sys.capsule_plane
	if not my_self.pairs:
	  return P(jnp.zeros_like(qp.vel), jnp.zeros_like(qp.ang))

	@jax.vmap
	def apply(cap, cap_end, radius, qp_cap, qp_plane, contact_xyz):
		cap_end_world = qp_cap.pos + math.rotate(cap_end, qp_cap.rot)
		normal = math.rotate(jnp.array([0.0, 0.0, 1.0]), qp_plane.rot)
		pos = cap_end_world - normal * radius
		rpos_off = pos - qp_cap.pos
		rvel = jnp.cross(qp_cap.ang, rpos_off)
		vel = qp_cap.vel + rvel
		penetration = jnp.dot(pos - qp_plane.pos, normal)
		dp = _collide(my_self.config, cap, qp_cap, pos, vel, normal, penetration, contact_xyz)
		colliding = jnp.where(penetration < 0., 1., 0.)
		return dp, colliding
	
	qp_cap = take(qp, my_self.cap.idx)
	qp_plane = take(qp, my_self.plane.idx)
	dp, colliding = apply(my_self.cap, my_self.cap_end, my_self.cap_radius, qp_cap,
						  qp_plane, forceContact)

	# sum across both contact points
	num_bodies = len(my_self.config.bodies)
	colliding = ops.segment_sum(colliding, my_self.cap.idx, num_bodies)
	vel = ops.segment_sum(dp.vel, my_self.cap.idx, num_bodies)
	ang = ops.segment_sum(dp.ang, my_self.cap.idx, num_bodies)

	# equally distribute contact force over possible collision points
	vel = vel / jnp.reshape(1e-8 + colliding, (vel.shape[0], 1))
	ang = ang / jnp.reshape(1e-8 + colliding, (ang.shape[0], 1))

	return P(vel, ang)

def box_plane_sideinfo(qp: QP, env_sys : System, forceContact : jnp.ndarray) -> P:
	"""Returns impulse from a collision between box corners and a static plane.

	Note that impulses returned by this function are *not* scaled by dt when
	applied to parts.  Collision impulses are applied directly as velocity and
	angular velocity updates.

	Args:
	  qp: Coordinate/velocity frame of the bodies.
	  dt: Integration time step length.

	Returns:
	  dP: Delta velocity to apply to the box bodies in the collision.
	  colliding: Mask for each body: 1 = colliding, 0 = not colliding.
	"""
	my_self = env_sys.box_plane
	if not my_self.pairs:
		return P(jnp.zeros_like(qp.vel), jnp.zeros_like(qp.ang))
	
	@jax.vmap
	def apply(box, corner, qp_box, qp_plane, contact_xyz):
		pos, vel = math.to_world(qp_box, corner)
		normal = math.rotate(jnp.array([0.0, 0.0, 1.0]), qp_plane.rot)
		penetration = jnp.dot(pos - qp_plane.pos, normal)
		dp = _collide(my_self.config, box, qp_box, pos, vel, normal, penetration, contact_xyz)
		collided = jnp.where(penetration < 0., 1., 0.)
		return dp, collided
	
	qp_box = take(qp, my_self.box.idx)
	qp_plane = take(qp, my_self.plane.idx)
	dp, colliding = apply(my_self.box, my_self.corner, qp_box, qp_plane, forceContact)

	# collapse/sum across all corners
	num_bodies = len(my_self.config.bodies)
	colliding = ops.segment_sum(colliding, my_self.box.idx, num_bodies)
	vel = ops.segment_sum(dp.vel, my_self.box.idx, num_bodies)
	ang = ops.segment_sum(dp.ang, my_self.box.idx, num_bodies)

	# equally distribute contact force over each box
	vel = vel / jnp.reshape(1e-8 + colliding, (vel.shape[0], 1))
	ang = ang / jnp.reshape(1e-8 + colliding, (ang.shape[0], 1))

	return P(vel, ang)


def _collide(config: config_pb2.Config, body: bodies.Body, qp: QP,
			 pos_c: jnp.ndarray, vel_c: jnp.ndarray, normal_c: jnp.ndarray,
			 penetration: float, contact_xyz: jnp.ndarray) -> P:
	"""Calculates velocity change due to a collision.
			 Args:
			 config: A brax config.
			 body: Body participating in collision
			 qp: State for body
			 pos_c: Where the collision is occuring in world space
			 vel_c: How fast the collision is happening
			 normal_c: Normal vector of surface providing collision
			 penetration: Amount of penetration between part and plane
			 dt: Integration timestep length
			 Returns:
			 dP: The impulse on this body result from the collision.
		"""
	rel_pos_a = pos_c - qp.pos

	# baumgarte_rel_vel = config.baumgarte_erp * penetration / dt
	normal_rel_vel = jnp.dot(normal_c, vel_c)
	rel_vel_d = vel_c - normal_rel_vel * normal_c
	
	# # temp1 = jnp.matmul(body.inertia, jnp.cross(rel_pos_a, normal_c))
	# ang = jnp.dot(normal_c, jnp.cross(temp1, rel_pos_a))

	# Obtain the impulse from the neural network
	impulse = contact_xyz[0]

	# impulse = (-1. * (1. + config.elasticity) * normal_rel_vel - baumgarte_rel_vel) / ((1. / body.mass) + ang)
	
	dp_n = body.impulse(qp, impulse * normal_c, pos_c)
	
	# # apply drag due to friction acting parallel to the surface contact
	# impulse_d = math.safe_norm(rel_vel_d) / ((1. / (body.mass)) + ang)

	# Obtain the drag impulse from the neural network
	impulse_d = contact_xyz[1]

	# The neural network should learn explicitly this constraint -> so to not include
	# # drag magnitude cannot exceed max friction
	# impulse_d = jnp.where(impulse_d < config.friction * impulse, impulse_d,
	# 					config.friction * impulse)

	dir_d = rel_vel_d / (1e-6 + math.safe_norm(rel_vel_d))
	dp_d = body.impulse(qp, -impulse_d * dir_d, pos_c)
	
	# apply collision normal if penetrating, approaching, and oriented correctly
	colliding_n = jnp.where(penetration < 0., 1., 0.)
	colliding_n *= jnp.where(normal_rel_vel < 0, 1., 0.)
	# colliding_n *= jnp.where(impulse > 0., 1., 0.) # -> SHould learn positive normal forces
	
	# apply drag if moving laterally above threshold
	colliding_d = colliding_n
	colliding_d *= jnp.where(math.safe_norm(rel_vel_d) > (1. / 100.), 1., 0.)
	
	# factor of 2.0 here empirically helps object grip
	# TODO: expose friction physics parameters in config
	return dp_n * colliding_n + dp_d * colliding_d * 2.0