import yaml
from collections import namedtuple
import pickle

# Import JAX
import jax
import jax.numpy as jnp

# Import Brax utils functions
from brax import envs

from physics_constrained_nn.utils import SampleLog
from physics_constrained_nn import utils_brax

from tqdm.auto import tqdm

import time


def load_config_yaml(path_config_file, extra_args={}):
	""" Load the yaml configuration file giving the training/testing information
		:param path_config_file : Path to the adequate yaml file
		:param extra_args : Extra argument from the command line
	"""
	# Load the configuration file and append command line arguments if given
	yml_file = open(path_config_file).read()
	m_config = yaml.load(yml_file, yaml.SafeLoader)
	m_config = {**m_config, **extra_args}
	print('################# Configuration file #################')
	print(m_config)
	print('######################################################')

	# Parse all the arguments
	env_name = m_config['env_name'] # Environment name
	env_extra_args = m_config.get('env_extra_args', {})
	output_file = m_config.get('output_file', env_name) # Target file to save the training/TEsting data
	seed_number = m_config.get('seed', 1)

	# Bound on the control applied when training and testing
	lowU_train_val = m_config.get('utrain_lb', -0.2)
	highU_train_val = m_config.get('utrain_ub', 0.2)
	lowU_test_val = m_config.get('utest_lb', -0.5)
	highU_test_val = m_config.get('utest_ub', 0.5)

	# Disable the internal number of substep of brax system
	disable_substep = m_config.get('disable_substep', False)

	# Extract the control policy if given by the user
	control_policy = m_config.get('control_policy', None)

	# Extract the number of rollout 
	n_rollout = m_config.get('n_rollout', 1)

	# Extract the maximum episode length
	max_episode_length = m_config.get('max_episode_length', 1000)

	# Extract the number of trainig data -> A list of number of data
	num_data_train = m_config.get('num_data_train', [2000])

	# Extract the number of testing data
	num_data_test = m_config.get('num_data_test', 500)

	# Check if a video of the training and testing trajectories must be saved
	save_video = m_config.get('save_video', False)

	# Get colocation related information
	num_data_colocation = m_config.get('num_data_colocation', 0)
	extra_noise_colocation = m_config.get('extra_noise_coloc', 0)
	assert num_data_colocation >  0, 'Colocation points are required'

	# Return parameters
	return env_name, env_extra_args, output_file, seed_number, lowU_train_val, highU_train_val, lowU_test_val, highU_test_val,\
			max_episode_length, num_data_train, num_data_test, save_video, disable_substep, control_policy, n_rollout,\
			num_data_colocation, extra_noise_colocation

def main_fn(path_config_file, extra_args={}):
	""" Root function in the main file.
		:param path_config_file : Path to the adequate yaml file
		:param extra_args : Extra argument from the command line
	"""
	# Load the data for generating the training and testing samples
	env_name, env_extra_args, output_file, seed_number, \
		lowU_train_val, highU_train_val, lowU_test_val, highU_test_val,\
			max_episode_length, num_data_train, num_data_test, save_video,\
				disable_substep, control_policy, n_rollout, num_data_colocation, extra_noise_colocation = \
					load_config_yaml(path_config_file, extra_args)

	print ('Load environment\t : {}'.format(env_name))
	env_fn = envs.create_fn(env_name=env_name, **env_extra_args)
	env = env_fn()
	# In JAX, the env.sys.config.dt / env.sys.config.substeps is the actual time step
	# It is better to take that time step because the approach works better with smaller timestep
	if disable_substep: # if disable_substep, the time step is only env.sys.config.dt / env.sys.config.substeps
		old_substep_val = int(env.sys.config.substeps)
	else: # If not disable_substep, take the full time step: env.sys.config.dt
		old_substep_val = 1
	# The max_episode_length should be scale according to the actual substeps
	max_episode_length *= old_substep_val
	env.episode_length *= old_substep_val
	# Update the time step of the environment
	env.sys.config.dt /= old_substep_val
	env.sys.config.substeps = int(env.sys.config.substeps / old_substep_val)
	actual_dt = float(env.sys.config.dt)
	jit_env_reset = jax.jit(env.reset)
	jit_env_step = jax.jit(env.step)
	print ('Load done.')

	# Initial random key generator: m_rng will be the one used and saved across the file
	m_rng = jax.random.PRNGKey(seed=seed_number)

	# Jit the reset function of the environment
	print ('Jit reset function of {}'.format(env_name))
	jit_time = time.time()
	full_state = jit_env_reset(rng=m_rng)
	jit_time = time.time() - jit_time
	print('JIT TIME RESET\t\t : {:.4f}'.format(jit_time))

	# Jit the step function of the environment
	print ('Jit step function of {}'.format(env_name))
	jit_time = time.time()
	rnd_act = jax.random.uniform(m_rng, shape=(env.action_size,), minval=-1.0, maxval=1.0)
	state_1 = jit_env_step(full_state, rnd_act)
	jit_time = time.time() - jit_time
	print('JIT TIME STEP\t\t : {:.4f}'.format(jit_time))

	# Get the full number of states in the environment 
	nbodies = env.sys.num_bodies
	fulln_pos, fulln_rot, fulln_vel, fulln_ang = full_state.qp.pos.size, full_state.qp.rot.size, full_state.qp.vel.size, full_state.qp.ang.size 

	# # Get the time step of the environement
	# actual_dt = env.sys.config.dt

	# Get the actual problem dimension --> remove inactive states from the full states
	pos_indx, quat_indx, rot_indx = utils_brax.index_active_posrot(env.sys)
	qp_indx = jnp.concatenate((pos_indx.ravel(), quat_indx.ravel(),pos_indx.ravel(), rot_indx.ravel()))

	# STore the actual problem state dimension
	n_pos, n_rot, n_vel, n_ang = jnp.sum(pos_indx), jnp.sum(quat_indx), jnp.sum(pos_indx), jnp.sum(rot_indx)
	n_state = int(n_pos + n_rot + n_vel + n_ang)
	n_control = int(env.action_size)

	# Training range for the control inputs
	lowU_train_val = lowU_train_val if control_policy is None else -control_policy.get('noise_train',0.0)
	highU_train_val = highU_train_val if control_policy is None else control_policy.get('noise_train',0.0)
	lowU_train = jnp.array([lowU_train_val for i in range(n_control)]) # actSpec.minimum
	highU_train = jnp.array([highU_train_val for i in range(n_control)]) # actSpec.maximum

	# Testing range for the control inputs
	lowU_test_val = lowU_test_val if control_policy is None else -control_policy.get('noise_test',0.0)
	highU_test_val = highU_test_val if control_policy is None else control_policy.get('noise_test',0.0)
	lowU_test = jnp.array([lowU_test_val for i in range(n_control)]) # actSpec.minimum
	highU_test = jnp.array([highU_test_val for i in range(n_control)]) # actSpec.maximum

	# Get the highest number of trajectories
	max_traj = int(jnp.max(jnp.array(num_data_train)))
	# Generate the Training set via the set of trajectory size
	xTrainList, xTrainList_args, uTrainList, xnextTrainList, m_rng = \
		utils_brax.generate_data(m_rng, lowU_train, highU_train, jit_env_reset, 
			jit_env_step, qp_indx, num_data=max_traj, max_length=max_episode_length,
					repeat_u=old_substep_val, control_policy=control_policy, n_rollout=n_rollout)


	# Generate the Test set
	xTest, xTest_args, uTest, xnextTest, m_rng = utils_brax.generate_data(m_rng, lowU_test, highU_test, jit_env_reset, 
										jit_env_step, qp_indx, num_data=num_data_test, max_length=max_episode_length,
										repeat_u=old_substep_val, control_policy=control_policy, n_rollout=n_rollout)

	# Generate the set of colocation points
	xColoc, _, uColoc, _, m_rng = utils_brax.generate_data(m_rng, lowU_test - extra_noise_colocation, highU_test + extra_noise_colocation, jit_env_reset, 
										jit_env_step, qp_indx, num_data=num_data_colocation, max_length=max_episode_length,
										repeat_u=old_substep_val, control_policy=control_policy, n_rollout=1)

	# Save the output as a NamedTuple
	mLog = SampleLog(xTrain=xTrainList, xTrainExtra=(xTrainList_args, (xColoc, uColoc[0], None)), uTrain=uTrainList, 
			xnextTrain=xnextTrainList, lowU_train=lowU_train, highU_train=highU_train,
			xTest=xTest, xTestExtra=xTest_args, uTest=uTest, xnextTest=xnextTest, lowU_test=lowU_test, highU_test = highU_test, 
			env_name=env_name, env_extra_args=env_extra_args, m_rng=m_rng, seed_number=seed_number, 
			qp_indx=qp_indx, qp_base=full_state.qp, n_state=n_state, n_control=n_control,
			actual_dt = actual_dt, control_policy=control_policy, disable_substep=(disable_substep, num_data_train, max_episode_length), n_rollout=n_rollout)

	# Save the log using pickle
	mFile = open(output_file+'.pkl', 'wb')
	pickle.dump(mLog, mFile)
	mFile.close()
	
	print('Env extra args : {}'.format(env_extra_args))
	print('Number inputs\t\t : {}'.format(n_control))
	print('Number active states\t : {}'.format(n_state))
	print('Time step dt\t\t : {}'.format(actual_dt))
	print('Initial seed number\t : {}'.format(seed_number))
	print('Full state\t\t : nbodies = {}, npos = {}, nrot = {}, nvel = {}, nang = {}'.format(nbodies,fulln_pos, fulln_rot, fulln_vel, fulln_ang))
	print('Active state\t\t : nbodies = {}, npos = {}, nrot = {}, nvel = {}, nang = {}'.format(nbodies, n_pos, n_rot, n_vel, n_ang))
	print('Training control range\t : {}'.format([(x,y) for x,y in zip(lowU_train, highU_train)]))
	print('Testing control range\t : {}'.format([(x,y) for x,y in zip(lowU_test, highU_test)]))
	print('Control policy\t : {}'.format(control_policy))
	print('Resulting RNG\t\t : {}'.format(m_rng))
	print('Training size\t\t : {}'.format(num_data_train))
	print('Testing size\t\t : {}'.format(num_data_test))
	print('No. rollout\t\t : {}'.format(n_rollout))
	print('############# Base state #############')
	print(full_state,'\n')
	print('############# Active state 1D pos -> ang #############')
	print(qp_indx)

	if save_video:
		# Save only the latest trajectory in the list of trajectories
		utils_brax.visualize_traj(env, (xTrainList, xTrainList_args), full_state.qp, qp_indx, filename=output_file+'_video_train.html')
		utils_brax.visualize_traj(env, (xTest, xTest_args), full_state.qp, qp_indx, filename=output_file+'_video_test.html')

if __name__ == "__main__":
	import time
	import argparse
	# python generate_sample.py --cfg reacher_brax/dataset_gen.yaml --output_file reacher_brax/testdata --seed 101 --disable_substep 0 --save_video 1
	# Command line argument for setting parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', required=True, type=str, help='yaml configuration file for training/testing information: see reacher_cfg1.yaml for more information')
	parser.add_argument('--env_name', type=str, default='', help='The name of the brax environemnt')
	parser.add_argument('--output_file', type=str, default='', help='File to save the generated trajectories')
	parser.add_argument('--n_rollout', type=int, default=0, help='Number of rollout step')
	parser.add_argument('--seed', type=int, default=-1, help='The seed for the trajetcories generation')
	parser.add_argument('--max_episode_length', type=int, default=-1, help='Number of time step before resetting the environment -> Depends on disable_substep')
	parser.add_argument('--disable_substep', type=int, default=-1, help='Enable lower time step in the environement')
	parser.add_argument('--save_video', type=int, default=-1, help='Save the video of the training and testing process')
	parser.add_argument('--num_data_train', nargs='+', help='A list containing the number of trajectories for each training set')
	parser.add_argument('--num_data_test', type=int, default=0, help='A list containing the number of trajectories for each testing set')
	args = parser.parse_args()
	args = parser.parse_args()
	m_config_aux = {'cfg' : args.cfg}
	if args.env_name != '':
		m_config_aux['env_name'] = args.env_name
	if args.output_file != '':
		m_config_aux['output_file']  = args.output_file
	if args.n_rollout > 0:
		m_config_aux['n_rollout']  = args.n_rollout
	if args.seed >= 0:
		m_config_aux['seed']  = args.seed
	if args.max_episode_length > 0:
		m_config_aux['max_episode_length']  = args.max_episode_length
	if args.disable_substep >=0:
		m_config_aux['disable_substep'] = args.disable_substep >= 1
	if args.save_video >= 0:
		m_config_aux['save_video'] = args.save_video >= 1
	if args.num_data_test > 0:
		m_config_aux['num_data_test'] = args.num_data_test
	if args.num_data_train is not None and len(args.num_data_train) > 0:
		m_config_aux['num_data_train'] = [int(val) for val in args.num_data_train]
	# print(m_config_aux)
	main_fn(args.cfg, m_config_aux)

# from google.colab import files
# files.download('policy_learned.pkl')