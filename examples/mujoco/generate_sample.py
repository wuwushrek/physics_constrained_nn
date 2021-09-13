import logging
import sys
import os 
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root_logger.addHandler(handler)


import numpy as np

import yaml
from collections import namedtuple
import pickle


from physics_constrained_nn.utils import SampleLog

from tqdm import tqdm

# Import the environment for test
from os import path
from dm_control import suite

from dm_control import viewer
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import mjlib

# File path 
import os
current_file_path = os.path.dirname(os.path.realpath(__file__))
xml_path = current_file_path + '/../../../dm_control/dm_control/suite/'

# max_episode_len, num_episodes, n_rollout, init_iter=0):
def generate_data(act_lb, act_ub, env, num_data, max_length, repeat_u=1, control_policy=None, n_rollout=1, merge_traj=True): 
	# Store the active and the inactive states
	res_x = []
	res_xnext, res_u = [[] for r in range(n_rollout)], [[] for r in range(n_rollout)]
	iMq_acc, qacc = [], []

	# Iterate over the number of training/testing data
	for i in tqdm(range(num_data)):
		time_step = env.reset()
		state_repr = np.hstack((env.physics.data.qpos.ravel(), env.physics.data.qvel.ravel()))
		temp_resx = [state_repr]
		temp_iMq_acc, temp_qacc = [], []
		temp_resu = []
		for j in tqdm(range(max_length+n_rollout-1), leave=False):
			if j % repeat_u == 0: # Generate a new control input every repeat_u iteration
				m_act = np.random.uniform(act_lb, act_ub, size=act_lb.shape)
			# Do one step in the environment
			time_step = env.step(m_act)
			# Save the active state from the next QP
			state_repr_next = np.hstack((env.physics.data.qpos.ravel(), env.physics.data.qvel.ravel()))
			# Save the current state and the current control applied
			temp_resu.append(m_act)
			# Save the activate state 
			temp_resx.append(state_repr_next)
			# Update the replac buffer with the resulting accelatrion
			temp_qacc.append(env.physics.data.qacc.ravel())
			# Compute the mass matrix for finding the unknown forces
			mass_matrix = np.ndarray(shape=(env.physics.model.nv**2,),dtype=np.float64, order='C')
			mjlib.mj_fullM(env.physics.model.ptr, mass_matrix, env.physics.data.qM)
			Mval = np.reshape(mass_matrix, (env.physics.model.nv, env.physics.model.nv))
			temp_iMq_acc.append(Mval @ env.physics.data.qacc)
		# Merge iMqacc and qacc
		iMq_acc.extend(temp_iMq_acc[:max_length])
		qacc.extend(temp_qacc[:max_length])
		if merge_traj:
			res_x.extend(temp_resx[:max_length])
		else:
			res_x.append(temp_resx[:max_length])
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
	return res_x, res_u, res_xnext, iMq_acc, qacc


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

	# Define the environment
	domain_name = env_name
	task_name = env_extra_args['task_name']

	np.random.seed(seed_number)

	# Launch the environment
	env = suite.load(domain_name, task_name, task_kwargs={'random': seed_number})
	print ('Load environment\t : ENV_NAME={}, TASK_NAME={}'.format(env_name, task_name))

	# DIsable DM CONTROL substeps if needed
	if disable_substep: # if disable_substep, the time step is only env.sys.config.dt else it is env.sys.config.dt * n_sub_steps
		old_substep_val = int(env._n_sub_steps)
	else: # If not disable_substep, take the full time step: env.sys.config.dt
		old_substep_val = 1
	# Update the time step of the environment
	env._n_sub_steps =  int(env._n_sub_steps / old_substep_val)
	actual_dt = env._n_sub_steps*env.physics.model.opt.timestep
	print ('Load done.')

	n_state = env.physics.model.nq + env.physics.model.nv
	n_control = env.physics.model.nu
	print('################# DM time step = {} -> n_steps = {}, opt_time_steps = {} #################'.format(actual_dt, env._n_sub_steps, env.physics.model.opt.timestep))

	# Training range for the control inputs
	lowU_train_val = lowU_train_val if control_policy is None else -control_policy.get('noise_train',0.0)
	highU_train_val = highU_train_val if control_policy is None else control_policy.get('noise_train',0.0)
	lowU_train = np.array([lowU_train_val for i in range(n_control)]) # actSpec.minimum
	highU_train = np.array([highU_train_val for i in range(n_control)]) # actSpec.maximum

	# Testing range for the control inputs
	lowU_test_val = lowU_test_val if control_policy is None else -control_policy.get('noise_test',0.0)
	highU_test_val = highU_test_val if control_policy is None else control_policy.get('noise_test',0.0)
	lowU_test = np.array([lowU_test_val for i in range(n_control)]) # actSpec.minimum
	highU_test = np.array([highU_test_val for i in range(n_control)]) # actSpec.maximum

	# Get the highest number of trajectories
	max_traj = int(np.max(np.array(num_data_train)))

	# Generate the Training set via the set of trajectory size
	xTrainList, uTrainList, xnextTrainList, iMq_acc, qacc = \
		generate_data(lowU_train, highU_train, env, num_data=max_traj, max_length=max_episode_length,
					repeat_u=old_substep_val, control_policy=control_policy, n_rollout=n_rollout)

	# Need to check the order of magnitude of x qacc with respect to forces instabilities
	print(np.array(qacc).shape, np.array(iMq_acc).shape)
	regTerm =  1.0 / np.mean(np.abs(np.array(qacc).T / np.sum(iMq_acc, axis=1)))
	# regTerm = jnp.max(jnp.abs(Mq_acc), axis=0)
	print('## Quotient F/acc = {}'.format(regTerm))

	# Generate the Test set
	xTest, uTest, xnextTest, _,_ = generate_data(lowU_test, highU_test, env, num_data=num_data_test, max_length=max_episode_length,
										repeat_u=old_substep_val, control_policy=control_policy, n_rollout=n_rollout)

	# Generate the set of colocation points
	xColoc, uColoc, _, _, _ = generate_data(lowU_test - extra_noise_colocation, highU_test + extra_noise_colocation, env, 
										num_data=num_data_colocation, max_length=max_episode_length,
										repeat_u=old_substep_val, control_policy=control_policy, n_rollout=1)

	# Save the output as a NamedTuple
	mLog = SampleLog(xTrain=xTrainList, xTrainExtra=(None, (xColoc, uColoc[0], None)), uTrain=uTrainList, 
			xnextTrain=xnextTrainList, lowU_train=lowU_train, highU_train=highU_train,
			xTest=xTest, xTestExtra=None, uTest=uTest, xnextTest=xnextTest, lowU_test=lowU_test, highU_test = highU_test, 
			env_name=env_name, env_extra_args=env_extra_args, m_rng=seed_number, seed_number=seed_number, 
			qp_indx=None, qp_base=(env.physics.model.nq, env.physics.model.nv, regTerm), n_state=n_state, n_control=n_control,
			actual_dt = actual_dt, control_policy=control_policy, disable_substep=(disable_substep, num_data_train, max_episode_length), n_rollout=n_rollout)

	# Save the log using pickle
	mFile = open(output_file+'.pkl', 'wb')
	pickle.dump(mLog, mFile)
	mFile.close()
	
	print('Env extra args : {}'.format(env_extra_args))
	print('Number inputs\t\t : {}'.format(n_control))
	print('Time step dt\t\t : {}'.format(actual_dt))
	print('Initial seed number\t : {}'.format(seed_number))
	print('Training control range\t : {}'.format([(x,y) for x,y in zip(lowU_train, highU_train)]))
	print('Testing control range\t : {}'.format([(x,y) for x,y in zip(lowU_test, highU_test)]))
	print('Control policy\t : {}'.format(control_policy))
	print('Resulting RNG\t\t : {}'.format(seed_number))
	print('Training size\t\t : {}'.format(num_data_train))
	print('Testing size\t\t : {}'.format(num_data_test))
	print('No. rollout\t\t : {}'.format(n_rollout))
	print('Size data:\t\t Train = {},  Test = {}'.format(len(xTrainList), len(xTest)))

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