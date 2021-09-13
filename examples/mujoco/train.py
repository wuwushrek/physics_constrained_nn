# import logging
# import sys
# import os 
# root_logger = logging.getLogger()
# root_logger.setLevel(logging.DEBUG)

# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# handler.setFormatter(formatter)
# root_logger.addHandler(handler)

from jax.config import config
config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np

from physics_constrained_nn.mujoco_jax_primitives import initialize_problem_var, iM_product_vect, quaternion_mapping, compute_qpos_der

from physics_constrained_nn.phyconstrainednets import build_learner_with_sideinfo
from physics_constrained_nn.utils import SampleLog, HyperParamsNN, LearningLog
from physics_constrained_nn.utils import _INITIALIZER_MAPPING, _ACTIVATION_FN, _OPTIMIZER_FN

# Import the environment for test
from os import path
from dm_control import suite

from dm_control import viewer
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import mjlib


import yaml
from collections import namedtuple
import pickle
import time

from physics_constrained_nn.phyconstrainednets import build_learner_with_sideinfo
from physics_constrained_nn.utils import SampleLog, HyperParamsNN, LearningLog
from physics_constrained_nn.utils import _INITIALIZER_MAPPING, _ACTIVATION_FN, _OPTIMIZER_FN


import optax

from tqdm.auto import tqdm

# File path 
import os
current_file_path = os.path.dirname(os.path.realpath(__file__))
xml_path = current_file_path + '/../../../dm_control/dm_control/suite/'

def build_params(m_params, output_size=None, input_index=None):
	""" Build the parameters of the neural network from a dictionary
		:param m_params: A dictionary specifying the number of layers and inputs of the neural network
		:param output_size : Provide the size of the output layer of the neural network
		:param input_index : Provide a mapping of the neural network input to the full state represnetation
	"""
	# Some sanity check
	assert (output_size is None and input_index is None) or (input_index is not None and output_size is not None)

	# Store the resulting neural network
	nn_params = dict()

	# Check if the inputs indexes are given -> If yes this is not the nn for the apriori enclosure
	if input_index is not None:
		nn_params['input_index'] = input_index

	# Check of the size of the output layer is given -> If yes thus us not the nn for the apriori enclosure
	if output_size is not None:
		nn_params['output_sizes'] = (*m_params['output_sizes'], output_size)
	else:
		nn_params['output_sizes'] = m_params['output_sizes']

	# Activation function
	nn_params['activation'] = _ACTIVATION_FN[m_params['activation']]

	# Initialization of the biais values
	b_init_dict = m_params['b_init']
	nn_params['b_init'] = _INITIALIZER_MAPPING[b_init_dict['initializer']](**b_init_dict['params'])

	# Initialization of the weight values
	w_init_dict = m_params['w_init']
	nn_params['w_init'] = _INITIALIZER_MAPPING[w_init_dict['initializer']](**w_init_dict['params'])

	return nn_params

def build_learner(paramsNN, env=None, baseline='base'):
	""" Given a structure like HyperParamsNN, this function generates the functions to compute the loss,
		to update the weight and to predict the next state
		:param paramsNN 	: HyperParamsNN structure proving the parameters to build the NN
		:param env 			: The name of the environment that we are operating on
		:param baseline 	: The odesolver to use for the estimation of the next state
	"""

	# The optimizer for gradient descent over the loss function -> See yaml for more details
	lr = paramsNN.optimizer['learning_rate_init'] 		# Optimizer initial learning rate
	lr_end = paramsNN.optimizer['learning_rate_end'] 	# Optimizer end learning rate

	# Customize the gradient descent algorithm
	chain_list = [_OPTIMIZER_FN[paramsNN.optimizer['name']](**paramsNN.optimizer.get('params', {}))]

	# Add weight decay if enable
	decay_weight = paramsNN.optimizer.get('weight_decay', 0.0)
	if decay_weight > 0.0:
		chain_list.append(optax.add_decayed_weights(decay_weight))

	# Add scheduler for learning rate decrease --> Linear decrease given bt learning_rate_init and learning_rate_end
	m_schedule = optax.linear_schedule(-lr, -lr_end, paramsNN.num_gradient_iterations)
	chain_list.append(optax.scale_by_schedule(m_schedule))

	# Add gradient clipping if enable
	grad_clip_coeff = paramsNN.optimizer.get('grad_clip', 0.0)
	if grad_clip_coeff > 0.0:
		chain_list.append(optax.adaptive_grad_clip(clipping=grad_clip_coeff))

	# Build the solver finally
	opt = optax.chain(*chain_list)

	# The random number generator
	rng_gen = jax.random.PRNGKey(seed=paramsNN.seed_init)

	# Build nn_params using the input in the yaml file
	dict_params = paramsNN.nn_params.copy()

	# Extract the type of side information
	type_sideinfo = dict_params.pop('type_sideinfo', None)
	assert type_sideinfo is not None, 'The type of side information is not specified'
	###############################################################

	# Extract the apriori enclosure neural network
	apriori_net = dict_params.pop('apriori_encl', None)
	assert apriori_net is not None, 'The params for the apriori nnn are not specified'
	apriori_net = baseline if (baseline =='base' or baseline =='rk4') else build_params(apriori_net) 
	tqdm.write("{}\n".format(apriori_net))
	###############################################################

	#################################################################
	## This is specific to the type of environemnt
	## Specific parameters for the case side information is given
	# Get the actual problem dimension --> remove inactive states from the full states
	# Initialize the parameters for the model
	n_pos, n_vel, regTerm = paramsNN.qp_base
	# Find the quaternion distribution over the model
	quat_arr = quaternion_mapping(n_pos)
	compute_qpos_der_partial = lambda qpos, qvel : compute_qpos_der(qpos, qvel, quat_arr)
	print('Quaternion address: ', quat_arr)

	###############################################################

	# Build the nn depending on the type of side information
	if type_sideinfo == 0:

		# assert len(dict_params) == 1, 'There should be only one remaining key'
		paramsNN.pen_constr['num_ineq_constr'] = 0
		paramsNN.pen_constr['num_eq_constr'] = 0

		# Specify if we train usin constraints
		train_with_constraints = False

		# Build the neural network parameter
		nn_params = {'vector_field' : build_params(dict_params['vector_field'], 
						input_index=jnp.array([i for i in range(paramsNN.n_state+paramsNN.n_control)]), 
						output_size=paramsNN.n_state) }

		# Build the learner main functions
		params, pred_xnext, (loss_fun, loss_fun_constr), (update, update_lagrange) = \
								build_learner_with_sideinfo(rng_gen, opt, paramsNN.model_name, paramsNN.actual_dt,
												paramsNN.n_state, paramsNN.n_control, nn_params, apriori_net, known_dynamics=None, 
												constraints_dynamics=None, pen_l2= paramsNN.pen_l2, pen_constr=paramsNN.pen_constr, 
												batch_size=paramsNN.batch_size, train_with_constraints=train_with_constraints, normalize=False)

	################################################################
	elif type_sideinfo == 1:

		# Build the neural network parameter
		nn_params = {'vector_field' : build_params(dict_params['vector_field'], 
						input_index=jnp.array([i for i in range(paramsNN.n_state+paramsNN.n_control)]), 
						output_size=n_vel)}

		# Specify if we train using constraints
		train_with_constraints = False

		paramsNN.pen_constr['num_ineq_constr'] = 0
		paramsNN.pen_constr['num_eq_constr'] = 0

		# Define the function 'f' providing the side information -- Not vectorized -> VMAP should be applied
		# TODO: Normalize the quaternion inputs for N-step control
		def sideinfo(x, u, extra_args=None, vector_field=None): # Not vmap yet
			qpos = x[:n_pos]
			qvel = x[n_pos:]
			qdot = compute_qpos_der_partial(qpos, qvel)
			return jnp.hstack((qdot, vector_field))

		# # Build the learner main functions
		params, pred_xnext, (loss_fun, loss_fun_constr), (update, update_lagrange) = \
									build_learner_with_sideinfo(rng_gen, opt, paramsNN.model_name, 
												paramsNN.actual_dt, paramsNN.n_state, paramsNN.n_control,  
												nn_params, apriori_net, known_dynamics=jax.vmap(sideinfo), 
												constraints_dynamics=None, pen_l2= paramsNN.pen_l2,
												pen_constr=paramsNN.pen_constr, batch_size=paramsNN.batch_size,
												train_with_constraints=train_with_constraints, normalize=False)

	#################################################################
	elif type_sideinfo == 2:

		# Build the neural network parameter
		nn_params = {'vector_field' : build_params(dict_params['vector_field'], 
						input_index=jnp.array([i for i in range(paramsNN.n_state+paramsNN.n_control)]), 
						output_size=n_vel)}

		# Specify if we train using constraints
		train_with_constraints = False

		paramsNN.pen_constr['num_ineq_constr'] = 0
		paramsNN.pen_constr['num_eq_constr'] = 0

		# Define the function 'f' providing the side information -- Not vectorized -> VMAP should be applied
		# TODO: Normalize the quaternion inputs for N-step control
		def sideinfo(x, u, extra_args=None, vector_field=None): # Not vmap yet
			qpos = x[:, :n_pos]
			qvel = x[:, n_pos:]
			qdot = jax.vmap(compute_qpos_der_partial)(qpos, qvel)
			vTerm = iM_product_vect(qpos, vector_field * regTerm)
			return jnp.hstack((qdot, vTerm))

		# # Build the learner main functions
		params, pred_xnext, (loss_fun, loss_fun_constr), (update, update_lagrange) = \
									build_learner_with_sideinfo(rng_gen, opt, paramsNN.model_name, 
												paramsNN.actual_dt, paramsNN.n_state, paramsNN.n_control,  
												nn_params, apriori_net, known_dynamics=sideinfo, 
												constraints_dynamics=None, pen_l2= paramsNN.pen_l2,
												pen_constr=paramsNN.pen_constr, batch_size=paramsNN.batch_size,
												train_with_constraints=train_with_constraints, normalize=False)
	else:
		raise("Not implemented yet !")

	# Jit the functions for the learner
	pred_xnext = jax.jit(pred_xnext)
	loss_fun = jax.jit(loss_fun)
	update = jax.jit(update)
	update_lagrange = jax.jit(update_lagrange)
	loss_fun_constr = jax.jit(loss_fun_constr)
	return rng_gen, opt, params, pred_xnext, loss_fun, update, update_lagrange, loss_fun_constr, train_with_constraints

def load_config_file(path_config_file, extra_args={}):
	""" Load the yaml configuration file giving the training/testing information and the neural network params
		:param path_config_file : Path to the adequate yaml file
	"""
	yml_file = open(path_config_file).read()
	m_config = yaml.load(yml_file, yaml.SafeLoader)
	m_config = {**m_config, **extra_args}
	print('################# Configuration file #################')
	print(m_config)
	print('######################################################')

	# File containing the training and testing samples
	data_set_file = m_config['train_data_file']
	mFile = open(data_set_file, 'rb')
	mSampleLog = pickle.load(mFile)
	mFile.close()

	model_name = m_config['model_name']
	out_file = m_config['output_file']

	type_baseline = m_config.get('baseline', 'rk4')

	opt_info = m_config['optimizer']

	seed_list = m_config['seed']

	batch_size = m_config['batch_size']
	patience = m_config.get('patience', -1)
	pen_l2 = m_config['pen_l2']

	pen_constr = m_config['pen_constr']

	# Set the colocation set to be the total size of the training, testing, and coloc data set
	# n_coloc = len(mSampleLog.xTrain) + len(mSampleLog.xTest) + (len(mSampleLog.xTrainExtra[1][0]) if type(mSampleLog.xTrainExtra) == tuple else 0)
	n_coloc = len(mSampleLog.xTest) + (len(mSampleLog.xTrainExtra[1][0]) if type(mSampleLog.xTrainExtra) == tuple else 0)
	pen_constr['coloc_set_size'] = n_coloc # Colocation set
	print('Numbber of colocation : {}'.format(n_coloc))

	nn_params_temp = m_config['nn_params']
	if 'side_info' in extra_args:
		nn_params_temp['type_sideinfo'] = extra_args['side_info']

	num_gradient_iterations = m_config['num_gradient_iterations']
	freq_accuracy = m_config['freq_accuracy']
	freq_save = m_config['freq_save']

	# Build the hyper params NN structure
	mParamsNN_list = [HyperParamsNN(model_name=model_name, n_state=int(mSampleLog.n_state), n_control=int(mSampleLog.n_control), actual_dt=float(mSampleLog.actual_dt), 
							 	nn_params=nn_params_temp, optimizer=opt_info, seed_init=int(seed_val), qp_base=mSampleLog.qp_base, batch_size=int(batch_size), 
							 	pen_l2=float(pen_l2), pen_constr=pen_constr, num_gradient_iterations=int(num_gradient_iterations), 
							 	freq_accuracy=freq_accuracy, freq_save=int(freq_save), patience=patience)\
						for seed_val in seed_list]

	return mSampleLog, mParamsNN_list, (type_baseline,data_set_file, out_file)


def main_fn(path_config_file, extra_args={}):
	# Read the configration file
	mSampleLog, mParamsNN_list, (type_baseline, data_set_file, out_file) = load_config_file(path_config_file, extra_args)

	# Save a copy of the Sample log without the ehavy data
	reducedSampleLog = SampleLog(xTrain=None, xTrainExtra=None, uTrain=None, xnextTrain=None, 
						lowU_train=mSampleLog.lowU_train, highU_train=mSampleLog.highU_train, 
						xTest=None, xTestExtra=None, uTest=None, xnextTest=None, 
						lowU_test=mSampleLog.lowU_test, highU_test=mSampleLog.highU_test, 
						env_name=mSampleLog.env_name, env_extra_args=mSampleLog.env_extra_args, 
						m_rng=mSampleLog.m_rng, seed_number= mSampleLog.seed_number, qp_indx=mSampleLog.qp_indx, 
						qp_base=mSampleLog.qp_base, n_state=mSampleLog.n_state, n_control=mSampleLog.n_control, 
						actual_dt=mSampleLog.actual_dt, disable_substep=mSampleLog.disable_substep, 
						control_policy = mSampleLog.control_policy, n_rollout= mSampleLog.n_rollout)

	# Print the sample log info
	print ("###### SampleLog ######")
	print (reducedSampleLog)

	# Create the brax environemnt 
	env = mSampleLog.env_name
	initialize_problem_var(xml_path+env+'.xml', eps_diff=1e-7)

	# Training data
	xTrainList = np.asarray(mSampleLog.xTrain)
	# print(xTrainList)
	xTrainExtraList = None # np.asarray(mSampleLog.xTrainExtra[0]) if type(mSampleLog.xTrainExtra) == tuple else np.asarray(mSampleLog.xTrainExtra)
	xnextTrainList = np.asarray(mSampleLog.xnextTrain)
	uTrainList = np.asarray(mSampleLog.uTrain)

	# Save the number of training trajectories
	(_, num_traj_data, trajectory_length) = mSampleLog.disable_substep

	# Save in the GPU memory the colocation set
	coloc_set, u_coloc_set, coloc_set_extra = (None, None, None) # if not type(mSampleLog.xTrainExtra) == tuple else mSampleLog.xTrainExtra[1]
	assert coloc_set_extra is None, 'coloc_set_extra should be NOne as no extra data is required in brax environments'
	coloc_set, u_coloc_set, coloc_set_extra = (np.asarray(coloc_set) if coloc_set is not None else coloc_set), (np.asarray(u_coloc_set) if u_coloc_set is not None else u_coloc_set), None

	# Testing data
	xTest = jnp.asarray(mSampleLog.xTest)
	xTestExtra = None # jnp.asarray(mSampleLog.xTestExtra)
	xTestNext = jnp.asarray(mSampleLog.xnextTest)
	uTest = jnp.asarray(mSampleLog.uTest)

	# For stopping early, the algorithm evaluate on a subset of colocation points, we take the last data from the coloc set --> this assume size coloc >= test
	assert coloc_set is None or xTest.shape[0] < coloc_set.shape[0]
	coloc_early_stopping = jnp.asarray(coloc_set[-xTest.shape[0]:, :]) if coloc_set is not None else None
	u_coloc_early_stopping = jnp.asarray(u_coloc_set[-xTest.shape[0]:, :]) if u_coloc_set is not None else None


	# DIctionary for logging training details
	final_log_dict = {i : {} for i in range(len(num_traj_data))}
	final_params_dict = {i : {} for i in range(len(num_traj_data))}

	for i, n_train in tqdm(enumerate(num_traj_data),  total=len(num_traj_data)):
		# The total number of point in this trajectory
		total_traj_size = n_train*trajectory_length

		# Dictionary to save the loss and the optimal params per radom seed 
		dict_loss_per_seed = {mParamsNN.seed_init : {} for mParamsNN in mParamsNN_list}
		dict_params_per_seed = {mParamsNN.seed_init : None for mParamsNN in mParamsNN_list}

		# Iterate for the current trajectory through the number of seed
		for _, mParamsNN in tqdm(enumerate(mParamsNN_list), total=len(mParamsNN_list), leave=False):

			# Some logging execution 
			dbg_msg = "###### Hyper parameters: {} ######\n".format(type_baseline)
			dbg_msg += "{}\n".format(mParamsNN)
			tqdm.write(dbg_msg)

			# Build the neural network, the update, loss function and paramters of the neural network structure
			rng_gen, opt, (params, m_pen_eq_k, m_pen_ineq_k, m_lagr_eq_k, m_lagr_ineq_k), pred_xnext,\
				loss_fun, update, update_lagrange, loss_fun_constr, train_with_constraints =\
					build_learner(mParamsNN, env, type_baseline)

			# Initialize the optimizer with the initial parameters
			opt_state = opt.init(params)

			# Get the frequency at which to evaluate on the testing
			high_freq_record_rg = int(mParamsNN.freq_accuracy[0]*mParamsNN.num_gradient_iterations)
			high_freq_val = mParamsNN.freq_accuracy[1]
			low_freq_val = mParamsNN.freq_accuracy[2]
			update_freq = jnp.array([ (j % high_freq_val)==0 if j <= high_freq_record_rg else ((j % low_freq_val)==0 if j < mParamsNN.num_gradient_iterations-1 else True) \
										for j in range(mParamsNN.num_gradient_iterations)])

			# Dictionary to save the loss, the rollout error, and the colocation accuracy
			dict_loss = {'total_loss_train' : list(), 'rollout_err_train' : list(), 'total_loss_test' : list(), 'rollout_err_test' : list(), 'coloc_err_train' : list(), 'coloc_err_test' : list()}

			# Store the best parameters attained by the learning process so far
			best_params = None 				# Weights returning best loss over the training set
			best_test_loss = None 			# Best test loss over the training set
			best_train_loss = None 			# Best training loss
			best_constr_val = None 			# Constraint attained by the best params
			best_test_noconstr = None 		# Best mean squared error without any constraints
			iter_since_best_param = None 	# Iteration since the last best weights values (iteration in terms of the loss evaluation period)
			number_lagrange_update = 0 		# Current number of lagrangian and penalty terms updates
			number_inner_iteration = 0 		# Current number of iteration since the latest update
			patience = mParamsNN.patience 	# Period over which no improvement yields the best solution
			tqdm.write('\t\tEarly stopping criteria = {}\n'.format(patience > 0))
			# tqdm.write('\t\tNumber of colocations = {}\n'.format(coloc_set.shape if coloc_set is not None else 0))

			# Iterate to learn the weight of the neural network
			for step in tqdm(range(mParamsNN.num_gradient_iterations), leave=False):

				# Random key generator for batch data
				rng_gen, subkey = jax.random.split(rng_gen)
				idx_train = jax.random.randint(subkey, shape=(mParamsNN.batch_size,), minval=0, maxval=total_traj_size)

				# Sample and put into the GPU memory the batch-size data
				xTrain, xTrainNext, xTrainExtra, uTrain = jnp.asarray(xTrainList[idx_train,:]), jnp.asarray(xnextTrainList[:,idx_train,:]), None, jnp.asarray(uTrainList[:,idx_train,:])

				# Sample batches of the colocation data sets -> Get random samples around train/test/ and coloc set
				if coloc_set is not None:
					# # Get the indexes in the trainign and testing set
					# rng_gen, subkey = jax.random.split(rng_gen)
					# idxtrain_coloc = jax.random.randint(subkey, shape=(mParamsNN.pen_constr['batch_size_train'],), minval=0, maxval=xTrainList.shape[0])
					rng_gen, subkey = jax.random.split(rng_gen)
					idxtest_coloc = jax.random.randint(subkey, shape=(mParamsNN.pen_constr['batch_size_test'],), minval=0, maxval=xTest.shape[0])
					rng_gen, subkey = jax.random.split(rng_gen)
					idxcoloc_coloc = jax.random.randint(subkey, shape=(mParamsNN.pen_constr['batch_size_coloc'],), minval=0, maxval=coloc_set.shape[0])
					# Get the correct set of lagragian multipliers
					# print(m_lagr_eq_k.shape)
					lag_eq_temp = m_lagr_eq_k if m_lagr_eq_k.shape[0]<=0 else jnp.vstack(( m_lagr_eq_k[idxtest_coloc], m_lagr_eq_k[xTest.shape[0]+idxcoloc_coloc]))
					lag_ineq_temp = m_lagr_ineq_k if m_lagr_ineq_k.shape[0]<=0 else jnp.vstack((m_lagr_ineq_k[idxtest_coloc], m_lagr_ineq_k[xTest.shape[0]+idxcoloc_coloc]))
					# Get the proper colocation points
					x_coloc_temp = jnp.vstack((xTest[idxtest_coloc,:], coloc_set[idxcoloc_coloc,:]))
					u_coloc_temp = jnp.vstack((uTest[0,idxtest_coloc,:], u_coloc_set[idxcoloc_coloc,:]))
					# print(x_coloc_temp.shape, u_coloc_temp.shape)
				else:
					lag_eq_temp = m_lagr_eq_k
					lag_ineq_temp = m_lagr_ineq_k
					x_coloc_temp = coloc_set
					u_coloc_temp = u_coloc_set

				# Update the parameters of the NN and the state of the optimizer
				params, opt_state = update(params, opt_state, xTrainNext, xTrain, uTrain, extra_args=xTrainExtra, 
											pen_eq_k=m_pen_eq_k, pen_ineq_sq_k=m_pen_ineq_k, 
											lagr_eq_k=  lag_eq_temp, lagr_ineq_k = lag_ineq_temp, 
											extra_args_colocation=(x_coloc_temp, u_coloc_temp, None) if coloc_set is not None \
																	else (None, None, None))

				# If it is time to evaluate the models do it
				if update_freq[number_inner_iteration]:
					if not train_with_constraints: # In case there is no constraints -> Try to also log the constraints incurred by the current neural structure
						loss_tr, (spec_data_tr, coloc_ctr) = loss_fun_constr(params, xTrainNext, xTrain, uTrain, extra_args=xTrainExtra)
						loss_te, (spec_data_te, coloc_cte) = loss_fun_constr(params, xTestNext, xTest, uTest, extra_args=xTestExtra, 
																	extra_args_colocation=(coloc_early_stopping, u_coloc_early_stopping, None))
					else:
						loss_tr, (spec_data_tr, coloc_ctr) = loss_fun(params, xTrainNext, xTrain, uTrain, extra_args=xTrainExtra)
						loss_te, (spec_data_te, coloc_cte) = loss_fun(params, xTestNext, xTest, uTest, extra_args=xTestExtra, 
																	pen_eq_k=m_pen_eq_k, pen_ineq_sq_k=m_pen_ineq_k, 
																	lagr_eq_k=m_lagr_eq_k[-xTest.shape[0]:] if m_lagr_eq_k.shape[0]>0 else m_lagr_eq_k, 
																	lagr_ineq_k = m_lagr_ineq_k[-xTest.shape[0]:] if m_lagr_ineq_k.shape[0]>0 else m_lagr_ineq_k, 
																	extra_args_colocation=(coloc_early_stopping, u_coloc_early_stopping, None))

					# Log the value obtained by evaluating the current model
					dict_loss['total_loss_train'].append(float(loss_tr))
					dict_loss['total_loss_test'].append(float(loss_te))
					dict_loss['rollout_err_train'].append(spec_data_tr)
					dict_loss['rollout_err_test'].append(spec_data_te)
					dict_loss['coloc_err_train'].append(coloc_ctr)
					dict_loss['coloc_err_test'].append(coloc_cte)

					# Initialize the parameters for the best model so far
					if number_inner_iteration == 0:
						best_params = params
						best_test_noconstr = jnp.mean(spec_data_te[:,1]) # Rollout mean
						best_test_loss = loss_te
						best_train_loss = loss_tr
						best_constr_val = coloc_cte
						iter_since_best_param = 0
						
					# Check if the validation metrics has improved
					if loss_te < best_test_loss:
						best_params = params
						best_test_loss = loss_te
						best_test_noconstr = jnp.mean(spec_data_te[:,1])
						best_constr_val = coloc_cte
						best_train_loss = loss_tr if loss_tr < best_train_loss else best_train_loss
						iter_since_best_param = 0
					else:
						best_train_loss = loss_tr if loss_tr < best_train_loss else best_train_loss
						iter_since_best_param += 1

				# Period at which we save the models
				if number_inner_iteration % mParamsNN.freq_save == 0: # Period at which we save the models

					# Update for the current seed what is the lost and the best params found so far
					dict_loss_per_seed[mParamsNN.seed_init] = dict_loss
					final_log_dict[i] = dict_loss_per_seed

					# Update the best params found so far
					dict_params_per_seed[mParamsNN.seed_init] = best_params
					final_params_dict[i] = dict_params_per_seed

					# Create the log file
					mLog = LearningLog(env_name=mSampleLog.env_name, env_extra_args=mSampleLog.env_extra_args, baseline=type_baseline,
						sampleLog=reducedSampleLog, nn_hyperparams= mParamsNN_list, loss_evol=final_log_dict, learned_weights=final_params_dict, 
						data_set_file=data_set_file)

					# Save the current information in the file and close the file
					mFile = open(out_file+'.pkl', 'wb') 
					pickle.dump(mLog, mFile)
					mFile.close()

					# Debug message to bring on the consolde
					dbg_msg = '[Iter[{}][{}], N={}]\t train : {:.9f} | Loss test : {:.9f}\n'.format(step, number_lagrange_update, n_train, loss_tr, loss_te)
					dbg_msg += '\t\tBest Loss Function : Train = {:.9f} | Test = {:.9f}, {:.9f} | Constr = {}\n'.format(best_train_loss, best_test_loss, best_test_noconstr, best_constr_val.round(9))
					dbg_msg += '\t\tPer rollout loss train : {} | Loss test : {}\n'.format(spec_data_tr[:,1].round(9), spec_data_te[:,1].round(9))
					dbg_msg += '\t\tPer rollout EQ Constraint Train : {} | Test : {}\n'.format(spec_data_tr[:,2].round(9), spec_data_te[:,2].round(9))
					dbg_msg += '\t\tPer rollout INEQ Constraint Train : {} | Test : {}\n'.format(spec_data_tr[:,3].round(9), spec_data_te[:,3].round(9))
					dbg_msg += '\t\tColocotion loss : {}\n'.format(coloc_cte.round(8))
					tqdm.write(dbg_msg)

				# Update the number of iteration the latest update of the lagrangian terms
				number_inner_iteration += 1

				# If the patience periode has been violated, try to break
				if patience > 0 and iter_since_best_param > patience:

					# Debug message
					tqdm.write('####################### EARLY STOPPING [{}][{}] #######################'.format(step, number_lagrange_update))
					tqdm.write('Best Loss Function : Train = {:.6f} | Test = {:.6f}, {:.6f}'.format(best_train_loss, best_test_loss, best_test_noconstr))

					# If there is no constraints then break
					if not train_with_constraints:
						break

					# Check if the constraints threshold has been attained
					if best_constr_val[1] < mParamsNN.pen_constr['tol_constraint_eq'] and best_constr_val[2] < mParamsNN.pen_constr['tol_constraint_ineq']:
						tqdm.write('Constraints satisfied: [eq =  {}], [ineq = {}]\n'.format(best_constr_val[1], best_constr_val[2]))
						tqdm.write('##############################################################################\n')
						break

					# If the constraints threshold hasn't been violated, update the lagrangian and penalty coefficients
					m_pen_eq_k, m_pen_ineq_k, m_lagr_eq_k, m_lagr_ineq_k = \
									update_lagrange(params, pen_eq_k=m_pen_eq_k, pen_ineq_sq_k=m_pen_ineq_k, 
									lagr_eq_k= m_lagr_eq_k, lagr_ineq_k = m_lagr_ineq_k, extra_args_colocation=(jnp.vstack((xTest, coloc_set)), jnp.vstack((uTest[0], u_coloc_set)), None))


					# Update the new params to be the best params
					params = best_params

					# UPdate the number of inner iteration since the last lagrangian terms update
					number_inner_iteration = 0

					# Update the number of lagrange update so far
					number_lagrange_update += 1

					# Reinitialize the optimizer
					opt_state = opt.init(params)

					# Some printing for debig
					tqdm.write('Update Penalty : [eq = {:.4f}, lag_eq = {:.4f}] | [ineq = {:.4f}, lag_ineq = {:.4f}]'.format(m_pen_eq_k, jnp.sum(jnp.abs(m_lagr_eq_k)) , m_pen_ineq_k, jnp.sum(m_lagr_ineq_k)))

			# Once the algorithm has converged, collect and save the data and params
			dict_loss_per_seed[mParamsNN.seed_init] = dict_loss
			final_log_dict[i] = dict_loss_per_seed
			dict_params_per_seed[mParamsNN.seed_init] = best_params
			final_params_dict[i] = dict_params_per_seed

			# Cretae the log
			mLog = LearningLog(env_name=mSampleLog.env_name, env_extra_args=mSampleLog.env_extra_args, baseline=type_baseline,
						sampleLog=reducedSampleLog, nn_hyperparams= mParamsNN_list, loss_evol=final_log_dict, learned_weights=final_params_dict, 
						data_set_file=data_set_file)
			
			# Save the current information
			mFile = open(out_file+'.pkl', 'wb') 
			pickle.dump(mLog, mFile)
			mFile.close()


if __name__ == "__main__":
	import time
	import argparse

	# Command line argument for setting parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', required=True, type=str, help='yaml configuration file for training/testing information: see reacher_cfg1.yaml for more information')
	parser.add_argument('--input_file', type=str, default='')
	parser.add_argument('--output_file', type=str, default='')
	parser.add_argument('--batch_size', type=int, default=0)
	parser.add_argument('--baseline', type=str, default='')
	parser.add_argument('--num_grad', type=int, default=0)
	parser.add_argument('--side_info', type=int, default=-1)
	args = parser.parse_args()
	m_config_aux = {'cfg' : args.cfg}
	if args.output_file != '':
		m_config_aux['output_file']  = args.output_file
	if args.batch_size > 0:
		m_config_aux['batch_size']  = args.batch_size
	if args.baseline != '':
		m_config_aux['baseline']  = args.baseline
	if args.num_grad > 0:
		m_config_aux['num_gradient_iterations']  = args.num_grad
	if args.input_file != '':
		m_config_aux['train_data_file']  = args.input_file
	if args.side_info >= 0:
		m_config_aux['side_info']  = args.side_info

	main_fn(args.cfg, m_config_aux)