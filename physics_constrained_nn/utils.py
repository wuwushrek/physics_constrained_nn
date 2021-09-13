from haiku._src.data_structures import FlatMap
import pickle
from pathlib import Path
from typing import Union, Optional, Tuple
from collections import namedtuple

# Haiku for Neural networks
import haiku as hk
import jax
import jax.numpy as jnp

# Optax for the optimization scheme
import optax

suffix = '.pickle'

_INITIALIZER_MAPPING = \
{
	'Constant' : hk.initializers.Constant,
	'RandomNormal' : hk.initializers.RandomNormal,
	'RandomUniform' : hk.initializers.RandomUniform,
	'TruncatedNormal' : hk.initializers.TruncatedNormal,
	'VarianceScaling' : hk.initializers.VarianceScaling,
	'UniformScaling' : hk.initializers.UniformScaling
}

_ACTIVATION_FN = \
{
	'relu' : jax.nn.relu,
	'sigmoid' : jax.nn.sigmoid,
	'softplus' : jax.nn.softplus,
	'hard_tanh' : jax.nn.hard_tanh,
	'selu' : jax.nn.selu
}

# _OPTIMIZER_FN = \
# {
# 	'adam' : optax.adam,
# 	'adabelief' : optax.adabelief,
# 	'adagrad' : optax.adagrad,
# 	'fromage' : optax.fromage,
# 	'noisy_sgd' : optax.noisy_sgd,
# 	'sgd' : optax.sgd,
# 	'adamw' : optax.adamw,
# 	'lamb' : optax.lamb,
# 	'yogi' : optax.yogi,
# 	'rmsprop' : optax.rmsprop
# }
_OPTIMIZER_FN = \
{
	'adam' : optax.scale_by_adam,
	'adabelief' : optax.scale_by_belief,
	# 'adagrad' : optax.adagrad,
	# 'fromage' : optax.fromage,
	# 'noisy_sgd' : optax.noisy_sgd,
	# 'sgd' : optax.sgd,
	# 'adamw' : optax.adamw,
	# 'lamb' : optax.lamb,
	'yogi' : optax.scale_by_yogi,
	# 'rmsprop' : optax.scale_by_
}

SampleLog = namedtuple("SampleLog", 
						"xTrain xTrainExtra uTrain xnextTrain lowU_train highU_train "+\
						"xTest xTestExtra uTest xnextTest lowU_test highU_test "+\
						"env_name env_extra_args m_rng seed_number "+\
						"qp_indx qp_base n_state n_control actual_dt "+\
						"disable_substep control_policy n_rollout")

# Save information related to the training and hyper parameters of the NN
HyperParamsNN = namedtuple("HyperParamsNN", "model_name n_state n_control actual_dt nn_params optimizer "+\
							"seed_init qp_base batch_size pen_l2 pen_constr num_gradient_iterations freq_accuracy freq_save patience")

# Save information rekated to the learninf steps
LearningLog = namedtuple("LearningLog", "env_name env_extra_args baseline sampleLog nn_hyperparams loss_evol learned_weights data_set_file")


def save(data: FlatMap, path: Union[str, Path], overwrite: bool = False):
	""" Pickle a FlapMap into a file
	"""
	path = Path(path)
	if path.suffix != suffix:
		path = path.with_suffix(suffix)
	path.parent.mkdir(parents=True, exist_ok=True)
	if path.exists():
		if overwrite:
			path.unlink()
		else:
			raise RuntimeError(f'File {path} already exists.')
	with open(path, 'wb') as file:
		pickle.dump(data, file)


def load(path: Union[str, Path]) -> FlatMap:
	""" Load a FlapMap from a given file directory
	"""
	path = Path(path)
	if not path.is_file():
		raise ValueError(f'Not a file: {path}')
	if path.suffix != suffix:
		raise ValueError(f'Not a {suffix} file: {path}')
	with open(path, 'rb') as file:
		data = pickle.load(file)
	return data

def build_learner_without_sideinfo(rng_key, optim, model_name, nstate, ncontrol, nn_params, 
		rollout_length=1, pen_l2=1e-4, batch_size=1, rk4=False, actual_dt=1e-2):
	""" This function builds a neural network to estimate future state values without any side information. 
		Specifically, it returns a function to estimate next state and a function to update the network parameters.
		:param rng_key                  : A key for random initialization of the parameters of the neural networs
		:param optim                    : The optimizer for the update of the neural networks parameters
		:param model_name               : A name for the model of interest. This must be unique as it is useful to load and save parameters of the model.
		:param nstate                   : The number of state of the system
		:param ncontrol                 : The number of control inputs of the system
		:params nn_params               : Dictionary containing the parameters of the NN estimating next state
											nn_params = {output_sizes' : , 'w_init' : , 'b_init' : , 'with_bias' : , 'activation' :, 'activate_final' :}.
											The keys of this dictionary should matches the arguments of the function 'known_dynamics' below.
		:param rollout_length 			: The rollout length of the learning process
		:param pen_l2                   : The penalty coefficient applied to the l2 norm regularizer
		:param batch_size 				: The batch size for initial compilation of the haiku pure function
		:param rk4 						: Specify the ODESolver used for the baseline
		:param actual_dt 				: The step size used for the integration scheme
	"""

	# First define a function to estimate the future state
	def pred_next_state(x, u=None):
		""" This function estimates the next state given the current state and current control input. It also returns
			the estimation of the vector fied, the unknown terms, the remainder term, the equality and inequality constraints at x and u.
			The inputs x and u must be two dimensional array as batches.
			:param x                : The current state of the system
			:param u                : The current control of the system
		"""
		assert u is None or x.shape[0] == u.shape[0], 'The (batch size of u) should be equal to (batch size of x)'
		assert 'output_sizes' in nn_params, 'Size of the hidden+output layers should be specified with the key <output_sizes>'
		dictParams_cpy = {key : val for key, val in nn_params.items()}
		input_size = nstate + ncontrol
		default_params = {'name' : model_name, 'w_init' : hk.initializers.TruncatedNormal(1. / jnp.sqrt(input_size)), 
								'b_init' : jnp.zeros, 'activation' : jax.nn.relu, 'activate_final': False}
		objNN = hk.nets.MLP(**({**default_params,**dictParams_cpy}))
		fus_xu = jnp.hstack((x,u)) if u is not None else x
		if not rk4:
			return objNN(fus_xu)
		else:
			k1 = objNN(fus_xu)
			k2 = objNN(jnp.hstack((x + 0.5 * actual_dt * k1 , u)) if u is not None else x + 0.5 * actual_dt * k1)
			k3 = objNN(jnp.hstack((x + 0.5 * actual_dt * k2 , u)) if u is not None else x + 0.5 * actual_dt * k2)
			k4 = objNN(jnp.hstack((x + actual_dt * k3 , u)) if u is not None else x + actual_dt * k3)
			return x + (actual_dt/6.0)*(k1 + k2 + k3 + k4)

	# Random x and u initialization to build network parameters 
	# -> For more general batching setting probably should not define this as a 2D array
	dummy_x_init = jax.random.uniform(rng_key, (batch_size,nstate))
	dummy_u_init = None if ncontrol == 0 else jax.random.uniform(rng_key, (batch_size,ncontrol))
	# Build the prediction function
	pred_fn_pure = hk.without_apply_rng(hk.transform(pred_next_state))
	# Initialize the parameters for the prediction function
	params_init = pred_fn_pure.init(rng_key, x=dummy_x_init, u=dummy_u_init)
	# Define the function to compute the next state
	pred_xnext = pred_fn_pure.apply

	# Then define a function to compute the loss function needed to train the model
	# This is where the roolout should be done
	def loss_fun(params : hk.Params, xnext : jnp.ndarray, x : jnp.ndarray, u : Optional[jnp.ndarray] = None):
		""" Compute the loss function given the current parameters of the custom neural network
			:param params       : Weights of all the neural networks
			:param xnext      	: The ROLLED target next state used in the mean squared product -> The first dimension should be of size rollout_length
			:param x            : The state for which to estimate the next state value -> One dimension less than xnext
			:param u            : The control signal applied at each state x rollout index -> The first dimension should be of size rollout_length
		"""
		# assert u is None or x.shape[0]==u.shape[0], 'The (batch size of u) should be equal to (batch size of x) '
		assert u is None or (len(x.shape) == len(u.shape) and x.shape[0]==u.shape[0]) or x.shape[0] == u.shape[1] , 'The (batch size of u) should be equal to (batch size of x) '
		assert (u is None or xnext.shape[0] == u.shape[0]) and xnext.shape[1] == x.shape[0]
		if pen_l2 > 0:
			l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
		else:
			l2_loss = 0
		# Scan function for roolout
		def rollout(carry, extra):
			curr_x, params = carry
			print(extra)
			curr_u, true_nextx = extra
			next_x = pred_xnext(params, curr_x, curr_u)
			return (next_x, params), jnp.sum(jnp.square(next_x - true_nextx)) / curr_x.shape[0]
		_, loss_val = jax.lax.scan(rollout, (x, params), (u, xnext))
		loss_sum = jnp.sum(loss_val) / xnext.shape[0]
		return  loss_sum + pen_l2 * l2_loss, ((loss_sum, loss_val[0]), l2_loss)
		# next_state_est = pred_xnext(params, x, u)
		# actualLoss = jnp.sum(jnp.square(next_state_est - xnext)) / x.shape[0]
		# return actualLoss + pen_l2 * l2_loss, (actualLoss, l2_loss)

	# Define the update step
	def update(params: hk.Params, opt_state: optax.OptState, xnext : jnp.ndarray, x: jnp.ndarray, u : Optional[jnp.ndarray] = None, n_iter: int = 1) -> Tuple[hk.Params, optax.OptState]:
		"""Update the parameters of the neural netowkrs via one of the gradient descent optimizer
			:param params       : The current weight of the neural network
			:param opt_state    : The current state of the optimizer
			:param xnext      	: The target next state used in the mean squared product
			:param x            : The state for which to estimate the next state value
			:param u            : The control signal applied at each state x
			:param n_iter		: The number of inner update of the weight of the underlying NN
		"""
		grad_fun = jax.grad(loss_fun, has_aux=True)
		def loop_fun_val(p_loop, extra):
			new_params, new_opt_state = p_loop
			grads, _ = grad_fun(new_params, xnext, x, u)
			updates, new_opt_state = optim.update(grads, new_opt_state, new_params)
			new_params = optax.apply_updates(new_params, updates)
			return (new_params, new_opt_state), None
		return jax.lax.scan(loop_fun_val, (params, opt_state), None, length=n_iter)[0]
	return params_init, pred_xnext, loss_fun, update