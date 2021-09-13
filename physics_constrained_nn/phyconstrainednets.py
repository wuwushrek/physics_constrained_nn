# Jax imports
import jax
import jax.numpy as jnp

# Haiku for Neural networks
import haiku as hk

# Optax for the optimization scheme
import optax

# Typing functions
from typing import Optional, Tuple

# Physics-based neural networks
class PhyConstrainedNets(hk.Module):
	""" Encode an unknown function representing the unknown Lipschitz-continuous dynamics 
		of a system. This is done via deep neural networks estimating unknown terms of the dynamics.
		In addition to using prior knowledge of known terms of the dynamics, the neural nets can 
		also enforce constraints on the unknown terms assumed to be extra side information 
		on the dynamics of the system.
	"""
	def __init__(self, ns, nu, nn_params, known_dynamics=None, constraints_dynamics=None, 
						time_step=0.1, ODESolver='rk4', name='struct_nn', data_stats=None):
		""" This initialization assumes the state is fully observable
			:params ns 						: The number of states of the system
			:params nu 						: The number of inputs of the system
			:params nn_params 				: Dictionary containing the parameters of the NN of each unknown terms
												nn_params = {'unknown_f_name' : {'input_index' : , 'output_sizes' : , 'w_init' : , 'b_init' : , 'with_bias' : , 'activation' :, 'activate_final' :}, ...}
			:params known_dynamics			: \dot{x} = f(x, u, g_1(x,u), g_2(x,u), ...) where f is the known part but the functions g_1(.), g_2(.) of the susbset of variables 
												and controls are potentially unknown. 'input_index' specifies the input dependency when merging the arrays x and u.
												THIS FUNCTION MUST BE VECTORIZED TO HANDLE BATCH INPUTS.
			:params constraints_dynamics 	: Known constraints on the dynamics that will be encoded as soft constraints on the cost function. THIS FUNCTION MUST BE VECTORIZED TO HANDLE BATCH INPUTS.
			:params time_step 				: The intergration time step used by the dynamical system
			:params ODESolver 				: This is either a string specifying the ODESolver to use ('rk4' or 'base')
			:params name 					: Name of the instance of the module
			:params data_stats 				: A tuple (mean, std) representing the (moving average) current mean and std of the inputs of the neural networks
		"""
		super().__init__(name=name)
		# If no known dynamics is given, then nn_params should contains only a single unknown term
		assert not (known_dynamics is None and len(nn_params) != 1), ' If no known dynamics is given, then nn_params should contains only a single unknown term'
		assert ODESolver == 'rk4' or ODESolver == 'base' or type(ODESolver) == dict, 'ODESolver must be either <rk4>, <base>, or a dictionary encoding the parameter of the apriori enclosure term'
		self._ns = ns
		self._nu = nu
		self._ODESolver = ODESolver
		self._time_step = time_step
		self._time_step_sqr = time_step**2 / 2.0
		self._known_dynamics = known_dynamics
		self._constraints_dynamics = constraints_dynamics
		self._data_stats = data_stats
		# Build the different neural networks and the integration neural network
		self._unknown_terms = dict()
		self.build_unknown_nets(nn_params, ODESolver)

	def build_unknown_nets(self, nn_params, ODESolver):
		""" Define the neural network for each unknown variables and for the a priori enclosure term
			and save it as attributes of this class
			:params nn_params : Define a dictionary with the parameters of each neural networks
			:params ODESolver : it defines the type of integrator or the parameters of the a priori enclosure layer
		"""
		# Build the unknown function neural networks
		for var_name, dictParams in nn_params.items():
			# The size of the neural network should be specified
			assert 'output_sizes' in dictParams and 'input_index' in dictParams, 'Size of the output or input layers should be specified with the key <output_sizes>, <input_size>'
			dictParams_cpy = {key : val for key, val in dictParams.items() if key != 'input_index'}
			input_size = dictParams['input_index'].shape[0]
			default_params = {'name' : var_name, 'w_init' : hk.initializers.TruncatedNormal(1. / jnp.sqrt(input_size)), 
								'b_init' : jnp.zeros, 'activation' : jax.nn.relu, 'activate_final': False}
			self._unknown_terms[var_name] = (hk.nets.MLP(**({**default_params,**dictParams_cpy})), dictParams['input_index'])
			# setattr(self, '_'+var_name, hk.nets.MLP(**({**default_params,**dictParams_cpy})))

		# Build the a priori enclosure estimation neural network
		if not (ODESolver == 'rk4' or ODESolver == 'base'):
			outSize = (*ODESolver['output_sizes'], self._ns)
			default_params = {'name' : 'apriori', 'w_init' : hk.initializers.TruncatedNormal(1. / (self._ns+self._nu)), 
								'b_init' : jnp.zeros, 'activation' : jax.nn.relu, 'activate_final': False}
			integrate_cpy = {key : (val if key != 'output_sizes' else outSize) for key, val in ODESolver.items()}
			self._apriori = hk.nets.MLP(**({**default_params,**integrate_cpy}))
		else:
			self._apriori = None


	# @abstractmethod
	def vector_field_and_unknown(self, x, u=None, extra_args=None, dropout_rate=None, rng=None):
		""" This function defines the underlying dynamics of the system and encodes the known terms. 
			Specifically, we assume that \dot{x} = f(x, u, g_1(x,u), g_2(x,u), ...) where f is known but
			the functions g_1(.), g_2(.) of the susbset of variables and controls are potentially
			unknown. We encode these unknown functions as a set of small size feed forward neural networks 
			for which we develop a custom forward and back propagation to train their parameters.

			The function returns both the resulting vector field and the estimate of the unknown terms
			THIS FUNCTION ASSUMES THAT THE INPUTS ARE BATCHES, I.E., TWO DIMENSIONAL ARRAYS

			:params x : The current state vector
			:params u : The current control value
			:param extra_args	: State dependent extra parameters used in side information and constraints
			:params dropout_rate : The dropout rate applied to the neural network to estimate k(x_t,u_t)
			:params rng : A key for randomness when dropout is applied
		"""
		# Fuse the vector x and vector u into one vector (assume to be row vectors)
		fus_xu = jnp.hstack((x,u)) if u is not None else x
		# Standardize the data inputs if required by the user
		if self._data_stats is not None:
			mean_val, std_val = self._data_stats
			fus_xu = (fus_xu - mean_val) / std_val

		# Need to handle the case where the inputs are one dimensional compared to batch inputs
		dictEval = { var_name : nn_val(fus_xu[:,xu_index], dropout_rate, rng)  \
						for (var_name, (nn_val, xu_index)) in self._unknown_terms.items()}
		if self._known_dynamics is None:
			(fName, fValue), = dictEval.items()
			return fValue, dictEval
		if extra_args is not None:
			return self._known_dynamics(x, u, extra_args=extra_args, **dictEval), dictEval
		else:
			return self._known_dynamics(x, u, **dictEval), dictEval

	def vector_field(self, x, u=None, extra_args=None, dropout_rate=None, rng=None):
		""" This function defines the underlying dynamics of the system and encodes the known terms. 
			Specifically, we assume that \dot{x} = f(x, u, g_1(x,u), g_2(x,u), ...) where f is known but
			the functions g_1(.), g_2(.) of the susbset of variables and controls are potentially
			unknown. We encode these unknown functions as a set of small size feed forward neural networks 
			for which we develop a custom forward and back propagation to train their parameters.

			This function only returns the estimate value of the vector field.
			THIS FUNCTION ASSUMES THAT THE INPUTS ARE BATCHES, I.E., TWO DIMENSIONAL ARRAYS

			:params x : The current state vector
			:params u : The current control value
			:param extra_args	: State dependent extra parameters used in side information and constraints
			:params dropout_rate : The dropout rate applied to the neural network to estimate k(x_t,u_t)
			:params rng : A key for randomness when dropout is applied
		"""
		vfield, _ = self.vector_field_and_unknown(x, u, extra_args, dropout_rate, rng)
		return vfield

	# @abstractmethod
	def constraints(self, x, u=None, extra_args=None):
		""" This function expresses known constraints on the unknown vector field as soft constraints
			in the loss function.
			For this function x might denotes the next state (output of our estimation) or the set of
			points for which a constraint on the part of the vector field must be satisfied.
			THIS FUNCTION ASSUMES THAT THE INPUTS ARE BATCHES, I.E., TWO DIMENSIONAL ARRAYS

			:params x : The current state of the system
			:params u : The current control value of the system
			:param extra_args	: State dependent extra parameters used in side information and constraints
		"""
		if x is None or self._constraints_dynamics is None:
			return jnp.array([]), jnp.array([]) 
		if extra_args is not None:
			return self._constraints_dynamics(x,u,extra_args=extra_args,**self._unknown_terms)
		else:
			return self._constraints_dynamics(x,u,**self._unknown_terms)


	def __call__(self, x, u=None, extra_args=None, dropout_rate=None, rng=None):
		""" This function implements the formula that extracts the next state of the system given 
			its current state and the currently-applied control signal
		"""
		# Evaluate the vector field at the current state and input
		xField, nnEvals = self.vector_field_and_unknown(x, u, extra_args, dropout_rate, rng)

		# In case the integrator is a more complicated scheme as RK4
		if self._ODESolver == 'rk4':
			k1 = xField
			k2, _ = self.vector_field_and_unknown(x+ 0.5 * self._time_step * k1, u, extra_args, dropout_rate, rng)
			k3, _ = self.vector_field_and_unknown(x+ 0.5 * self._time_step * k2, u, extra_args, dropout_rate, rng)
			k4, _ = self.vector_field_and_unknown(x+ 1.0 * self._time_step * k3, u, extra_args, dropout_rate, rng)
			return x + (self._time_step/6.0)*(k1 + k2 + k3 + k4), xField, nnEvals, jnp.zeros_like(xField)

		# In case the integrator is the simpler 1 step euler approach
		if self._ODESolver == 'base':
			return x + xField *self._time_step, xField, nnEvals, jnp.zeros_like(xField)

		if self._ODESolver == 'base_base':
			return xField, xField, nnEvals, jnp.zeros_like(xField)

		raise ('Not implemented')


def build_learner_with_sideinfo(rng_key, optim, model_name, time_step, nstate, ncontrol, nn_params, ODESolver, 
								known_dynamics=None, constraints_dynamics=None, pen_l2=1e-4, pen_constr={}, 
								batch_size=1, extra_args_init=None, train_with_constraints = False, normalize=True):
	""" This function builds a neural network to estimate future state values while encoding side information about 
		the dynamics. Specifically, it returns a function to estimate next state and a function to update the network parameters.
		:param rng_key 					: A key for random initialization of the parameters of the neural networs
		:param optim 					: The optimizer for the update of the neural networks parameters
		:param model_name 				: A name for the model of interest. This must be unique as it is useful to load and save parameters of the model.
		:param time_step 				: The integrate time step of the system
		:param nstate 					: The number of state of the system
		:param ncontrol 				: The number of control inputs of the system
		:params nn_params 				: Dictionary containing the parameters of the NN of each unknown terms
											nn_params = {'g_1' : {'input_index' : , 'output_sizes' : , 'w_init' : , 'b_init' : , 'with_bias' : , 'activation' :, 'activate_final' :}, ...}.
											The keys of this dictionary should matches the arguments of the function 'known_dynamics' below.
		:params ODESolver 				: This is either a string specifying the ODESolver to use ('rk4' or 'base')
		:params known_dynamics			: \dot{x} = known_dynamics(x, u, g_1(x,u), g_2(x,u), ...) where 'known_dynamics' is the known part but the functions g_1(.), g_2(.) of the susbset of variables 
											and controls are potentially unknown. 'input_index' specifies the input dependency when merging the arrays x and u.
											The extra arguments name g_1, g_2 pf this function should match the keys of the dictionary of 'nn_params' above.
											THIS FUNCTION MUST BE VECTORIZED TO HANDLE BATCH INPUTS.
		:params constraints_dynamics 	: Known constraints on the dynamics that will be encoded as soft constraints on the cost function.
											This function takes as argument x, u and the functions g_1, .., g_d and returns two jnp.ndarray (first for equality and second for ineq)
											THIS FUNCTION MUST BE VECTORIZED TO HANDLE BATCH INPUTS.
		:param pen_l2 					: The penalty coefficient applied to the l2 norm regularizer
		:param pen_constr				: The penalty coefficient to take into account the known constraints on the dynamics
		:param batch_size 				: The batch size for initial compilation of the haiku pure function
		:param extra_args_init 			: The size of the last dimension of the extra argument to use in known_dynamics
		:param train_with_constraints 	: Specify if the learning should include the constraints or only use the constraints for loss function metric evaluation
	"""
	
	# First define a function to estimate the future state
	def pred_next_state_full(x, u=None, extra_args=None, extra_args_colocation=(None, None, None), constr_fun=None):
		""" This function estimates the next state given the current state and current control input. It also returns
			the estimation of the vector fied, the unknown terms, the remainder term, the equality and inequality constraints at x and u.
			The inputs x and u must be two dimensional array as batches.
			:param x 				: The current state of the system
			:param u 				: The current control of the system
			:param extra_args		: State dependent extra parameters used in side information and constraints		
			:param extra_args_colocation : 
			:paran constr_fun 		: DEnote the function that computes the constraints
		"""
		assert u is None or x.shape[0] == u.shape[0], 'The (batch size of u) should be equal to (batch size of x)'
		objNN = PhyConstrainedNets(nstate, ncontrol, nn_params, known_dynamics, constr_fun, time_step, ODESolver, model_name)
		# Compute the constraints associated to the next prediction
		eqCterms, ineqCterms = objNN.constraints(x, u, extra_args)
		# Compute the constraints cost associated to the colocoation method 
		x_coloc, u_coloc, xextra_coloc = extra_args_colocation
		eqCterms_coloc, ineqCterms_coloc = objNN.constraints(x_coloc, u_coloc, xextra_coloc) if x_coloc is not None else (jnp.array([]), jnp.array([]))
		# Evaluate the ODESolver for obtaining the next state
		nextX, vectorfieldX, unkTermsAtX, remTermAtX =  objNN(x, u, extra_args) if x is not None else (jnp.array([]), jnp.array([]), jnp.array([]), jnp.array([]))
		return nextX, vectorfieldX, unkTermsAtX, remTermAtX, (eqCterms, eqCterms_coloc), (ineqCterms, ineqCterms_coloc)

	# Predict the next state with disabling constraints if requested
	pred_next_state = lambda x, u=None, extra_args=None, extra_args_colocation=(None, None, None): pred_next_state_full(x, u, extra_args, extra_args_colocation, None if not train_with_constraints else constraints_dynamics)

	# Predict the next state with constraints enabled
	pred_next_state_constr = lambda x, u=None, extra_args=None, extra_args_colocation=(None, None, None): pred_next_state_full(x, u, extra_args, extra_args_colocation, constraints_dynamics)

	# Random x and u initialization to build network parameters
	dummy_x_init = jax.numpy.zeros((batch_size,nstate))
	dummy_u_init = None if ncontrol == 0 else jax.numpy.zeros((batch_size,ncontrol))
	dummy_args = None if extra_args_init is None else jax.numpy.zeros((batch_size, extra_args_init))

	# Build the prediction function
	pred_fn_pure = hk.without_apply_rng(hk.transform(pred_next_state))
	pred_next_state_constr_pure = hk.without_apply_rng(hk.transform(pred_next_state_constr))

	# Initialize the parameters for the prediction function -> Can use the defualt extra_args_colocation here
	params_init = pred_fn_pure.init(rng_key, x=dummy_x_init, u=dummy_u_init, extra_args=dummy_args)
	pred_next_state_constr_pure.init(rng_key, x=dummy_x_init, u=dummy_u_init, extra_args=dummy_args)

	# Penalization init scheme
	if type(pen_constr) == dict:
		pen_eq_init, beta_eq, pen_ineq_init, beta_ineq = pen_constr['pen_eq_init'], pen_constr['beta_eq'], pen_constr['pen_ineq_init'], pen_constr['beta_ineq']
		pen_eq_shape, pen_ineq_shape = pen_constr['num_eq_constr'], pen_constr['num_ineq_constr']
		total_constr = pen_constr['coloc_set_size']
	else:
		pen_eq_init, beta_eq, pen_ineq_init, beta_ineq, pen_eq_shape, pen_ineq_shape, total_constr = 0,0,0,0,0,0,0
	m_pen_eq_k  = pen_eq_init
	m_pen_ineq_k = pen_ineq_init
	m_lagr_eq_k = jnp.array([]) if pen_eq_shape <= 0 else jnp.zeros((total_constr, pen_eq_shape))
	m_lagr_ineq_k = jnp.array([]) if pen_ineq_shape <= 0 else jnp.zeros((total_constr, pen_ineq_shape))

	# Define the function to compute the next state
	pred_xnext = pred_fn_pure.apply
	pred_xnext_constr = pred_next_state_constr_pure.apply

	# Then define a function to compute the loss function needed to train the model
	def loss_fun(params : hk.Params, xnext : jnp.ndarray, x : jnp.ndarray, u : Optional[jnp.ndarray] = None, 
					extra_args : Optional[jnp.ndarray] = None, 
					pen_eq_k : Optional[float] = 0, pen_ineq_sq_k: Optional[float] = 0.0, 
					lagr_eq_k : Optional[jnp.ndarray] = 0.0,
					lagr_ineq_k : Optional[jnp.ndarray] = 0.0,
					extra_args_colocation : Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] =(None, None, None)):
		""" Compute the loss function given the current parameters of the custom neural network
			:param params 		: Weights of all the neural networks
			:param xnext 		: The target netx state used in the mean squared product
			:param x 			: The state for which to estimate the next state value
			:param u 			: The control signal applied at each state x
			:param extra_args	: State dependent extra parameters used in side information and constraints
			:param pen_eq_k 	: Penalty coefficient for the equality constraints 
			:param pen_ineq_sq_k: Penalty coefficient for the inequality constraints Phi(x,u,...) <= 0
			:param lagr_eq_k 	: Lagrangier multiplier for the equality constraints
			:param lagr_ineq_k 	: Lagranfier multiplier for the inequality constraints
			:param extra_args_colocation : Point used in order to enforce the constraints on unlabbelled data
		"""
		# assert u is None or x.shape[0]==u.shape[0], 'The (batch size of u) should be equal to (batch size of x)'
		assert u is None or (u.shape[0] == xnext.shape[0] and len(x.shape) == len(xnext.shape)-1), \
			'Mismatch ! xnext and u should have one more dimension than x of size roolout'

		# Don't compute the L2 norm if the penalization term is zero
		l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)) if pen_l2 > 0.0 else 0.0

		# Scan function for rolling out the dynamics
		def rollout(carry, extra):
			""" Rollout the computation of the next state
			"""
			curr_x, params = carry
			curr_u, true_nextx = extra

			# Predict the next state
			next_x, _, _, _, (eqCterms,_), (ineqCterms,_) = pred_xnext(params, curr_x, curr_u, extra_args)

			# COmpute the cost associated to the constraints
			cTerm_eq = 0.0 if eqCterms.shape[0] <= 0 else jnp.sum(jnp.square(eqCterms))/ eqCterms.size
			cTerm_ineq = 0.0 if ineqCterms.shape[0] <= 0 else (jnp.sum(jnp.where(ineqCterms > 0, 1.0, 0.0) * jnp.square(ineqCterms)) /  ineqCterms.size)

			# Measure the mean squared difference
			meanSquredDiff = jnp.sum(jnp.square(next_x - true_nextx)) / (x.shape[0]*(1 if not normalize else x.shape[-1]))

			# Compute the total loss
			totalLoss =  meanSquredDiff + pen_eq_k * cTerm_eq + pen_ineq_sq_k * cTerm_ineq
			return (next_x, params), jnp.array([totalLoss, meanSquredDiff, cTerm_eq, cTerm_ineq])

		# Rollout and compute the rolled out error term
		_, m_res = jax.lax.scan(rollout, (x, params), (u, xnext))

		# Compute the loss associated to the collocation points 
		_, _, _, _, (_, eqCterms), (_, ineqCterms) = pred_xnext(params, None, None, None, extra_args_colocation=extra_args_colocation)

		# Check the satisfaction  of equality constraints
		cTerm_eq = 0.0 if eqCterms.shape[0] <= 0 else jnp.sum(jnp.square(eqCterms))/ eqCterms.size

		# Check the satisfaction of inequality constraints
		cTerm_ineq = 0.0 if ineqCterms.shape[0] <= 0 else (jnp.sum(jnp.where(ineqCterms > 0, 1.0, 0.0) * jnp.square(ineqCterms)) /  ineqCterms.size)

		# Compute the total constraint satisfaction
		coloc_cost = pen_eq_k * cTerm_eq + pen_ineq_sq_k * cTerm_ineq + (0.0 if eqCterms.shape[0] <= 0 else (jnp.sum(lagr_eq_k * eqCterms)/eqCterms.size)) + \
						(0.0 if ineqCterms.shape[0] <=0 else (jnp.sum(lagr_ineq_k * ineqCterms) / ineqCterms.size))

		# Return the composite
		return  (jnp.sum(m_res[:,0]) / (m_res.shape[0])) + pen_l2 * l2_loss + coloc_cost, (m_res, jnp.array([coloc_cost, cTerm_eq, cTerm_ineq]))

	# Util function solely for computing colocation loss ehn no constraints are given 
	def loss_fun_constr(params, xnext : jnp.ndarray, x : jnp.ndarray, u : Optional[jnp.ndarray] = None, extra_args : Optional[jnp.ndarray] = None,
							extra_args_colocation : Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] =(None, None, None)):
		""" Compute the loss function given the current parameters of the custom neural network
			:param params 		: Weights of all the neural networks
			:param xnext 		: The target netx state used in the mean squared product
			:param x 			: The state for which to estimate the next state value
			:param u 			: The control signal applied at each state x
			:param extra_args	: State dependent extra parameters used in side information and constraints
			:param extra_args_colocation : Point used in order to enforce the constraints on unlabbelled data
		"""
		assert u is None or (u.shape[0] == xnext.shape[0] and len(x.shape) == len(xnext.shape)-1), \
			'Mismatch ! xnext and u should have one more dimension than x of size roolout'

		# L2 loss if provived by the user
		l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params)) if pen_l2 > 0.0 else 0.0

		# Rollout strategy
		def rollout(carry, extra):
			""" Rollout the computation of the next state
			"""
			curr_x, params = carry
			curr_u, true_nextx = extra
			# Predict the next state
			next_x, _, _, _, (eqCterms,_), (ineqCterms,_) = pred_xnext_constr(params, curr_x, curr_u, extra_args)
			# COmpute the cost associated to the constraints
			cTerm_eq = 0.0 if eqCterms.shape[0] <= 0 else jnp.sum(jnp.square(eqCterms))/ eqCterms.size
			cTerm_ineq = 0.0 if ineqCterms.shape[0] <= 0 else (jnp.sum(jnp.where(ineqCterms > 0, 1.0, 0.0) * jnp.square(ineqCterms)) /  ineqCterms.size)
			# Measure the mean squared difference
			meanSquredDiff = jnp.sum(jnp.square(next_x - true_nextx)) / (x.shape[0]*(1 if not normalize else x.shape[-1]))
			return (next_x, params), jnp.array([meanSquredDiff, meanSquredDiff, cTerm_eq, cTerm_ineq])

		# Rollout and compute the rolled out error term
		_, m_res = jax.lax.scan(rollout, (x, params), (u, xnext))

		# Compute the loss associated to the collocation points 
		_, _, _, _, (_, eqCterms), (_, ineqCterms) = pred_xnext_constr(params, None, None, None, extra_args_colocation=extra_args_colocation)

		# Check the satisfaction  of equality constraints
		cTerm_eq = 0.0 if eqCterms.shape[0] <= 0 else jnp.sum(jnp.square(eqCterms))/ eqCterms.size

		# Check the satisfaction  of ineequality constraints
		cTerm_ineq = 0.0 if ineqCterms.shape[0] <= 0 else (jnp.sum(jnp.where(ineqCterms > 0, 1.0, 0.0) * jnp.square(ineqCterms)) /  ineqCterms.size)

		return (jnp.sum(m_res[:,0]) / (m_res.shape[0])) + pen_l2 * l2_loss, (m_res, jnp.array([1e-15, cTerm_eq, cTerm_ineq]))

	# Define the update step
	def update(params: hk.Params, opt_state: optax.OptState, xnext : jnp.ndarray, x: jnp.ndarray, u : Optional[jnp.ndarray] = None, 
				extra_args : Optional[jnp.ndarray] = None, 
				pen_eq_k : Optional[float] = 0, pen_ineq_sq_k: Optional[float] = 0.0, 
				lagr_eq_k : Optional[jnp.ndarray] = jnp.array([]),
				lagr_ineq_k : Optional[jnp.ndarray] = jnp.array([]),
				extra_args_colocation : Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] =(None, None, None),
				n_iter : int = 1) -> Tuple[hk.Params, optax.OptState]:
		"""Update the parameters of the neural netowkrs via one of the gradient descent optimizer
			:param params 		: The current weight of the neural network
			:param opt_state 	: The current state of the optimizer
			:param x 			: The state for which to estimate the next state value
			:param u 			: The control signal applied at each state x
			:param extra_args	: State dependent extra parameters used in side information and constraints
			:param pen_eq_k 	: Penalty coefficient for the equality constraints 
			:param pen_ineq_sq_k: Penalty coefficient for the inequality constraints Phi(x,u,...) <= 0
			:param lagr_eq_k 	: Lagrangier multiplier for the equality constraints
			:param lagr_ineq_k 	: Lagranfier multiplier for the inequality constraints
			:param extra_args_colocation : Point used in order to enforce the constraints on unlabbelled data
			:param n_iter 		: Number of iteration for each update
			... 				: similar aruguments related to colocationa as the loss function
		"""
		grad_fun = jax.grad(loss_fun, has_aux=True)
		def loop_fun_val(p_loop, extra):
			new_params, new_opt_state = p_loop
			grads, _ = grad_fun(new_params, xnext, x, u, extra_args, pen_eq_k, pen_ineq_sq_k, lagr_eq_k, lagr_ineq_k, extra_args_colocation)
			updates, new_opt_state = optim.update(grads, new_opt_state, new_params)
			new_params = optax.apply_updates(new_params, updates)
			return (new_params, new_opt_state), None
		return jax.lax.scan(loop_fun_val, (params, opt_state), None, length=n_iter)[0]

	# Define the update step for the lagrangier multipliers term
	def update_lagrange(params, pen_eq_k : Optional[float] = 0, pen_ineq_sq_k: Optional[float] = 0.0, 
				lagr_eq_k : Optional[jnp.ndarray] = 0, lagr_ineq_k : Optional[jnp.ndarray] = 0,
				extra_args_colocation : Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] =(None, None, None)):
		""" This function defines the update rule for the lagrangian multiplier for satisfaction of constraints
			:param params 		: The current weight of the neural network
			:param pen_eq_k 	: Penalty coefficient for the equality constraints 
			:param pen_ineq_sq_k: Penalty coefficient for the inequality constraints Phi(x,u,...) <= 0
			:param lagr_eq_k 	: Lagrangier multiplier for the equality constraints
			:param lagr_ineq_k 	: Lagranfier multiplier for the inequality constraints
			:param extra_args_colocation : Point used in order to enforce the constraints on unlabbelled data
		"""
		# Compute the loss associated to the collocation points 
		_, _, _, _, (_, eqCterms), (_, ineqCterms) = pred_xnext(params, None, None, None, extra_args_colocation=extra_args_colocation)
		# Update lagrangian term for equality constraints
		n_lagr_eq_k = jnp.array([]) if eqCterms.shape[0] <= 0 else lagr_eq_k + 2 * pen_eq_k * eqCterms
		# Update lagrangian term for inequality constraints
		n_lagr_ineq_k = jnp.array([]) if ineqCterms.shape[0] <= 0 else jnp.maximum(lagr_ineq_k + 2 * pen_ineq_sq_k * ineqCterms, 0)
		return pen_eq_k * beta_eq, pen_ineq_sq_k * beta_ineq, n_lagr_eq_k, n_lagr_ineq_k

	return (params_init, m_pen_eq_k, m_pen_ineq_k, m_lagr_eq_k, m_lagr_ineq_k), pred_xnext, (loss_fun, loss_fun_constr), (update, update_lagrange)