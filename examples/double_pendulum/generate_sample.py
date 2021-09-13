import yaml
from collections import namedtuple
import pickle

# Import JAX
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit

from jax.experimental.ode import odeint
from physics_constrained_nn.utils import SampleLog

from tqdm import tqdm


def pendulum_unknown_terms(state, u=None, t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
	""" Function defining the unknown terms f1 and f2 of the dynamics
	"""
	t1, t2, w1, w2 = state
	return {'f1' : jnp.array([-(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * jnp.sin(t1 - t2) - (g / l1) * jnp.sin(t1)]), 
			'f2' : jnp.array([(l1 / l2) * (w1**2) * jnp.sin(t1 - t2) - (g / l2) * jnp.sin(t2)])}


def pendulum_unknown_terms_aux(state, u=None, t=0, m1=1, m2=1, l1=1, l2=1, g=9.8):
	""" Function defining the unknown terms f1 and f2 of the dynamics
	"""
	t1, t2, w1, w2 = state
	return jnp.array([-(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * jnp.sin(t1 - t2) - (g / l1) * jnp.sin(t1)]), jnp.array([(l1 / l2) * (w1**2) * jnp.sin(t1 - t2) - (g / l2) * jnp.sin(t2)])

# # Build the batch version of this function
# batch_unknown_fun = jax.jit(vmap(unkn_fun))

def pendulum_known_terms(state : jnp.ndarray, u: jnp.ndarray =None , f1 : jnp.ndarray =None, f2 : jnp.ndarray =None, 
				t: jnp.float32=0 , m1 : jnp.float32 =1, m2 : jnp.float32 =1, l1 : jnp.float32 =1, 
				l2 : jnp.float32 =1, g : jnp.float32 =9.8) -> jnp.ndarray:
	""" Function defining the known part of the dynamics given the unknown terms
	"""
	# f1 and f2 are functions that return a scalar. So here make sure to use them as scalars vectors
	t1, t2, w1, w2 = state
	a1 = (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(t1 - t2)
	a2 = (l1 / l2) * jnp.cos(t1 - t2)
	g1 = (f1[0] - a1 * f2[0]) / (1 - a1 * a2)
	g2 = (f2[0] - a2 * f1[0]) / (1 - a1 * a2)
	return jnp.stack([w1, w2, g1, g2])

# # Build the batch version of the function
# batch_known_fun = jax.jit(vmap(known_fun))

def pendulum_analytical(state : jnp.ndarray, u: jnp.ndarray = None, t: jnp.float32=0 , m1 : jnp.float32 =1, 
				m2 : jnp.float32 =1, l1 : jnp.float32 =1, l2 : jnp.float32 =1, g : jnp.float32 =9.8) -> jnp.ndarray:
	""" Double pendulum full known dynamics
	"""
	return pendulum_known_terms(state, u, **pendulum_unknown_terms(state,u,t,m1,m2,l1,l2,g), t=t, m1=m1, m2=m2, l1=l1, l2=l2, g=g)

# # Build the batch version of this function
# batch_dfun = jax.jit(jax.vmap(f_analytical))


def custom_constraints(state, u=None, f1=None, f2=None):
	""" Build the constraints on the unknown terms of the dynamics
		THIS FUNCTION TAKES BATCHES AS INPUTS AND RETURNS A COUPLE OF EQUALITY AND INEQUALITY CONSTRAINTS
	"""
	ineqContr = jnp.array([])
	xfus = jnp.hstack((state,u)) if u is not None else state
	f1_fun, ind1 = f1
	f2_fun, ind2 = f2
	# c1 = f1_fun(xfus[:,ind1]) + f1_fun(-xfus[:,ind1])
	# c2 = f2_fun(xfus[:,ind2]) + f2_fun(-xfus[:,ind2])
	f1Val = f1_fun(xfus[:,ind1])
	f2Val = f2_fun(xfus[:,ind2])
	c3 = f1Val + f1_fun(jnp.hstack((-xfus[:,0:2], xfus[:,3:4])))
	c4 = f2Val + f2_fun(jnp.hstack((-xfus[:,0:2], xfus[:,2:3])))
	c5 = f1Val - f1_fun(jnp.hstack((xfus[:,0:2], -xfus[:,3:4])))
	c6 = f2Val - f2_fun(jnp.hstack((xfus[:,0:2], -xfus[:,2:3])))
	return jnp.hstack((c3,c4,c5,c6)), ineqContr

# def custom_constraints(state, u=None, f1=None, f2=None):
# 	""" Unbatched function
# 	"""
# 	ineqContr = jnp.array([])
# 	xfus = jnp.hstack((state,u)) if u is not None else state
# 	f1_fun, ind1 = f1
# 	f2_fun, ind2 = f2
# 	# c1 = f1_fun(xfus[:,ind1]) + f1_fun(-xfus[:,ind1])
# 	# c2 = f2_fun(xfus[:,ind2]) + f2_fun(-xfus[:,ind2])
# 	f1Val = f1_fun(xfus[ind1])
# 	f2Val = f2_fun(xfus[ind2])
# 	c3 = f1Val + f1_fun(jnp.hstack((-xfus[0:2], xfus[3:4])))
# 	c4 = f2Val + f2_fun(jnp.hstack((-xfus[0:2], xfus[2:3])))
# 	c5 = f1Val - f1_fun(jnp.hstack((xfus[0:2], -xfus[3:4])))
# 	c6 = f2Val - f2_fun(jnp.hstack((xfus[0:2], -xfus[2:3])))
# 	return jnp.hstack((c3,c4,c5,c6)), ineqContr


@jit
def solve_analytical(initial_state, times):
	""" Given an initial state and a set of time instant, compute a trajectory of the system at each time in the set
	"""
	return odeint(pendulum_analytical, initial_state, t=times, rtol=1e-10, atol=1e-10)

def gen_domain_shift(rng_key, time_step, num_traj, trajectory_length, n_rollout, x0_init_lb, x0_init_ub, merge_traj=True):
	""" Generate trajectory of the pendulum from a random initial distribution between x0_init_lb and x0_init_ub
	"""
	# # Integrate for smaller time step
	# canonical_step_size = 0.01
	# sup_points = int(round(time_step / canonical_step_size))
	# tIndexes = jnp.linspace(0, sup_points*(trajectory_length + n_rollout) * time_step,
	# 						num=sup_points*(trajectory_length + n_rollout)) # dtype specified
	# indexX = jnp.array([ i for i in range(0, tIndexes.shape[0], sup_points)])
	tIndexes = jnp.linspace(0, 10*(trajectory_length + n_rollout) * time_step,
							num= 10*(trajectory_length + n_rollout)) # dtype specified
	res_currx = []
	res_shift = [ [] for r in range(n_rollout)]
	for i in tqdm(range(num_traj), leave=False):
		while True:
			rng_key, subkey = jax.random.split(rng_key)
			x0val = jax.random.uniform(subkey, shape=x0_init_lb.shape, minval=x0_init_lb, maxval=x0_init_ub)
			xnextVal = solve_analytical(x0val, tIndexes) # Two dimension array with
			fknown1, fknown2 = jax.vmap(pendulum_unknown_terms_aux, (0,))(xnextVal)
			idx_pairs = jnp.where(jnp.diff(jnp.hstack(([False],jnp.logical_or(fknown1.ravel()<=0, fknown2.ravel()>=0),[False]))))[0].reshape(-1,2)
			length_seqs = jnp.diff(idx_pairs,axis=1)
			val_seq = int(jnp.diff(idx_pairs,axis=1).argmax())
			start_seq, end_seq = idx_pairs[val_seq,0], idx_pairs[val_seq,1]
			if end_seq - start_seq >= (trajectory_length + n_rollout):
				xnextVal = xnextVal[start_seq:end_seq,:]
				break

		# Check wether a long enough trajectory can be extracted that satisfy pendulum_unknown_terms(state) < 0  or pendulum_unknown_terms(state) > 0
		# xnextVal = xnextVal[indexX, :]
		if merge_traj:
			res_currx.extend(xnextVal[:trajectory_length,:])
		else:
			res_currx.append(xnextVal[:trajectory_length,:])
		for j, r in enumerate(res_shift):
			r.extend(xnextVal[(j+1):(j+1+trajectory_length),:])
	return rng_key, jnp.array(res_currx), jnp.array(res_shift)

def gen_samples(rng_key, time_step, num_traj, trajectory_length, n_rollout, x0_init_lb, x0_init_ub, merge_traj=True):
	""" Generate trajectory of the pendulum from a random initial distribution between x0_init_lb and x0_init_ub
	"""
	# # Integrate for smaller time step
	# canonical_step_size = 0.01
	# sup_points = int(round(time_step / canonical_step_size))
	# tIndexes = jnp.linspace(0, sup_points*(trajectory_length + n_rollout) * time_step,
	# 						num=sup_points*(trajectory_length + n_rollout)) # dtype specified
	# indexX = jnp.array([ i for i in range(0, tIndexes.shape[0], sup_points)])
	tIndexes = jnp.linspace(0, (trajectory_length + n_rollout) * time_step,
							num=(trajectory_length + n_rollout)) # dtype specified
	res_currx = []
	res_shift = [ [] for r in range(n_rollout)]
	for i in tqdm(range(num_traj), leave=False):
		rng_key, subkey = jax.random.split(rng_key)
		x0val = jax.random.uniform(subkey, shape=x0_init_lb.shape, minval=x0_init_lb, maxval=x0_init_ub)
		xnextVal = solve_analytical(x0val, tIndexes) # Two dimension array with
		# xnextVal = xnextVal[indexX, :]
		if merge_traj:
			res_currx.extend(xnextVal[:trajectory_length,:])
		else:
			res_currx.append(xnextVal[:trajectory_length,:])
		for j, r in enumerate(res_shift):
			r.extend(xnextVal[(j+1):(j+1+trajectory_length),:])
	return rng_key, jnp.array(res_currx), jnp.array(res_shift)


def load_config_yaml(path_config_file, extra_args={}):
	""" Load the yaml configuration file giving the training/testing information
		:param path_config_file : Path to the adequate yaml file
	"""
	yml_file = open(path_config_file).read()
	m_config = yaml.load(yml_file, yaml.SafeLoader)
	m_config = {**m_config, **extra_args}
	print('################# Configuration file #################')
	print(m_config)
	print('######################################################')

	# Parse the time step
	time_step = m_config.get('time_step', 0.01)

	# Parse the different seed
	seed_number = m_config.get('seed', 1) # A list of seed for multiple evaluation

	# Initial point randomly generated from these sets in the training/testing
	x0_init_lb_train = m_config.get('x0_init_lb_train', [-3.14/10, -3.14/10, -1.0, -1.0])
	x0_init_ub_train = m_config.get('x0_init_ub_train', [0, 0, 1.0, 1.0])
	x0_init_lb_test = m_config.get('x0_init_lb_test', [-3.14/10, -3.14/10, -1.0, -1.0])
	x0_init_ub_test = m_config.get('x0_init_ub_test', [3.14/4, 3.14/4, 1.0, 1.0])

	# Define the number of roolout
	n_rollout = m_config.get('n_rollout', 5)

	# Number of points for colocation method
	n_coloc = m_config.get('n_coloc', 0)

	# Number of training data and testing data
	num_trajectory_train = m_config.get('num_trajectory_train', [20])
	num_trajectory_test = m_config.get('num_trajectory_test', 200)
	size_trajectory = m_config.get('size_trajectory', 500)

	# Parse the output file 
	output_file = m_config.get('output_file', 'double_pendulum')

	return time_step, seed_number, jnp.array(x0_init_lb_train), jnp.array(x0_init_ub_train),\
			jnp.array(x0_init_lb_test), jnp.array(x0_init_ub_test),\
			n_rollout, num_trajectory_train, num_trajectory_test, size_trajectory, n_coloc, output_file


def main_fn(path_config_file, extra_args={}):
	# Get the parameters from the config file
	time_step, seed_number, x0_init_lb_train, x0_init_ub_train, x0_init_lb_test, x0_init_ub_test,\
			n_rollout, num_trajectory_train, num_trajectory_test, size_trajectory, n_coloc, output_file = \
			load_config_yaml(path_config_file, extra_args)
	# Random number genrator
	rng_key = jax.random.PRNGKey(seed_number)
	f_xTrain, f_xTrainRolled = list(), list()
	# f_xTest, f_xTestRolled = list(), list()
	rng_key, xTest, xTestRolled = gen_samples(rng_key, time_step, num_trajectory_test, size_trajectory, n_rollout, x0_init_lb_test, x0_init_ub_test)
	# Get the highest number of trajectories
	max_traj = int(jnp.max(jnp.array(num_trajectory_train)))
	# Generate the Training set via the set of trajectory size
	# rng_key, xTrain, xTrainRolled = gen_samples(rng_key, time_step, max_traj, size_trajectory, n_rollout, x0_init_lb_train, x0_init_ub_train)
	rng_key, xTrain, xTrainRolled = gen_domain_shift(rng_key, time_step, max_traj, size_trajectory, n_rollout, x0_init_lb_train, x0_init_ub_train)
	# exit()
	# Generate the colocation points
	coloc_set = jax.random.uniform(rng_key, shape=(n_coloc, x0_init_lb_test.shape[0]), minval=x0_init_lb_test, maxval=x0_init_ub_test)
	# # Iterate through to generate data
	# for n_train in tqdm(num_trajectory_train, total=len(num_trajectory_train)):
	# 	rng_key, xTrain, xTrainRolled = gen_samples(rng_key, time_step, n_train, size_trajectory, n_rollout, x0_init_lb_train, x0_init_ub_train)
	# 	f_xTrain.append(xTrain)
	# 	f_xTrainRolled.append(xTrainRolled)
	m_log = SampleLog(xTrain=xTrain, xTrainExtra=(None, coloc_set), uTrain=None, xnextTrain=xTrainRolled, 
						lowU_train=x0_init_lb_train, highU_train=x0_init_ub_train, xTest=xTest, xTestExtra=None, 
						uTest=None, xnextTest=xTestRolled, lowU_test=x0_init_lb_test, highU_test=x0_init_ub_test, 
						env_name='double_pendulum', env_extra_args=None, m_rng=None, seed_number=seed_number, qp_indx=None, qp_base=None,
						n_state=xTrain.shape[1], n_control=0, actual_dt=time_step, disable_substep=(False, num_trajectory_train, size_trajectory), 
						control_policy=None, n_rollout=n_rollout)
	# m_log = SampleLog(f_xTrain, None, None, f_xTrainRolled, None, None, f_xTest, None, None, f_xTestRolled, x0_init_lb_test, 
	# 	    x0_init_ub_test, None, None, None, seed_number, None, None, xTrain.shape[1], 0, time_step, 
	# 	    size_trajectory, None, n_rollout)
	mFile = open(output_file+'.pkl', 'wb') 
	pickle.dump(m_log, mFile)
	mFile.close()

if __name__ == "__main__":
	import time
	import argparse
	example_command ='python generate_sample.py --cfg dataset_gen.yaml --output_file data/xxxx --time_step 0.01 --n_rollout 5'
	# Command line argument for setting parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--cfg', required=True, type=str, help=example_command)
	parser.add_argument('--output_file', type=str, default='')
	parser.add_argument('--time_step', type=float, default=0)
	parser.add_argument('--n_rollout', type=int, default=0)
	args = parser.parse_args()
	m_config_aux = {'cfg' : args.cfg}
	if args.output_file != '':
		m_config_aux['output_file']  = args.output_file
	if args.time_step > 0:
		m_config_aux['time_step']  = args.time_step
	if args.n_rollout > 0:
		m_config_aux['n_rollout']  = args.n_rollout

	main_fn(args.cfg, m_config_aux)