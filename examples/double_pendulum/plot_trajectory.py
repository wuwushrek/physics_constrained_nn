import numpy as np
import pickle

import matplotlib.pyplot as plt

import time
import argparse

import math

from physics_constrained_nn.utils import SampleLog

example_command ='python plot_trajectory.py --cfg data/double_pendulum_dt0p01.pkl --num_trajectory 3 --num_col 2 --size_x 5 --size_y 5'
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True, type=str, help=example_command)
parser.add_argument('--num_data', type=int, default=0)
parser.add_argument('--num_trajectory', type=int, default=0)
parser.add_argument('--num_trajectory_test', type=int, default=0)
parser.add_argument('--num_col', type=int, default=2)
parser.add_argument('--size_x', type=int, default=5)
parser.add_argument('--size_y', type=int, default=5)
args = parser.parse_args()

# Load the data file
m_file = open(args.cfg, 'rb')
mSampleLog = pickle.load(m_file)
m_file.close()

# Training data
xTrain = np.array(mSampleLog.xTrain)
# xnextTrainList = mSampleLog.xnextTrain
xtrain_lb = mSampleLog.lowU_train
xtrain_ub = mSampleLog.highU_train

# Testing data
xTest = np.array(mSampleLog.xTest)
# xnextTest = mSampleLog.xnextTest
xtest_lb = mSampleLog.lowU_test
xtest_ub = mSampleLog.highU_test

delta_time = mSampleLog.actual_dt

# xnextTrain = xnextTrainList[args.num_data]

size_traj = mSampleLog.disable_substep[2]
id_trajectory = args.num_trajectory * size_traj
id_trajectory_test = args.num_trajectory_test * size_traj
assert id_trajectory < xTrain.shape[0]
assert id_trajectory_test < xTest.shape[0]


ncols_fig = args.num_col
nrows_fig = math.ceil(xTrain.shape[1] / ncols_fig)
sizeSubFig_x = args.size_x
sizeSubFig_y = args.size_y
fig, axs = plt.subplots(nrows=nrows_fig, ncols=ncols_fig, 
			figsize=(ncols_fig*sizeSubFig_x,nrows_fig*sizeSubFig_y), 
			sharex=True, sharey=False)
# fig.suptitle('Training and testing data sets')
axis_name = [r'$\Theta_1$', r'$\Theta_2$', r'$\Omega_1$',  r'$\Omega_2$']
linewidth = 2
color_test = 'green'
color_train = 'red'
time_val = [ i * delta_time for i in range(size_traj)]
for i, yname, ax in zip(range(xTrain.shape[1]),axis_name, axs.flatten()):
	ax.plot(time_val, xTrain[id_trajectory:(id_trajectory+size_traj),i], linewidth= linewidth, color=color_train, label='Train data')
	ax.fill_between(time_val, [xtrain_lb[i]]*size_traj, [xtrain_ub[i]]*size_traj, alpha=0.5,  facecolor=color_train, edgecolor= 'dark'+color_train, linewidth=1, label='Train sample domain')
	ax.plot(time_val, xTest[id_trajectory_test:(id_trajectory_test+size_traj),i], linewidth= linewidth, color=color_test, label='Test data')
	ax.fill_between(time_val, [xtest_lb[i]]*size_traj, [xtest_ub[i]]*size_traj, alpha=0.5,  facecolor=color_test, edgecolor= 'dark'+color_test, linewidth=1, label='Test sample domain')
	ax.set_xlabel(r'Time (s)')
	ax.set_ylabel(yname)
	ax.grid()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fancybox=True)
plt.show()