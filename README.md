# Neural Networks with Physics-Informed Architectures and Constraints for Dynamical Systems Modeling

This module builds custom deep neural networks to learn dynamics when prior physics knowledge and constraints about the underlying unknown dynamics are considered. 
The corresponding paper can be found [here](https://arxiv.org/pdf/2109.06407.pdf).

## Installation

The code is written both in C++ and Python.

### Quick Installation


This package requires [``jax``](https://github.com/google/jax) to be installed: The choice of CPU or GPU version depends on the user but the CPU is installed by default along with the package.
The package further requires [``dm-haiku``](https://github.com/deepmind/dm-haiku) for neural networks in jax and [``optax``](https://github.com/deepmind/optax) a gradient processing and optimization library for JAX, and [``Brax``](https://github.com/google/brax) :  a differentiable physics engine that simulates environments made up of rigid bodies, joints, and actuators. The following commands install everything that is required (except for the GPU version of JAX which must be installed manually):

```
git clone https://github.com/wuwushrek/physics_constrained_nn.git
cd physics_constrained_nn/
python3 -m pip install -e . 
```

### Detailed Installation

This package implements several jax primitives in C++ of ``MuJoCo`` functions that can be used as prior physics knowledge. Then it uses [``pybind11``]() to import the primitives and use it in Python. To include such primitives, ``MuJoCo`` needs to be installed on the target computer with a valid activation key. 

#### MuJoCo Experiments
Follow [the installation procedure](https://www.roboti.us/) to install ``MuJoCo``. Then, set the environment variables ``MUJOCO_PY_MJKEY_PATH`` and ``MUJOCO_PY_MUJOCO_PATH`` ( these names are typically used by ``mujoco-py`` to find the MuJoCo library files). For example, if the binaries, include and libraries of MuJoCo are unzipped in ``~/.mujoco/mujoco200_linux``, then you can excute the following
```
echo 'export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco200_linux' >> ~/.bashrc 
echo 'export MUJOCO_PY_MJKEY_PATH=~/.mujoco/mujoco200_linux/bin/mjkey.txt' >> ~/.bashrc 
source ~/.bashrc
```

Finally, install the following dependencies to compile the C++ code
```
sudo apt install build-essential libomp-dev
```

#### Brax Experiments
Follow [the installation procedure](https://github.com/google/brax) to install ``brax`` :  a differentiable physics engine that simulates environments made up of rigid bodies, joints, and actuators.


## Examples

### Double Pendulum Training

To first generate the data required to train the neural network, modify the ``dataset_gen.yaml`` file inside the double pendulum file and generate the dataset as follows:
```
cd physics_constrained_nn/examples/double_pendulum
python generate_sample.py --cfg dataset_gen.yaml --output_file DEST_FILE/datatrain
```

After the files has been generated, modify the parameters of your training from ``nets_params.yaml`` and proceed to the training as follows
```
python train.py --cfg nets_params.yaml --input_file DEST_FILE/datatrain.pkl --output_file DEST_FILE/base_datatrain_si0 --baseline base --side_info 0
```
where the baseline is either `base` or `rk4` and the side info is either `0` (no side information), `1` (structural knowledge of vector field), and `2` (structural knowledge + symmetry constraints).

Finally, to plot the results, execute the command line
```
python perform_comparison.py --logdirs DEST_FILE/base_datatrain_si0 DEST_FILE/base_datatrain_si1 ... --legend 'No SI' 'Si 2' ... --colors red green ... --num_traj 100 --num_point_in_traj 100 --seed 5 --show_constraints --window 5
```

### Brax Environment Training

The training is performed similarly to the Double pendulum training. In the `examples/brax` file, there is a list of files associated to each Brax environments. To generate the data for training on the `reacher` environment for example, execute the following
```
cd physics_constrained_nn/examples/brax
python generate_sample.py --cfg reacher_brax/dataset_gen.yaml --output_file DEST_FILE/datatrain
```

After the files has been generated, modify the parameters of your training from ``reacher_brax/nets_params.yaml`` and proceed to the training as follows

```
python train.py --cfg reacher_brax/nets_params.py --input_file DEST_FILE/datatrain.pkl --output_file DEST_FILE/base_datatrain_si0 --baseline base --side_info 0
```
where the baseline is either `base` or `rk4` and the side info is either `0` (no side information), `1` (structural knowledge of vector field), and `2` (structural knowledge + symmetry constraints).

Finally, to plot the results, execute the command line
```
python perform_comparison.py --logdirs DEST_FILE/base_datatrain_si0 DEST_FILE/base_datatrain_si1 ... --legend 'No SI' 'Si 2' ... --colors red green ... --num_traj 100 --num_point_in_traj 100 --seed 5 --show_constraints --window 5
```
