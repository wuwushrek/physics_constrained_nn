from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

def _parse_requirements(requirements_txt_path):
  with open(requirements_txt_path) as fp:
    return fp.read().splitlines()

setup(
   name='physics_constrained_nn',
   version='0.0.1',
   description='A module for training neural networks of unknown dynamical systems that can incorporate physics knowledge in form of structural knowledge of constraints on the internal state of the model',
   license="GNU 3.0",
   long_description=long_description,
   author='Franck Djeumou and Cyrus Neary',
   author_email='fdjeumou@utexas.edu, cneary@utexas.edu',
   url="https://github.com/wuwushrek/physics_constrained_nn.git",
   packages=['physics_constrained_nn'],
   package_dir={'physics_constrained_nn': 'physics_constrained_nn/'},
   install_requires=_parse_requirements('requirements.txt'),
)
