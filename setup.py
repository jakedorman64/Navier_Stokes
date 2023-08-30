from setuptools import setup

setup(
	name='navier_stokes',
	version='0.1',
	description='A Navier Stokes Solver using JAX',
	author='Jake Dorman',
	packages=['navier_stokes'],
	install_requires=['jax']
)
