from setuptools import setup, find_packages

setup(
    name='icem_mpc',
    version='0.0.1',
    description='ICEM MPC package for gripper task',

    packages=find_packages(include=['icem_mpc', 'icem_mpc.*']),
    python_requires='>=3.9',
)
