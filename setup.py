from setuptools import setup

setup(
    name='myDiffX',
    version='0.0.1',    
    description='A JAX based python library for numerical solvers for differential equations',
    url='https://github.com/akagam1/myDiffX',
    author='Arjun Puthli',
    author_email='arjunputhli2003@gmail.com',
    license='MIT',
    packages=['myDiffX'],
    install_requires=['jax',                     
                      ],
)
