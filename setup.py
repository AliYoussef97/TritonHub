from setuptools import setup, find_packages

setup(
    name='TritonFactory',
    version='1.0.1',
    author='Ali Youssef',
    description='A collection of Triton-based neural network modules',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AliYoussef97/Triton-Factory',
    packages=find_packages(exclude=['build', 'dist', 'TritonFactory.egg-info', 'UnitTests']),
    python_requires='>=3.7',
    install_requires=['torch',
                      'triton'],
    )