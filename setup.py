#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import versioneer

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ["click", "h5py", "matplotlib", "mmh3", "numpy", "packaging",
                "pyevtk", "pytest", "pytorch"]

setup_requirements = ['pytest-runner', 'pytest']

setup(
    author="Andreas Kraemer",
    author_email='kraemer.research@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.12',
    ],
    description="Lattice Boltzmann Python GPU",
    entry_points={
        'console_scripts': [
            'lettuce=lettuce.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    package_data={
        'lettuce.native_generator':
            ['lettuce/native_generator/template/setup.py']
    },
    include_package_data=True,
    keywords='lettuce',
    name='lettuce',
    packages=find_packages(include=[
        'lettuce',
        'lettuce.ext',
        'lettuce.ext._boundary',
        'lettuce.ext._collision',
        'lettuce.ext._equilibrium',
        'lettuce.ext._flows',
        'lettuce.ext._force',
        'lettuce.ext._reporter',
        'lettuce.ext._stencil',
        'lettuce.cuda_native',
        'lettuce.cuda_native.ext',
        'lettuce.cuda_native.ext._boundary',
        'lettuce.cuda_native.ext._collision',
        'lettuce.cuda_native.ext._equilibrium',
        'lettuce.cuda_native.ext._force',
        'lettuce.util']),
    setup_requires=setup_requirements,
    test_suite='tests',
    url='https://github.com/lettucecfd/lettuce',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
