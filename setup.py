#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
from subprocess import Popen

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import versioneer

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', "torch>=1.2", "numpy", "matplotlib", "pyevtk"]

setup_requirements = ['pytest-runner', 'pytest']


def get_cmdclass():
    """merge cmdclass of versioneer with the cmdclass of torch's cpp build extension"""

    cmdclass = versioneer.get_cmdclass()

    # This assert should not fail as versioneer (right now) not writes build_ext.
    # This assert should prevent bugs when versioneer changes its behavior.
    assert not ('build_ext' in cmdclass), "versioneer should not write 'build_ext' in cmdclass." \
                                          "Please contact the developers about this bug!"

    cmdclass['build_ext'] = BuildExtension
    return cmdclass


# def get_native_sources():
#     """"""
#
#     process = Popen(['python', '-m', 'lettuce.gen_native'])
#     _, stderr = process.communicate()
#     assert stderr is None, stderr
#
#     def source(f: str):
#         is_file = os.path.isfile(os.path.join('lettuce_native', f))
#         is_source = f.endswith('.cu') or f.endswith('.cpp')
#         return is_file and is_source
#
#     return [os.path.join('lettuce_native', f) for f in os.listdir('lettuce_native') if source(f)]


setup(
    author="Andreas Kraemer",
    author_email='kraemer.research@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Lattice Boltzmann Python GPU",
    entry_points={
        'console_scripts': [
            'lettuce=lettuce.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='lettuce',
    name='lettuce',
    # ext_modules=[
    #     CUDAExtension(
    #         name='lettuce_native',
    #         sources=get_native_sources()
    #     )
    # ],
    packages=find_packages(include=['lettuce', 'lettuce.flows', 'lettuce.gen_native', 'lettuce.native']),
    setup_requires=setup_requirements,
    test_suite='tests',
    url='https://github.com/lettucecfd/lettuce',
    version=versioneer.get_version(),
    cmdclass=get_cmdclass(),
    zip_safe=False,
)
