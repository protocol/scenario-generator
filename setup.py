#!/usr/bin/env python3

import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

__author__ = 'Kiran Karra, Tom Mellan'
__email__ = 'kiran.karra@protocol.ai, tom.mellan@protocol.ai'
__version__ = '0.1.0'

install_requires = ['numpy',
                    'pandas',
                    'numpyro',
                    'jax',
                    'mechaFIL',
                    ]

setuptools.setup(
    name="scenario_generator",
    version=__version__,

    description='Scenario generator for understanding Filecoin dynamics',
    long_description=long_description,
    long_description_content_type="text/markdown",

    url = 'https://github.com/protocol/scenario-generator',

    author=__author__,
    author_email=__email__,

    license='MIT License',

    python_requires='>=3',
    packages=['scenario_generator'],

    install_requires=install_requires,

    zip_safe=False
)