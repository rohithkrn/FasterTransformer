#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import setuptools.command.build_py
import subprocess
from setuptools import setup, find_packages


def copy_files():
    subprocess.run(["./install_utils.sh"])


def detect_version():
    with open("version.txt", "r") as f:
        return f.readlines()[0]


def pypi_description():
    with open('PyPiDescription.rst') as df:
        return df.read()


class BuildPy(setuptools.command.build_py.build_py):

    def run(self):
        setuptools.command.build_py.build_py.run(self)


if __name__ == '__main__':
    copy_files()
    pkgs = find_packages()
    version = detect_version()

    requirements = ['psutil', 'packaging', 'wheel', 'torch', 'numpy', 'transformers', 'peft']

    test_requirements = []

    setup(name='fastertransformer',
          version=version,
          description=
          'fastertransformer is the easy frontend for LLM on fastertransformer library',
          author='Deep Java Library team',
          author_email='djl-dev@amazon.com',
          long_description=pypi_description(),
          url='https://github.com/NVIDIA/FasterTransformer.git',
          keywords='FasterTransformer frontend API',
          packages=pkgs,
          cmdclass={
              'build_py': BuildPy,
          },
          install_requires=requirements,
          extras_require={'test': test_requirements + requirements},
          include_package_data=True,
          license='Apache License Version 2.0')
