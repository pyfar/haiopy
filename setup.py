#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'sounddevice', 'numpy', 'pyfar']

setup_requirements = ['pytest-runner']

test_requirements = ['pytest>=3']

python_requirements = '>=3.8'

setup(
    author="The pyfar developers",
    author_email='marco.berzborn@akustik.rwth-aachen.de',
    python_requires=python_requirements,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Scientists',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="Python package for handling playingback and recording.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='haiopy',
    name='haiopy',
    packages=find_packages(include=['haiopy', 'haiopy.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/pyfar/haiopy',
    version='0.1.0',
    zip_safe=False,
)
