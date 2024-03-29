# coding=utf-8
"""
Setup for scikit-surgery-sphere-fitting
"""

from setuptools import setup, find_packages
import versioneer

# Get the long description
with open('README.rst') as f:
    long_description = f.read()

setup(
    name='scikit-surgery-sphere-fitting',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='scikit-surgery-sphere-fitting implements a least squares sphere fitting algorithm, to read a vtk poly data file, a config file, and outputs the fitted sphere',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/mxochicale/sksurgeryspherefitting',
    author='Miguel Xochicale',
    author_email='YOUR-EMAIL@ucl.ac.uk',
    license='BSD-3 license',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',


        'License :: OSI Approved :: BSD License',


        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],

    keywords='medical imaging',

    packages=find_packages(
        exclude=[
            'doc',
            'tests',
        ]
    ),

    install_requires=[
        'numpy>=1.11',
        'spicy',
        'vtk>=9.2.5',
        'scikit-surgeryvtk==2.0rc0'
    ],

    entry_points={
        'console_scripts': [
            'sksurgeryspherefitting=sksurgeryspherefitting.__main__:main',
        ],
    },
)
