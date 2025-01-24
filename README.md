<a id="readme-top"></a>

# pyTTN: An Open Source Toolbox for Quantum Dynamics Simulations Using Tree Tensor Networks

<!--
[![ArXiv]()
[![Documentation Status]()
[![DOI]()
[![Tests status]()]()
[![Codecov]()
-->

## Links

* Gitlab:         <>
* Documentation:  <>
* PyPI:           <>

<!-- TABLE OF CONTENTS -->

# Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
- [Tutorials](#tutorials)


# About the Project

This open source project aims to provide an easy to use python interface for working with generic Tree Tensor Networks States to efficiently compute dynamics properties of quantum systems.  A key focus of this library is the easy setup of calculations employing either single or multiset tensor networks with generic tree structured connectivity.  Easy setup of Hamiltonians for arbitrary problems, with the ability to automatically apply techniques such as mode combination to reduce the total number of modes present in the system. Additionally, this library includes several tools to help facilitate applications of these approaches to study the dynamics of quantum systems that are strongly coupled to structured environment using both unitary methods (e.g. TEDOPA, T-TEDOPA and other representations of the system-bath Hamiltonian) as well as non-unitary approaches (e.g. Hierarchical Equations of Motion and Generalised Pseudomode method). 

pyTTN implements a range of numerically exact methods (methods that are systematically convergable to the exact results) for the dynamics of quantum system, and provides several example applications to
- Non-adiabatic dynamics of 24-mode pyrazine
- Exciton dynamics in a $n$-oligothiophene donor-C$_{60}$ fullerene acceptor system
- Dynamics of quantum systems coupled to a single or multiple bosonic or fermionic environment
- Interacting chains and trees of open quantum systems

-------------------------------------------------------------------------------


# Getting Started

## Prerequisites

## Installation
You can install pyTTN using pip like this:
```
$ cd ${pyTTN_ROOT_DIR}
$ python3 -m pip install .
```


## C++ Interface 
## Dependencies
External Libraries:
- Required: [Pybind11](https://github.com/pybind/pybind11) Python bindings
            [Catch2](https://github.com/catchorg/Catch2) C++ Unit Tests
            [BLAS](https://netlib.org/blas/) linear algebra
            [Lapack](https://netlib.org/lapack/) linear algebra
            [CMake](https://cmake.org/) Build System Version 3.11 onwards


The cmake build system can make use of Pybind11 and Catch2 version external libraries located in directory ${PyTTN_ROOT_DIR}/external. 
If the required libraries are not found it will attempt to pull them from github. 


When compiling with Clang or AppleClang this method searches for LLVM using the FindLLVM.cmake module that is included within CMake.

# Compile Instructions
This code requires cmake version 3.11 in order to compile. From the pyTTN base directory (${pyTTN_ROOT_DIR}) run:
```console
mkdir build
cd build
cmake-DBUILD_PYTHON_BINDINGS=OFF -DBUILD_SRC=ON ../ 
make
make install
```

This will build all .cpp files in the ${pyTTN_ROOT_DIR}/src folder. 

This code has been successfully tested on: 
* Linux Mint 21 Cinnamon with Kernel Version 5.15.0-50-generic using g++11.2.0 with OpenBLAS and clang-14.0.0-1ubuntu with the current system versions of Lapack and Blas
* CentOS release 6.6 with Kernel Version 2.6.32-504.16.2.el6.x86_64 using g++-10.1.0 and with MKL/17.0

Typical installation times are $\lesssim$ 2 minutes.

## Running the Software
Example python scripts for running the software are provided in ${PyTTN_ROOT_DIR}/examples

# Tutorials

The pyTTn repository contains several 










