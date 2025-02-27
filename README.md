<a id="readme-top"></a>

# pyTTN: An Open Source Toolbox for Open and Closed System Quantum Dynamics Simulations Using Tree Tensor Networks
<!-- Add badges when complete -->
<!--
[![ArXiv]()
[![Documentation Status]()
[![DOI]()
[![Tests status]()]()
[![Codecov]()
-->

## Links

<!-- Add links when complete -->
* National Physical Laboratory: <https://www.npl.co.uk/>
* Gitlab:         <https://gitlab.npl.co.uk/quantum-software/pyttn>
* Documentation:  <>
* PyPI:           <>

<!-- TABLE OF CONTENTS -->

# Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
- [Tutorials](#tutorials)


# About the Project

This open source project aims to provide an easy to use python interface for working with generic Tree Tensor Networks States to efficiently compute dynamics properties of quantum systems.  A key focus of this library is the easy setup of calculations employing either single or multiset tensor networks with generic tree structured connectivity.  Easy setup of Hamiltonians for arbitrary problems, with the ability to automatically apply techniques such as mode combination to reduce the total number of modes present in the system. Additionally, this library includes several tools to help facilitate applications of these approaches to study the dynamics of quantum systems that are strongly coupled to structured environment using both unitary methods (e.g. TEDOPA, T-TEDOPA and other representations of the system-bath Hamiltonian) as well as non-unitary approaches (e.g. Hierarchical Equations of Motion and Generalised Pseudomode method). 

<!-- Add hyperlinks to the examples here -->
pyTTN implements a range of numerically exact methods (methods that are systematically convergable to the exact results) for the dynamics of quantum system, and provides several example applications to
- Non-adiabatic dynamics of 24-mode pyrazine
- Exciton dynamics in a $n$-oligothiophene donor-C<sub>60</sub> fullerene acceptor system
- Dynamics of quantum systems coupled to a single or multiple bosonic or fermionic environment
- Interacting chains and trees of open quantum systems

The core Tree Tensor Network Library (ttnpp) has been developed in C++ with pyTTN using a pybind11 based python wrapper to expose the core functionality to python.  The core ttnpp library is a header only C++ library and it is straightforward to compile pure C++ programs using this library.  For details see [TTNPP Library](#ttnpp-library-compilation).

-------------------------------------------------------------------------------


# Getting Started

## Prerequisites
The core C++ library relies and [Pybind11](https://github.com/pybind/pybind11) make use of the [CMake](https://cmake.org/) build system and require Version 3.11 or onwards.


### Dependencies
The core C++ library (ttnpp) and the python wrapper (pyTTN) have the following key dependencies.  

External Libraries:
- [Pybind11](https://github.com/pybind/pybind11) Python bindings
- [Catch2](https://github.com/catchorg/Catch2) C++ Unit Tests
- [BLAS](https://netlib.org/blas/) linear algebra
- [Lapack](https://netlib.org/lapack/) linear algebra

The cmake build system can make use of the [Pybind11](https://github.com/pybind/pybind11) and [Catch2](https://github.com/catchorg/Catch2) external libraries located in directory ${PyTTN_ROOT_DIR}/external.  If these libraries are not found in this location it will attempt to pull them from github.  For [BLAS](https://netlib.org/blas/) and [Lapack](https://netlib.org/lapack/) linear algebra, the cmake build script uses the standard find_lapack and find_blas calls to locate the libraries. When compiling with Clang or AppleClang this method searches for LLVM using the FindLLVM.cmake module that is included within CMake.

pyTTN also offers experimental support for the use of a CUDA backend to accelerate the internal tensor operations.  When compiling the CUDA backend, pyTTN gains the following additional dependencies:
External Libraries:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 
- [cuTENSOR](https://developer.nvidia.com/cutensor)

By default pyTTN does not build the CUDA backend.  For details on how to do so see ...

## Installation
You can install pyTTN using pip like this:
```
$ cd ${pyTTN_ROOT_DIR}
$ python3 -m pip install .
```

This will build the script in 

### Building with CUDA Support

## Using the Software
Example python scripts showing the use of pyTTN for a range of application are provided in the ${PyTTN_ROOT_DIR}/examples.  These examples included

## TTNPP Library
It is possible to compile pure C++ programs that make use of the core C++ library (ttnpp) and currently a cmake script has been provided that can conditionally compile against 

### Compile Instructions
This code requires cmake version 3.11 in order to compile. From the pyTTN base directory (${pyTTN_ROOT_DIR}) run:
```console
mkdir build
cd build
cmake-DBUILD_PYTHON_BINDINGS=OFF -DBUILD_SRC=ON ../ 
make
make install
```

This will build all .cpp files in the ${pyTTN_ROOT_DIR}/src folder.  Typical installation times are $\lesssim$ 2 minutes.

# Tutorials

The pyTTN repository contains several 










