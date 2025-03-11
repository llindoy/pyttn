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

<!-- TABLE OF CONTENTS -->

## Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
- [Software Overview](#software-overview)

-------------------------------------------------------------------------------

## About the Project

This open source project aims to provide an easy to use python interface for working with generic Tree Tensor Networks States to efficiently compute dynamics properties of quantum systems.  A key focus of this library is the easy setup of calculations employing either single or multiset tensor networks with generic tree structured connectivity.  Easy setup of Hamiltonians for arbitrary problems, with the ability to automatically apply techniques such as mode combination to reduce the total number of modes present in the system. Additionally, this library includes several tools to help facilitate applications of these approaches to study the dynamics of quantum systems that are strongly coupled to structured environment using both unitary methods (e.g. TEDOPA, T-TEDOPA and other representations of the system-bath Hamiltonian) as well as non-unitary approaches (e.g. Hierarchical Equations of Motion and Generalised Pseudomode method). 

-------------------------------------------------------------------------------

# Getting Started

## Prerequisites
The core C++ library relies and [Pybind11](https://github.com/pybind/pybind11) make use of the [CMake](https://cmake.org/) build system and require Version 3.11 or onwards.

### Dependencies
The core C++ library (ttnpp) and the python wrapper (pyTTN) have the following key dependencies. 

External Libraries:
- [Pybind11](https://github.com/pybind/pybind11) Python bindings
- [BLAS](https://netlib.org/blas/) linear algebra
- [Lapack](https://netlib.org/lapack/) linear algebra
- [Catch2](https://github.com/catchorg/Catch2) C++ Unit Tests (Only required when running C++ test)

The cmake build system can make use of the [Pybind11](https://github.com/pybind/pybind11) and [Catch2](https://github.com/catchorg/Catch2) external libraries located in directory ${pyTTN_ROOT_DIR}/external.  If these libraries are not found in this location it will attempt to pull them from github.  For [BLAS](https://netlib.org/blas/) and [Lapack](https://netlib.org/lapack/) linear algebra, the cmake build script uses the standard find_lapack and find_blas calls to locate the libraries. When compiling with Clang or AppleClang this method searches for LLVM using the FindLLVM.cmake module that is included within CMake.

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

By default, this will make use of a single threaded build for compiling the Pybind11 wrapper and can take a number of minutes to complete.  It is possible to make use of multi-threaded builds when compiling the Pybind11.  This can be done by setting the environment variable `CMAKE_BUILD_PARALLEL_LEVEL`, e.g.
```
export CMAKE_BUILD_PARALLEL_LEVEL=8
```
to allow for the use of 8 threads when compiling.

<!-- Add badges when complete 
### Building with CUDA Support
[!Note]
Work in progress
-->

## Using the Software
Example python scripts showing the use of pyTTN for a range of application are provided in the ${pyTTN_ROOT_DIR}/examples. pyTTN implements a range of numerically exact methods (methods that are systematically convergable to the exact results) for the dynamics of quantum system  and provides several example applications to
- [Non-adiabatic dynamics of 24-mode pyrazine](examples/pyrazine/)
- [Exciton dynamics in a $n$-oligothiophene donor-C<sub>60</sub> fullerene acceptor system](examples/p3ht_pcbm_heterojunction/)
- Dynamics of quantum systems coupled to a [bosonic](examples/spin_boson_model/) or [fermionic](examples/anderson_impurity_model) environment
- Interacting [chains](examples/dissipative_spin_models/chain/) and [trees](examples/dissipative_spin_models/cayley_tree/) of open quantum systems

## Tutorials

## TTNPP Library
It is possible to compile pure C++ programs that make use of the core C++ library (ttnpp).  Example C++ programs are provided in the [src](src/) directory.

### Compile Instructions
This code requires cmake version 3.11 in order to compile. From the pyTTN base directory (${pyTTN_ROOT_DIR}) run:
```console
mkdir build
cd build
cmake-DBUILD_PYTHON_BINDINGS=OFF -DBUILD_SRC=ON ../ 
make
make install
```

This will build all .cpp files in the [src](src/) folder.  Typical installation times are $\lesssim$ 2 minutes.

-------------------------------------------------------------------------------

# Software Overview

## Objectives (Functional Requirements + User Requirements)

This package is intended to provide an easy-to-use and scalable method for simulating ground state and dynamics properties of quantum states relevant to materials science and quantum computing applications through the use of tree tensor network (and related multi-set) ansatz for representing the quantum state.  In order to address this aim the following key features were identified as requirements for the package:
1. The package provides the required functionality to handle tree tensor network states, providing several key algorithms for creating, accessing properties of, and editting such states, that enable the implementation of more advanced algorithms on top of these data structures.  
2. The package provides implementations of several key algorithms including the Density Matrix Renormalisation Group (DMRG) and Time-Dependent Variational Principle (TDVP) based time evolution that enable the preparation of ground states and time evolution of states, two key aspects of the pipeline required for application of these methods to representing quantum states of interest in materials science applications.
3. The package provides an easy-to-use interface for constructing generic Hamiltonian operators required for defining a new physical problem that provides a straightforward process for defining standard problems but also allows for easy extension to more general problems.
4. The package provides a python interface that simplifies the process of setting up new problems that avoids the need for direct interaction with the underlying C++ library, except in the case of significant extensions of the library beyond the scope outlined above.
5. In the instance of significant extensions of the library, the C++ interface is developed in a modular way enabling easy extension of core algorithms and data structures.

In addition to the core features developed above, which will be kept stable following release,  a key focus of this package is in supporting the research activities within the Quantum Software and Modelling team, this software is developed iteratively based on continuously adapted functionality requirements. This software contributes to WPs 1 and 3 of the EPSRC Digital Twins project which focus on the development and use of scalable classical methods for use as quantum digital twins, and the application of the resultant methods to develop hybrid classical and quantum approaches relevant to the simulation of quantum impurity models relevant to materials systems.

## General Software Layout (Functional Requirements)

The core data structures and algorithms for creating, accessing properties of, and editting Tree Tensor Network States are implemented in C++ with Python wrappers provided through the use of the [Pybind11](https://github.com/pybind/pybind11) library.  The C++ library (ttnpp) is a header only, template library that provides an extensible framework for constructing new tree tensor network data structures and algorithms acting on these data structures. The source code for this library can be found in [include/](include/) directory.  Example usage of this library within C++ are provided in the [src/](src/) directory.  The python wrapper is provided in the [python](python/) file and provides wrappers of all core functionality provided by the C++ library with explicit instantiation of all template classes with the use of a double precision floating point numbers as default.  Finally the [pyttn](pyttn/) folder provides additional Python interfaces to the underlying library, simplifying use of the overall package, and provides additional Python only functionality helpful in setting up specific types of models.

## Implementation Details (Code Design)

The core Tree Tensor Network Library (ttnpp) has been developed in C++ with pyTTN using a pybind11 based python wrapper to expose the core functionality to python.  The core ttnpp library is a header only C++ library and it is straightforward to compile pure C++ programs using this library.  For details see [TTNPP Library](#ttnpp-library-compilation).

-------------------------------------------------------------------------------

# Code Development Practices

The software will be developed iteratively with testing to progressively improve efficiency, build upon implementated functionality, and to ensure stability of the developed tools.

The software will be developed to NPL Software Integrity Level 3.

To adhere to the necessary requirements, please follow the following software development practices:

## Git

This software is developed based on standard git workflows (clearly documenting the version history and development cycles of the software).  This means that the code and repository are used as the main point of entry to track the development of the software and document the requirements.  Changes/updates to the codebase should be documented via atomic git commits with explanatory git commit messages.  If this refers to a bugfix, this should be explicitly noted in the git commit messages.  Changes to functional requirements (including extended functionality) may be documented via issues, also allowing for the reporting of bugs.

## Documentation

This software is documented directly within the code through easy-to-follow docstrings for the different functional units of the code (in particular classes and functions).  When implementing new/changing  functionality, please add/update the docstrings accordingly and add further comments if necessary to follow the implementation.  Additionally, this software makes of the [Sphinx](https://www.sphinx-doc.org/en/master) for generation of user documentation.  In addition to providing a complete collection of the documentation for all functional units of the code, this documentation provides a central location for additional IPython notebook tutorial files describing usage of the code with explanation of the functionality and expected outcomes.  When implementing new functionality please consider whether it would be appropriate to add additional tutorials documenting the usage of this new functionality.

## Coding Style

Please adhere to standard python coding conventions, [PEP-8](https://peps.python.org/pep-0008/).  The use of a linter (e.g. [Ruff](https://docs.astral.sh/ruff/)), and a formatter (e.g. [black](https://github.com/psf/black)) are recommended to ensure consistency.

## Testing (including validation and verification)
To ensure continuous testing of the functional units of the software separate tests are added to the ``\tests`` subfolder.  When adding/updating functionality, please ensure the functional units are tested appropriately and corresponing tests are added to the subfolders.  Currently, this software makes use of CI hooks to enable automated testing of components, however, we recommend that all developers should run tests locally.

To test the overall functionality, we have additionally provided a set of example applications enabling the validation of core features of the pyTTN library
- [Non-adiabatic dynamics of 24-mode pyrazine](examples/pyrazine/)
- [Exciton dynamics in a $n$-oligothiophene donor-C<sub>60</sub> fullerene acceptor system](examples/p3ht_pcbm_heterojunction/)
- Dynamics of quantum systems coupled to a [bosonic](examples/spin_boson_model/) or [fermionic](examples/anderson_impurity_model) environment
- Interacting [chains](examples/dissipative_spin_models/chain/) and [trees](examples/dissipative_spin_models/cayley_tree/) of open quantum systems

The expected results obtained from running these examples are presented in the [pyTTN paper](), and all datasets producing these results are available in the [data repository]().

## Code Review

Code review will be performed regularly by members of the team and will be led by the lead developer.  Code reviews are performed via GitLab pull requests.  Previous code reviews can be found via.

-------------------------------------------------------------------------------

## Links

<!-- Add links when complete -->
* National Physical Laboratory: <https://www.npl.co.uk/>
* Gitlab:         <https://gitlab.npl.co.uk/quantum-software/pyttn>
* Documentation:  <>
* PyPI:           <>
