# PyTTN

A Tree Tensor Network Software Package written in C++ with a Python Wrapper

## Disclaimer


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
This code requires cmake version 3.11 in order to compile. From PyTTN  base directory (${PyTTN_ROOT_DIR}) run:
```console
mkdir build
cd build
cmake ../
make
make install
```
to build the C++ executables
```console
mkdir build
cd build
cmake -DBUILD_PYTHON_BINDINGS=ON -DBUILD_SRC=OFF ../
make
make install
```
to build the python bindings and install the library in the pyttn folder.


This code has been successfully tested on: 
* Linux Mint 21 Cinnamon with Kernel Version 5.15.0-50-generic using g++11.2.0 with OpenBLAS and clang-14.0.0-1ubuntu with the current system versions of Lapack and Blas
* CentOS release 6.6 with Kernel Version 2.6.32-504.16.2.el6.x86_64 using g++-10.1.0 and with MKL/17.0

Typical installation times are $\lesssim$ 2 minutes.

## Running the Software
Example python scripts for running the software are provided in ${PyTTN_ROOT_DIR}/examples

