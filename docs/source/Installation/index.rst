##################
Installation Guide
##################

Dependencies
============

The core C++ library (ttnpp) and the python wrapper (pyTTN) have the following key dependencies.  

C++ Dependencies
----------------

The core C++ library requires C++11 features.

* External Libraries: - `Pybind11 <https://github.com/pybind/pybind11>`__
* Python bindings - `BLAS <https://netlib.org/blas/>`__ linear algebra -
* `Lapack <https://netlib.org/lapack/>`__ linear algebra -
* `Catch2 <https://github.com/catchorg/Catch2>`__ C++ Unit Tests (Only required when running C++ test)

The cmake build system can make use of the

* `Pybind11 <https://github.com/pybind/pybind11>`__ and
* `Catch2 <https://github.com/catchorg/Catch2>`__ external libraries
  
located in directory ${pyTTN_ROOT_DIR}/external. If these libraries are not found in this location it will attempt to pull them from their respective Github repositories. For `BLAS <https://netlib.org/blas/>`__ and `Lapack <https://netlib.org/lapack/>`__ linear algebra, the cmake build script uses the standard find_lapack and find_blas calls to locate the libraries. When compiling with Clang or AppleClang this method searches for LLVM using the FindLLVM.cmake module that is included within CMake.

Python Dependencies Dependencies
--------------------------------

The core python wrapper version supports Python versions >=3.9.

Additional python dependencies introduced by the core functionality of the pyTTN wrapper are: 

* `scipy <https://scipy.org/>`__  
* `numpy <https://numpy.org/>`__
* `networkx <https://networkx.org/>`__ 

Additionally, some of the examples depend upon the python packages: -
* `h5py <https://www.h5py.org/>`__ 
* `numba <https://numba.pydata.org/>`__

Finally, full tree visualisation functionality provided by the
``visualise_tree`` function depends upon the packages -
* `matplotlib <https://matplotlib.org/>`__ -
* `pydot <https://github.com/pydot/pydot>`__ -
* `graphviz <https://graphviz.org/>`__

With the final two dependencies only required for use of improved tree plotting functionality, e.g. when using ``prog = "dot"``. In order to use this improved tree plotting functionality it is necessary to install the system graphviz in addition to the graphviz python package.

All python packages are installed automatically when installing using pip, however, it is necessary to manually install graphviz to enable this functionality. ## Installation You can install pyTTN using pip like this:

.. code:: console

   cd ${pyTTN_ROOT_DIR}
   python3 -m pip install .

Multithreaded Build
===================

By default, this will make use of a single threaded build for compiling the Pybind11 wrapper and can take a number of minutes to complete. It is recommended to make use of multi-threaded builds when compiling the Pybind11. This can be done by setting the environment variable ``PARALLEL_BUILD_TTNPP``, e.g.

.. code:: console

   export PARALLEL_BUILD_TTNPP=8

to allow for the use of 8 threads when compiling.



.. raw:: html

   <!-- 
    Selecting BLAS
    ~~~~~~~~~~~~~~

   ### Building with CUDA Support
   [!Note]
   Work in progress
   -->

The TTNPP Library
=================

It is possible to compile pure C++ programs that make use of the core C++ library (``ttnpp``). Example C++ programs are provided in the `src <src/>`__ directory.

Compile Instructions
--------------------

This code requires cmake version 3.11 in order to compile. From the pyTTN base directory (${pyTTN_ROOT_DIR}) run:

.. code:: console

   mkdir build
   cd build
   cmake-DBUILD_PYTHON_BINDINGS=OFF -DBUILD_SRC=ON ../ 
   make
   make install

This will build all .cpp files in the `src <src/>`__ folder. Typical installation times are :math:`\lesssim` 2 minutes.