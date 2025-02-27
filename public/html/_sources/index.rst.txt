.. pyTTN documentation master file, created by
   sphinx-quickstart on Mon Jan 20 15:24:52 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#########################################################################################
pyTTN: An Open Source Toolbox for Quantum Dynamics Simulations Using Tree Tensor Networks
#########################################################################################

Welcome to the pyTTN documentation.  pyTTN is a python library for performing calculations with Tree Tensor Network states.  
It is designed to make getting started with TTNs quick and easy.  

The library provides implementations of several algorithms for updating the properties of TTNs including: 

   - the Density Matrix Renormalisation Group (DMRG) algorithm applied to Tree structures for ground state evaluation 
   - and the Time-Dependent Variational Principle (TDVP) based approach for performing time evolution. 

A key focus on the design of the library being was to provide an easy to use interface for setting up calculations 
involving new Hamiltonians and different tree structures.  Additionally, pyTTN provides direct support for using 
tensor network approaches to solve equation of motion based approaches for open quantum system dynamics.


.. toctree::
   :maxdepth: 2
   :caption: Installation:

   Installation </Installation/index>


.. toctree::
   :maxdepth: 2
   :caption: Quickstart: 

   Quickstart </Quickstart/index>


.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   Tutorials </Tutorials/index>


.. toctree::
   :maxdepth: 1
   :caption: pyTTN API:

   API Outline </pyttn/API Outline> 
   API Documents </pyttn/index>


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
