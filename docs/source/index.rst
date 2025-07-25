#########################################################################################
pyTTN: An Open Source Toolbox for Quantum Dynamics Simulations Using Tree Tensor Networks
#########################################################################################

|ArXiv| |Python| |Contributor Covenant| |License|

Welcome to the pyTTN documentation.  pyTTN is a python library for performing calculations with Tree Tensor Network states.  
It is designed to make getting started with TTNs quick and easy.  

The library provides implementations of several algorithms for updating the properties of TTNs including: 

   - the Density Matrix Renormalisation Group (DMRG) algorithm applied to Tree structures for ground state evaluation 
   - the Time-Dependent Variational Principle (TDVP) based approach for performing time evolution. 

A key focus on the design of the library being was to provide an easy to use interface for setting up calculations 
involving new Hamiltonians and different tree structures.  Additionally, pyTTN provides direct support for using 
tensor network approaches to solve equation of motion based approaches for open quantum system dynamics.


.. image::  images/pyttn_schematic_light.svg
   :target: https://gitlab.npl.co.uk/qsm/pyttn/
   :align: center
   :class: only-light

.. image::  images/pyttn_schematic_dark.svg
   :target: https://gitlab.npl.co.uk/qsm/pyttn/
   :align: center
   :class: only-dark


Citing pyTTN
------------

If you publish working using pyTTN, please cite the paper

-  **[pyTTN]** L.P. Lindoy, D. Rodrigo-Albert, Y. Rath, I. Rungger
   *pyTTN: An Open Source Toolbox for Open and Closed System Quantum
   Dynamics Simulations Using Tree Tensor Networks*,
   `arXiv:2503.15460 <https://arxiv.org/abs/2503.15460>`__.

::
   @misc{Lindoy2025,
     title = {pyTTN: An Open Source Toolbox for Open and Closed System Quantum Dynamics Simulations Using Tree Tensor Network},
     author = {Lindoy, Lachlan P. and Rodrigo-Albert, Daniel. and Rath, Yannic and Rungger, Ivan},
     year = {2025},
     eprint = {2503.15460}, 
     primaryClass={quant-ph},
     archivePrefix={arXiv}, 
     url={https://arxiv.org/abs/2503.15460}
   }

Installation:
-------------
.. toctree::
   :maxdepth: 2

   Installation </Installation/index>

Tutorials:
----------
.. toctree::
   :maxdepth: 2

   Tutorials </Tutorials/index>

Examples:
---------
.. toctree::
   :maxdepth: 2

   Examples </Examples/index>

pyTTN API:
----------
.. toctree::
   :maxdepth: 1

   API Outline </pyttn/API Outline> 
   API Documents </pyttn/index>


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |ArXiv| image:: https://img.shields.io/badge/arXiv-2503.15460-red
   :target: https://arxiv.org/abs/2503.15460
.. |Python| image:: https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13-blue.svg
   :target: https://gitlab.npl.co.uk/qsm/pyttn
.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covena nt-v2.0%20adopted-ff69b4.svg
   :target:  https://gitlab.npl.co.uk/qsm/pyttn/-/blob/main/CODE_OF_CONDUCT.md
.. |License| image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
