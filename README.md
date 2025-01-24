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

#Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
- [Tutorials](#tutorials)


# About the Project

This open source project aims to provide an easy to use python interface for working with generic Tree Tensor Networks States to efficiently compute dynamics properties of quantum systems.  A key focus of this library is the easy setup of calculations employing either single or multiset tensor networks with generic tree structured connectivity.  Easy setup of Hamiltonians for arbitrary problems, with the ability to automatically apply techniques such as mode combination to reduce the total number of modes present in the system. Additionally, this library includes several tools to help facilitate applications of these approaches to study the dynamics of quantum systems that are strongly coupled to structured environment using both unitary methods (e.g. TEDOPA, T-TEDOPA and other representations of the system-bath Hamiltonian) as well as non-unitary approaches (e.g. Hierarchical Equations of Motion and Generalised Pseudomode method). 

pyTTN implements a range of numerically exact methods (methods that are systematically convergable to the exact results) for the dynamics of quantum system, and provides several example applications to
- Non-adiabatic dynamics of 24-mode pyrazine
- Exciton dynamics in a $n$-oligothiophene donor-C$_60$ fullerene acceptor system
- Dynamics of quantum systems coupled to a single or multiple bosonic or fermionic environment
- Interacting chains and trees of open quantum systems

-------------------------------------------------------------------------------


# Getting Started

##Prerequisites

You can install pyTTN using pip like this:
```
$ cd ${pyTTN_ROOT_DIR}
$ python3 -m pip install .
```

##Installation

# Tutorials










