pyTTN package
=============

The pyTTN package: An Open Source Toolbox for Quantum Dynamics Simulations Using Tree Tensor Networks.


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

The core Tree Tensor Network Library (ttnpp) has been developed in C++ with pyTTN using [Pybind11](https://github.com/pybind/pybind11) to create a python wrapper to expose the core functionality to python.  The core ttnpp library is a header only C++ library and it is straightforward to compile pure C++ programs using this library, for details see [`ttnpp` Library](#ttnpp-library-compilation).

The ttnpp library contains several key components in order to implement the require functionality.  These are
- The ``ntree`` template class that is used for the specification of tree topologies
- The ``system_modes`` object defining the types of physical degrees of freedom in the system of interest.
- The ``SOP`` class providing an easy-to-use interface for representing generic Hamiltonian operators
- The ``ttn_base`` template class that acts as a base class for creation of explicit tree tensor network objects
- The ``sop_operator`` class providing efficient representation of generic operators for use within the various updating schemes used for Tree Tensor Network States
- The ``sweeping_algorithm`` template class that accepts as template parameters functions implementing core functionality required for implementing sweeping algorithms on tree tensor networks
- The ``matrix_element`` class that provides efficient evaluation of generic expectation values of operators using Tree Tensor Network States

A typical calculation will use implementations of each of these base objects as follows, we note that unless a specific dependency is specified the order of object creation can be interchanged:
- An ``ntree`` object defining the tree structure needs to be constructed
- A ``system_modes`` object is created defining the types of modes present in the system of interest.
- A ``SOP`` object needs to be constructed representing the Hamiltonian for the system of interest
- A Tree Tensor Network Object (a specific implementation of the ``ttn_base`` CRTP base class) will then be created an instantiated from the ``ntree`` object.
- Given the ``SOP`` and Tree Tensor Network Object (a specific implementation of the ``ttn_base`` CRTP base class) we construct a ``sop_operator`` object specifying a representation of the Hamiltonian that can be efficiently applied to the Tree Tensor Network.
- A specific instance of the ``sweeping_algorithm`` is created using the instance of the ``ttn_base`` and the ``sop_operator`` object.
- A ``matrix_element`` object is created using the instance of the ``ttn_base`` object.
- The instance of the ``sweeping_algorithm`` is repeatedly applied to the instance of the ``ttn_base`` object and properties are measured using the ``matrix_element`` object.

Below we discuss each component and additional tooling associated with it in more detail.



### The ``ntree`` class

Within pyTTN the specification of tree topologies is controlled by the ``ntree`` class.  This class represents a general tree where each node (an ``ntreeNode`` object) stores a single data value and has an arbitrary number of child nodes.  This class provides functions for accessing nodes in the tree, inserting nodes at various positions, and traversing the nodes in various objects.  When combined with the ``ntreeBuilder`` class, which provides several helper functions for constructing trees with certain structures, the `ntree` class provides a convenient interface for defining connectivities of tree tensor networks, and additionally defining the bond dimension associated with the bonds connecting the nodes. For further details on the use of ``ntree`` objects within the pyTTN Python interface please see the tutorial on [tree topologies](tutorials/tree_topologies.ipynb).


### The ``system_modes`` class

The ``system_modes`` class provides an interface for defining the types of physical degrees of freedom and their local Hilbert space dimensions.  Within the `ttnpp` library several types of degree of freedom are supported these are
- `fermion_mode` for fermionic degrees of freedom with a local Hilbert space dimension of 2
- `boson_mode` for bosonic degrees of freedom that allow for a user defined truncation of the local Hilbert space dimension
- `spin_mode` for a spin degree of freedom with a user defined $S$
- `tls_mode` for a two level system degree of freedom
- `generic_mode` for a mode that is not of any of the above and allows for a user defined truncation of the local Hilbert space dimension.

For the first four of these types a set of predefined dictionaries that map string representations of operators to operators are provided.  For `generic_mode`s the user must define an `operator_dictionary` object that maps string labels to operators.  For further details on the use of these tools within the python interface see the tutorial on [operator generation](tutorials/operator_generation.ipynb).

### The ``SOP`` class

The ``SOP`` class is the primary tool used for user definitions of Hamiltonians for use within the core pyTTN algorithms.  This class provides a compact representation of a generic operator represented as a Sum-Of-Product of individual site operators, which here are each represented by a string labelling the term This class depends on several other classes:
- The `coeff` class containing a potentially time-dependent scalar
- The `sOP` class defining a string operator acting on a single physical degree of freedom
- The `sPOP` class a product of `sOP` objects.
- The `sNBO` class the product of a `sPOP` object with a `coeff` object.
- The `sSOP` class containing a sum-of-product operator representation, that is easier to apply operations to than the `SOP` class but is less compact.

Within the `ttnpp` library (and consequently pyTTN as a whole) it is possible to add and multiply arbitrary combinations of these 5 classes to construct a new operator objects.  Adding these to a `SOP` class provides a compact representation of the sum-of-product operator that can then be used to construct `sop_operator` objects for use within calculations.  An analogous class exists for defining Hamiltonian objects that are to be applied to multiset Tree Tensor Networks, namely the `multiset_SOP`.  For further details on the generation of operators using the python interface to the `SOP` class see the tutorial on [operator generation](tutorials/operator_generation.ipynb).

### The ``ttn_base`` class

The ``ttn_base`` class is a base class for representing Tree Tensor Network objects that allows for easy extension of the information stored at each site tensor in the network.  This class provides an implementation of many of the core functions needed for a useful implementation of tree tensor network states but leaves implementation of operations acting on individual site tensors to the `node_type` object.

Within the `ttnpp` library two classes deriving from ``ttn_base`` are provided namely:
- The ``ttn`` class providing a general implementation of Tree Tensor Network ansatz that makes use of a `ttn_node` as its `node_type`
- The `ms_ttn` class providing an implementation of the multiset Tree Tensor Network ansatz that makes use of a `ms_ttn_node` as its `node_type`

### The ``sop_operator`` class

Provides an efficient representation of a generic operator for use within the various updating schemes used for Tree Tensor Network states.  This object optimise the representation of a `SOP` object given a `ttn` to provide a reduced bond dimension representation of the `SOP`.  This is done using the bipartite graph decomposition approach that is implemented within the `autoSOP` class.  An analogous class exists for handling multiset Hamiltonian objects, namely the `multiset_sop_operator` class.

### The ``sweeping_algorithm`` class

The ``sweeping_algorithm`` class provides the core functionality for classes that sweep over a Tree Tensor Network and update individual site tensors, and through template specialisation provides a convenient tool for defining various common algorithms for updating a TTN.  This class takes four template parameters:
1. The template class representing the TTN object the specific instantiation of the class can be applied to.
2. A callable template class that implements the action to be applied to each site tensor in the TTN.
3. A template class for storing the environment tensors of the system, this allows for substitution of the sum-of-product form used as standard with alternative Hamiltonian representations.
4. A callable template class for updating the tensors across a bond when shifting the orthogonality between sites, allowing for expansion of bond dimension at this step.

Within the ``ttnpp`` library several specialisations of this base class are provided.  In particular, the single and adaptive single-site DMRG and TDVP algorithms for both TTNs and multiset TTNs are provided as specific instantiations of these classes.  

### The ``matrix_element`` class 
The final core class within the `ttnpp` package is the ``matrix_element`` class.  This class handles all temporary storage and logic required to efficiently evaluate expectation values involving tensor networks and sop_operators.  In particular, this class has been implemented to where possible (and in particular when there is an orthogonality centre within the Tensor Networks being considered) to exploit sparsity in the representation of operators to avoid evaluation of trivial contractions.  For the evaluation of expectation values of single-body interactions this is essential for efficiency, leading to asymptotic improvements in cost over direct contractions.  

For details on the use of this class within the pyTTN Python library please see the tutorial on [ttn operations](tutorials/ttn_operations.ipynb) that provides an overview of the evaluation of generic expectation values within pyTTN.

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
