## Contributing

[code-of-conduct]: CODE_OF_CONDUCT.md

Contributions of all kinds are welcome.  Please get in touch if you have any suggestions, comments or questions regarding the code or documentaiton. Unfortunately, we are unable to provide direct access to Issues and Merge Requests on the NPL Gitlab. As such, please feel free to reach out to [Lachlan Lindoy](mailto:lachlan.lindoy@npl.co.uk).  

Please note that this project is released with a [Contributor Code of Conduct][code-of-conduct]. By participating in this project you agree to abide by its terms.

## How To Contribute

If you have suggestions for how this project could be improved, or want to report a bug, please feel free to contact the development team.


## General Guidelines
### Code Development Practices

The software will be developed iteratively with testing to progressively improve efficiency, build upon implementated functionality, and to ensure stability of the developed tools.

The software will be developed to NPL Software Integrity Level 3.

To adhere to the necessary requirements, please follow the following software development practices:

### Git

This software is developed based on standard git workflows (clearly documenting the version history and development cycles of the software).  This means that the code and repository are used as the main point of entry to track the development of the software and document the requirements.  Changes/updates to the codebase should be documented via atomic git commits with explanatory git commit messages.  If this refers to a bugfix, this should be explicitly noted in the git commit messages.  Changes to functional requirements (including extended functionality) may be documented via issues, also allowing for the reporting of bugs.

### Documentation

This software is documented directly within the code through easy-to-follow docstrings for the different functional units of the code (in particular classes and functions).  When implementing new/changing  functionality, please add/update the docstrings accordingly and add further comments if necessary to follow the implementation.  Additionally, this software makes of the [Sphinx](https://www.sphinx-doc.org/en/master) for generation of user documentation.  In addition to providing a complete collection of the documentation for all functional units of the code, this documentation provides a central location for additional IPython notebook tutorial files describing usage of the code with explanation of the functionality and expected outcomes.  When implementing new functionality please consider whether it would be appropriate to add additional tutorials documenting the usage of this new functionality.

### Coding Style

Please adhere to standard python coding conventions, [PEP-8](https://peps.python.org/pep-0008/).  The use of a linter (e.g. [Ruff](https://docs.astral.sh/ruff/)), and a formatter (e.g. [black](https://github.com/psf/black)) are recommended to ensure consistency.

### Testing (including validation and verification)
To ensure continuous testing of the functional units of the software separate tests are added to the ``\tests`` subfolder.  When adding/updating functionality, please ensure the functional units are tested appropriately and corresponing tests are added to the subfolders.  Currently, this software makes use of CI hooks to enable automated testing of components, however, we recommend that all developers should run tests locally.

To test the overall functionality, we have additionally provided a set of example applications enabling the validation of core features of the pyTTN library
- [Non-adiabatic dynamics of 24-mode pyrazine](examples/pyrazine/)
- [Exciton dynamics in a $n$-oligothiophene donor-C<sub>60</sub> fullerene acceptor system](examples/p3ht_pcbm_heterojunction/)
- Dynamics of quantum systems coupled to a [bosonic](examples/spin_boson_model/) or [fermionic](examples/anderson_impurity_model) environment
- Interacting [chains](examples/dissipative_spin_models/chain/) and [trees](examples/dissipative_spin_models/cayley_tree/) of open quantum systems

The results generated through these example scripts have, where possible, been verified against literature results. The iterature results may be viewed by following the appropriate references in both the example scripts and the [pyTTN paper](). The expected results obtained from running these examples are presented in the [pyTTN paper](), and all datasets producing these results are available in the [data repository]().

### Code Review

Code review will be performed regularly by members of the team and will be led by the lead developer.  Code reviews are performed via GitLab pull requests.  Previous code reviews can be found via the [gitlab interface](https://gitlab.npl.co.uk/quantum-software/pyttn/-/merge_requests?scope=all&state=merged).

-------------------------------------------------------------------------------

