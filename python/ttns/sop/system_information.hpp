#ifndef PYTHON_BINDING_SYSTEM_INFORMATION_HPP
#define PYTHON_BINDING_SYSTEM_INFORMATION_HPP

#include <ttns_lib/sop/sSOP.hpp>
#include <ttns_lib/sop/SOP.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

#include <utils/occupation_number_basis_indexing.hpp>


namespace py=pybind11;

inline void init_system_info(py::module &m)
{
    using namespace ttns;
    py::enum_<mode_type>(m, "mode_type", py::arithmetic(), "The type of a mode")
        .value("fermion_mode", mode_type::FERMION_MODE, "Flag for setting a mode to be a fermion")
        .value("boson_mode", mode_type::BOSON_MODE, "Flag for setting a mode to be a boson")
        .value("spin_mode", mode_type::SPIN_MODE, "Flag for setting a mode to be a spin degree of freedom")
        .value("tls_mode", mode_type::QUBIT_MODE, "Flag for setting a mode to be a two-level system degree of freedom")
        .value("generic_mode", mode_type::GENERIC_MODE, "Flag for setting a mode to be an unknown generic mode")
        .def("__str__", [](const mode_type& o){std::stringstream oss;   oss << o; return oss.str();});

    py::class_<primitive_mode_data>(m, "primitive_mode_data")
        .def(py::init(), "Constructs an empty primitive_mode_data object")
        .def(py::init<size_t>(), R"mydelim(
            Constructs a primitive_mode_data object with a generic_mode mode_type of a given dimensionality

            :Parameters:    - **d** (int) - Local Hilbert space dimension of mode
        )mydelim")

        .def(py::init<size_t, mode_type>(), R"mydelim(
            Constructs a primitive_mode_data object with a given mode_type of a given dimensionality

            :Parameters:    - **d** (int) - Local Hilbert space dimension of mode
                            - **type** (:class:`mode_type`) - The type of the mode
        )mydelim")
        .def(py::init<const primitive_mode_data&>(), R"mydelim(
            Copy constructs a primitive_mode_data object from another primitive_mode_data object

            :Parameters:    - **other** (:class:`primitive_mode_data`) - The other primitive mode data type
        )mydelim")
        .def("assign", [](primitive_mode_data& self, const primitive_mode_data& o){self=o;}, R"mydelim(
                Assign the value of the primitive_mode_data object from another primitive_mode_data object
        
                :param in: The input primitive mode data
                :type in: primitive_mode_data
        )mydelim")
        .def("__copy__",[](const primitive_mode_data& o){return primitive_mode_data(o);})
        .def("__deepcopy__", [](const primitive_mode_data& o, py::dict){return primitive_mode_data(o);}, py::arg("memo"))
        .def("fermionic", &primitive_mode_data::fermionic, R"mydelim(
            :returns: whether the mode is fermionic
            :rtype: bool
        )mydelim")
        .def("__str__", [](const primitive_mode_data& o){std::stringstream oss;   oss << o; return oss.str();})
        .def_property
            (
                "type", 
                static_cast<const mode_type& (primitive_mode_data::*)() const>(&primitive_mode_data::type),
                [](primitive_mode_data& o,  mode_type i){o.type() = i;},
                "The type of the mode. That is whether it is fermionic, bosonic, spin, tls, generic."
            )
        .def_property
          (
                "lhd", 
                static_cast<const size_t& (primitive_mode_data::*)() const>(&primitive_mode_data::lhd),
                [](primitive_mode_data& o,  size_t i){o.lhd() = i;}, R"mydelim(
                Return the local Hilbert space dimension of the composite mode.  This is the product of the local Hilbert space dimensions of the primitive modes

                :returns: The Local Hilbert Space Dimension
                :rtype: int
        )mydelim")
        .doc() = R"mydelim(
          A class for handling information about the primitive modes making up a system
        )mydelim";


    py::class_<mode_data>(m, "mode_data")
        .def(py::init(), "Default construct an empty mode data object.")
        .def(py::init<size_t>(), R"mydelim(
            Construct a mode data object consisting of a single generic primitive mode with a given local Hilbert space dimension

            :Parameters:    - **d** (int) - Local Hilbert space dimension
        )mydelim")
        .def(py::init<size_t, mode_type>(), R"mydelim(
            Construct a mode data object consisting of a single mode primitive mode with a given local Hilbert space dimension

            :Parameters:    - **d** (int) - Local Hilbert space dimension
                            - **type** (:class:`mode_type`) - The type of mode to construct
        )mydelim")

        .def(py::init<const mode_data&>(), R"mydelim(
            Construct a mode data object from another mode data object

            :Parameters:    - **data** (:class:`mode_data`) - The mode data object to construct from
        )mydelim")

        .def(py::init<const primitive_mode_data&>(), R"mydelim(
            Construct a mode data object consisting of a single primitive mode 

            :Parameters:    - **data** (:class:`primitive_mode_data`) - The value of the primitive mode to use
        )mydelim")

        .def(py::init<const std::vector<primitive_mode_data>&>(), R"mydelim(
            Construct a mode data object from a vector of primitive_mode_data objects 

            :Parameters:    - **data** (list[:class:`primitive_mode_data`]) - A list of primitive mode data objects that form the composite mode
        )mydelim")

        .def("assign", [](mode_data& self, const mode_data& o){self=o;}, R"mydelim(
            Assign the value of the mode data object from another mode data object

            :Parameters:    - **data** (:class:`mode_data`) - The mode data object used to assing this value
        )mydelim")
        .def("assign", [](mode_data& self, const primitive_mode_data& o){self=o;}, R"mydelim(
            Assign the value of the mode data object so that it contains a single primitive mode data object

            :Parameters:    - **data** (:class:`primitive_mode_data`) - The primitive mode data object used to assing this value
        )mydelim")

        .def("assign", [](mode_data& self, const std::vector<primitive_mode_data>& o){self=o;}, R"mydelim(
            Assign the value of the mode data object so that it contains a list single primitive mode data object

            :Parameters:    - **data** (list[:class:`primitive_mode_data`]) - The list of primitive_mode_data objects used to set the value of this object.
        )mydelim")

        .def("append", static_cast<void (mode_data::*)(const primitive_mode_data&)>(&mode_data::append), R"mydelim(
            Append a primitive mode to the current mode

            :Parameters:    - **data** (:class:`primitive_mode_data`) - The primitive mode information to append to this object
        )mydelim")

        .def("append", static_cast<void (mode_data::*)(const mode_data&)>(&mode_data::append), R"mydelim(
            Append all primitive modes contained in another mode data object to the current mode

            :Parameters:    - **data** (:class:`mode_data`) - The mode data object containing a list of primitive modes to append to this object
        )mydelim")

        .def("__copy__",[](const mode_data& o){return mode_data(o);})
        .def("__deepcopy__", [](const mode_data& o, py::dict){return mode_data(o);}, py::arg("memo"))
        .def("__str__", [](const mode_data& o){std::stringstream oss;   oss << o; return oss.str();})
        .def("liouville_space", &mode_data::liouville_space, R"mydelim(
            Constructs a new mode_data object that corresponds to a Liouville space representation of this object.  This is done
            by taking each mode and appending a dual space operator following this mode of the same type. 

            :returns: Liouville space mode_data object
            :rtype: mode_data
        )mydelim")

        .def("contains_fermion", &mode_data::contains_fermion, R"mydelim(
            :returns: whether a mode is fermionic
            :rtype: bool
        )mydelim")
        .def("lhd", static_cast<size_t (mode_data::*)() const>(&mode_data::lhd), R"mydelim(
            Return the local Hilbert space dimension of the composite mode.  This is the product of the local Hilbert space dimensions of the primitive modes

            :returns:  The Local Hilbert Space Dimension
            :rtype: int
        )mydelim")
        .doc() = R"mydelim(
          A class for handling information about composite modes used to represent the system information. 
          This class stores a set of primitive_mode_data defining each individual mode that forms this composite,
          and provides several helper functions for determining the properties of the underlying modes.  This
          functions support several overloaded constructors.
        )mydelim";

    m.def("fermion_mode", &fermion_mode, R"mydelim(
      Create a new mode_data object for a mode with Fermionic Exchange Statistics

      :returns: fermionic mode data object
      :rtype: mode_data
      )mydelim");
    m.def("boson_mode", &boson_mode, R"mydelim(
      Create a new mode_data object for a mode with Fermionic Exchange Statistics

      :param N: The local Hilbert space dimension to use for the mode
      :type N: int

      :returns: bosonic mode data object
      :rtype: mode_data
      )mydelim");

    m.def("qubit_mode", &qubit_mode, R"mydelim(
      Create a new mode_data object for a qubit.  That is a two level system that Pauli operators can act upon

      :returns: qubit mode data object
      :rtype: mode_data
      )mydelim");
    m.def("tls_mode", &qubit_mode, R"mydelim(
      Create a new mode_data object for a qubit.  That is a two level system that Pauli operators can act upon

      :returns: qubit mode data object
      :rtype: mode_data
      )mydelim");
    m.def("spin_mode", &spin_mode, R"mydelim(
      Create a new mode_data object for a spin-(N-1)/2 

      :param N: The local Hilbert space dimension to use for the mode
      :type N: int

      :returns: spin mode data object
      :rtype: mode_data
      )mydelim");
    m.def("generic_mode", &generic_mode, R"mydelim(
      Create a new genric mode mode_data object with arbitrary Hilbert space dimension.  When using such a mode it is necessary
      for the user to define all operators acting on this mode.

      :param N: The local Hilbert space dimension to use for the mode
      :type N: int

      :returns: generic mode data object
      :rtype: mode_data
      )mydelim");

    py::class_<system_modes>(m, "system_modes")
        .def(py::init(), "Construct an empty system_modes object.")
        .def(py::init<size_t>(), R"mydelim(
            Construct a system_modes object containing N modes

            :Parameters:    - **N** (int) - The number of modes defining the system
        )mydelim")

        .def(py::init<size_t, size_t>(), R"mydelim(
            Construct a system_modes object containing N modes, where each mode is a generic mode of sizes d

            :Parameters:    - **N** (int) - The number of modes defining the system
                            - **d** (int) - The local Hilbert space dimension of the modes
        )mydelim")

        .def(py::init<size_t, size_t, const std::vector<size_t>&>(), R"mydelim(
            Construct a system_modes object containing N modes, where each mode is a generic mode of sizes d.  Additionally
            provide a user defined ordering of the modes

            :Parameters:    - **N** (int) - The number of modes defining the system
                            - **d** (int) - The local Hilbert space dimension of the modes
                            - **ordering** (list[int]) - The ordering of modes
        )mydelim")
        .def(py::init<const system_modes&>(), R"mydelim(
            Copy construct the system mode from another system mode

            :Parameters:    - **other** (:class:`system_modes`) - The other system_modes object
        )mydelim")
        .def(py::init<const primitive_mode_data&>(), R"mydelim(
            Construt a system containing one primitive mode 

            :Parameters:    - **mode** (:class:`primitive_mode_data`) - The single primitive mode information
        )mydelim")
        .def(py::init<const mode_data&>(), R"mydelim(
            Construct a system containing one composite mode 

            :Parameters:    - **mode** (:class:`mode_data`) - The single composite mode information
        )mydelim")
        .def(py::init<const std::vector<mode_data>&>(), R"mydelim(
            Construct a system from a vector of modes 

            :Parameters:    - **mode** (list[:class:`mode_data`]) - The list of composite mode information
        )mydelim")
        .def(py::init<const std::vector<mode_data>&, const std::vector<size_t>&>(), R"mydelim(
            Construct a system from a vector of modes and user defined mode ordering

            :Parameters:    - **mode** (list[:class:`mode_data`]) - The list of composite mode information
                            - **ordering** (list[int]) - The ordering of modes
        )mydelim")
        .def("assign", [](system_modes& self, const system_modes& o){self=o;}, R"mydelim(
            Assign the value of the system_modes object from another system mode

            :Parameters:    - **other** (:class:`system_modes`) - The other system_modes object
        )mydelim")
        .def("__copy__",[](const system_modes& o){return system_modes(o);})
        .def("__deepcopy__", [](const system_modes& o, py::dict){return system_modes(o);}, py::arg("memo"))
        .def("nmodes", &system_modes::nmodes, R"mydelim(
            :returns: The number of composite modes in the system
            :rtype: int
        )mydelim")
        .def("nprimitive_modes", &system_modes::nprimitive_modes, R"mydelim(
            :returns: The number of primitive modes in the system
            :rtype: int
        )mydelim")
        .def("resize", &system_modes::resize, R"mydelim(
            Resize the internal buffers so that the system_modes object can contain N modes

            :param N: The number of composite modes the system is allowed to contain
            :type N: int
        )mydelim")
        .def("clear", &system_modes::clear, "Deallocate all internal storage of the system modes object")
        .def("liouville_space", &system_modes::liouville_space, R"mydelim(
            Constructs a new system_modes object that corresponds to a Liouville space representation of this object.  This is done
            by taking each mode and appending a dual space operator following this mode of the same type. 

            :returns: Liouville space system_modes object
            :rtype: system_modes
        )mydelim")


        .def("append", &system_modes::append_system, R"mydelim(
            Append the set of modes contained in the system_modes object to the back of this system.  This function preserves the ordering
            of the modes in the two system objects but shifts the index in the new modes so that they occur after the current modes

            :param sys: The new system modes to append to the end of this object
            :type sys: system_modes
        )mydelim")

        .def("contains_fermion", &system_modes::contains_fermion, R"mydelim(
            :returns: whether a mode is fermionic
            :rtype: bool
        )mydelim")


        .def("add_mode", static_cast<void (system_modes::*)(const primitive_mode_data&)>(&system_modes::add_mode), R"mydelim(
            Append a new mode to the end of the system constructed from a single primitive_mode_data

            :Parameters:    - **mode** (:class:`primitive_mode_data`) - The mode to append
        )mydelim")

        .def("add_mode", static_cast<void (system_modes::*)(const primitive_mode_data&, size_t)>(&system_modes::add_mode), R"mydelim(
            Append a new mode to the end of the system constructed from a single primitive_mode_data.  Providing and integer giving the index of this new mode in the system

            :Parameters:    - **mode** (:class:`primitive_mode_data`) - The mode to append
                            - **index** (int) - The index of this mode
        )mydelim")


        .def("add_mode", static_cast<void (system_modes::*)(const mode_data&)>(&system_modes::add_mode), R"mydelim(
            Append a new mode to the end of the system specified in a mode_data object.

            :Parameters:    - **mode** (:class:`mode_data`) - The mode to append
        )mydelim")
        .def("add_mode", static_cast<void (system_modes::*)(const mode_data&, size_t)>(&system_modes::add_mode), R"mydelim(
            Append a new mode to the end of the system specified in a mode_data object.  Providing and integer giving the index of this new mode in the system

            :Parameters:    - **mode** (:class:`mode_data`) - The mode to append
                            - **index** (int) - The index of this mode
        )mydelim")
        .def("__str__", [](const system_modes& o){std::stringstream oss;   oss << o; return oss.str();})
        .def("__len__", &system_modes::nmodes, R"mydelim(
            :returns: The number of composite modes in the system
            :rtype: int
        )mydelim") 
        .def(
                "__getitem__", 
                static_cast<mode_data& (system_modes::*)(size_t)>(&system_modes::operator[]),
                py::return_value_policy::reference, 
        R"mydelim(
                :Parameters:    - **i** (int) - The index of the mode to access

                :Returns:       The mode_data object in location i
                :Return Type:   :class:`mode_data`
        )mydelim"
            )
        .def(
                "__setitem__", 
                [](system_modes& o, size_t j, const mode_data& i){o[j]=i;}, 
        R"mydelim(
                Set the value of the mode data object at position i to the mode_data object specified by mode
                :Parameters:    - **i** (int) - The index of the mode to access
                                - **mode** (:class:`mode_data`) - The new value to set this mode to

        )mydelim"
            )       
        .def(
                "__setitem__", 
                [](system_modes& o, size_t j, const primitive_mode_data& i){o[j]=i;},
        R"mydelim(
                Set the value of the mode data object at position i to the mode_data object containing a single primitive mode defined by the variable mode
                :Parameters:    - **i** (int) - The index of the mode to access
                                - **mode** (:class:`primitive_mode_data`) - The primitive_mode_data object used to construct the new mode

        )mydelim"
            )       
        .def(
                "__setitem__", 
                [](system_modes& o, size_t j, const std::vector<primitive_mode_data>& i){o[j]=i;},
        R"mydelim(
                Set the value of the mode data object at position i to the mode_data object containing a list of primitive mode defined by the variable mode
                :Parameters:    - **i** (int) - The index of the mode to access
                                - **mode** (list[:class:`primitive_mode_data`]) - A list primitive_mode_data object used to construct the new mode

        )mydelim"
            )       
        .def(
                "mode", 
                static_cast<const mode_data& (system_modes::*)(size_t) const>(&system_modes::mode), 
                py::return_value_policy::reference, 
        R"mydelim(
                Returns a reference to the mode_data object at position i

                :Parameters:    - **i** (int) - The index of the mode to access

                :Returns:       A reference to the mode_data object in location i
                :Return Type:   :class:`mode_data`
        )mydelim"
            )
        .def(
                "primitive_mode", 
                static_cast<const primitive_mode_data& (system_modes::*)(size_t) const>(&system_modes::primitive_mode), 
                py::return_value_policy::reference, 
        R"mydelim(
                Returns a reference to the ith primitive_mode_data object in the system

                :Parameters:    - **i** (int) - The index of the mode to access

                :Returns:       A reference to the primitive_mode_data object in location i
                :Return Type:   :class:`primitive_mode_data`
        )mydelim"
            )
        .def(
                "set_primitive_mode", 
                [](system_modes& o, size_t i, const primitive_mode_data& d)
                {
                    o.primitive_mode(i) = d;
                },
        R"mydelim(
                Set the value of the ith primitive mode to the mode_data in the system.  This does not necessarily correspond to a primitive mode in the ith composite mode
                :Parameters:    - **i** (int) - The index of the primitive mode to access
                                - **mode** (list[:class:`primitive_mode_data`]) - A list primitive_mode_data object used to construct the new mode

        )mydelim"
            )
        .def("as_combined_mode", &system_modes::as_combined_mode, R"mydelim(
            Constructs a single combined mode_data object from the system_modes object

            :returns: The composite mode containing all primitive modes in this system
            :rtype: mode_data
        )mydelim")
        .def("mode_index", [](const system_modes& o, size_t i){return o.mode_index(i);})
        .def_property(
                "mode_indices", 
                &system_modes::mode_indices,
                &system_modes::set_mode_indices,
                "The user defined mode ordering for each of the modes in the system"
            )
        .doc() = R"mydelim(
          A class for handling the specification of all modes in the system.  This stores a list of each composite mode and an
          optional user supplied ordering of the modes
        )mydelim";


    m.def("combine_systems", &combine_systems, R"mydelim(
      A function for creating a new system_modes object by composing two system_modes objects together.  This
      function appends the second system_modes objects modes to the first and additionally, where user defined
      mode orderings have been specified, shifts the ordering of the modes correspond to the second object. By 
      the number of modes in the first object.

      :param sysA: The first system mode variable
      :type sysA: system_modes
      :param sysB: The second system mode variable
      :type sysB: system_modes

      :returns: The combined system_modes object.
      :rtype: system_modes

        )mydelim");
    
}

void initialise_system_info(py::module& m);

#endif

