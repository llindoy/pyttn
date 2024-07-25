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
        .value("fermion_mode", mode_type::FERMION_MODE)
        .value("boson_mode", mode_type::BOSON_MODE)
        .value("spin_mode", mode_type::SPIN_MODE)
        .value("qubit_mode", mode_type::QUBIT_MODE)
        .value("generic_mode", mode_type::GENERIC_MODE)
        .def("__str__", [](const mode_type& o){std::stringstream oss;   oss << o; return oss.str();});

    py::class_<mode_data>(m, "mode_data")
        .def(py::init())
        .def(py::init<size_t>())
        .def(py::init<size_t, mode_type>())
        .def(py::init<const mode_data&>())
        .def("assign", [](mode_data& self, const mode_data& o){self=o;})
        .def("__copy__",[](const mode_data& o){return mode_data(o);})
        .def("__deepcopy__", [](const mode_data& o, py::dict){return mode_data(o);}, py::arg("memo"))
        .def("fermionic", &mode_data::fermionic)
        .def("__str__", [](const mode_data& o){std::stringstream oss;   oss << o; return oss.str();})
        .def_property
            (
                "type", 
                static_cast<const mode_type& (mode_data::*)() const>(&mode_data::type),
                [](mode_data& o,  mode_type i){o.type() = i;}
            )
        .def_property
          (
                "lhd", 
                static_cast<const size_t& (mode_data::*)() const>(&mode_data::lhd),
                [](mode_data& o,  size_t i){o.lhd() = i;}
            );

    m.def("fermion_mode", &fermion_mode);
    m.def("boson_mode", &boson_mode);
    m.def("qubit_mode", &qubit_mode);
    m.def("spin_mode", &spin_mode);
    m.def("generic_mode", &generic_mode);

    py::class_<system_modes>(m, "system_modes")
        .def(py::init())
        .def(py::init<size_t>())
        .def(py::init<size_t, size_t>())
        .def(py::init<const system_modes&>())
        .def(py::init<const std::vector<mode_data>&>())
        .def("assign", [](system_modes& self, const system_modes& o){self=o;})
        .def("__copy__",[](const system_modes& o){return system_modes(o);})
        .def("__deepcopy__", [](const system_modes& o, py::dict){return system_modes(o);}, py::arg("memo"))
        .def("nmodes", &system_modes::nmodes)
        .def("resize", &system_modes::resize)
        .def("clear", &system_modes::clear)
        .def("__str__", [](const system_modes& o){std::stringstream oss;   oss << o; return oss.str();})
        .def(
                "__getitem__", 
                static_cast<const mode_data& (system_modes::*)(size_t) const>(&system_modes::operator[])
            )
        .def(
                "__setitem__", 
                [](system_modes& o, size_t j, const mode_data& i){o[j]=i;}
            )       
        .def(
                "mode", 
                static_cast<const mode_data& (system_modes::*)(size_t) const>(&system_modes::mode), 
                py::return_value_policy::reference
            );
    
}

void initialise_system_info(py::module& m);

#endif

