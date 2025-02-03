#ifndef PYTHON_BINDING_SOP_MODELS_HPP
#define PYTHON_BINDING_SOP_MODELS_HPP

#include <ttns_lib/sop/models/model.hpp>
#include <ttns_lib/sop/models/aim.hpp>
#include <ttns_lib/sop/models/electronic_structure.hpp>
#include <ttns_lib/sop/models/spin_boson.hpp>
#include <ttns_lib/sop/models/tfim.hpp>

#include "../../../utils.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py=pybind11;

template <typename T>
void init_models(py::module &m, const std::string& label)
{
    using namespace ttns;
    using real_type = typename linalg::get_real_type<T>::type;


    using conv = linalg::pybuffer_converter<linalg::blas_backend>;

    //the base model type
    py::class_<model<T>>(m, (std::string("model_")+label).c_str())
        .def("hamiltonian", static_cast<SOP<T> (model<T>::*)(real_type)>(&model<T>::hamiltonian), py::arg("tol") = 1e-14)
        .def("hamiltonian", static_cast<void (model<T>::*)(sSOP<T>&, real_type)>(&model<T>::hamiltonian), py::arg(), py::arg("tol") = 1e-14)
        .def("hamiltonian", static_cast<void (model<T>::*)(SOP<T>&, real_type)>(&model<T>::hamiltonian), py::arg(), py::arg("tol") = 1e-14)
        .def("system_info", static_cast<void (model<T>::*)(system_modes&)>(&model<T>::system_info))
        .def("system_info", static_cast<system_modes (model<T>::*)()>(&model<T>::system_info));

    py::class_<AIM<T>, model<T>>(m, (std::string("AIM_")+label).c_str())
        .def(py::init())
        .def(py::init<size_t, size_t>())
        .def(py::init<const linalg::matrix<T>&, const linalg::tensor<T, 4>&>())
        .def(py::init<const linalg::matrix<T>&, const linalg::tensor<T, 4>&, const std::vector<size_t>&>())
        .def(py::init([](py::buffer& _T, py::buffer& _U)
                {
                    linalg::matrix<T, linalg::blas_backend> mT;
                    linalg::tensor<T, 4, linalg::blas_backend> tU;
                    conv::copy_to_tensor(_T, mT);
                    conv::copy_to_tensor(_U, tU);
                    return AIM<T>(mT, tU);
                }
            )
        )
        .def(py::init([](py::buffer& _T, py::buffer& _U, const std::vector<size_t>& inds)
                {
                    linalg::matrix<T, linalg::blas_backend> mT;
                    linalg::tensor<T, 4, linalg::blas_backend> tU;
                    conv::copy_to_tensor(_T, mT);
                    conv::copy_to_tensor(_U, tU);
                    return AIM<T>(mT, tU, inds);
                }
            )
        )
        .def(py::init<const linalg::matrix<T>&, const linalg::tensor<T, 4>&>())
        .def_property
            (
                "impurity_indices", 
                static_cast<const std::vector<size_t>& (AIM<T>::*)() const>(&AIM<T>::impurity_indices),
                [](AIM<T>& o, const std::vector<size_t>& i){o.impurity_indices() = i;}
            )
        .def("impurity_index", static_cast<const size_t& (AIM<T>::*)(size_t) const>(&AIM<T>::impurity_index), py::return_value_policy::reference)

        .def_property
            (
                "T", 
                static_cast<const linalg::matrix<T>& (AIM<T>::*)() const>(&AIM<T>::T), 
                [](AIM<T>& o, py::buffer& i){conv::copy_to_tensor<T, 2>(i, o.T());}
            )
        .def("Tij", static_cast<const T& (AIM<T>::*)(size_t, size_t) const>(&AIM<T>::T), py::return_value_policy::reference)

        .def_property
            (
                "U", 
                static_cast<const linalg::tensor<T, 4>& (AIM<T>::*)() const>(&AIM<T>::U),
                [](AIM<T>& o, py::buffer& i){conv::copy_to_tensor<T, 4>(i, o.U());}
            )
        .def("Uijkl", static_cast<const T& (AIM<T>::*)(size_t, size_t, size_t, size_t) const>(&AIM<T>::U), py::return_value_policy::reference);    

    py::class_<electronic_structure<T>, model<T>>(m, (std::string("electronic_structure_")+label).c_str())
        .def(py::init())
        .def(py::init<size_t>())
        .def(py::init<const linalg::matrix<T>&, const linalg::tensor<T, 4>&>())
        .def(py::init([](py::buffer& _T, py::buffer& _U)
                {
                    linalg::matrix<T, linalg::blas_backend> mT;
                    linalg::tensor<T, 4, linalg::blas_backend> tU;
                    conv::copy_to_tensor(_T, mT);
                    conv::copy_to_tensor(_U, tU);
                    return electronic_structure<T>(mT, tU);
                }
            )
        )

        .def_property
            (
                "T", 
                static_cast<const linalg::matrix<T>& (electronic_structure<T>::*)() const>(&electronic_structure<T>::T), 
                [](electronic_structure<T>& o, py::buffer& i){conv::copy_to_tensor<T, 2>(i, o.T());}
            )
        .def("Tij", static_cast<const T& (electronic_structure<T>::*)(size_t, size_t) const>(&electronic_structure<T>::T), py::return_value_policy::reference)

        .def_property
            (
                "U", 
                static_cast<const linalg::tensor<T, 4>& (electronic_structure<T>::*)() const>(&electronic_structure<T>::U),
                [](electronic_structure<T>& o, py::buffer& i){conv::copy_to_tensor<T, 4>(i, o.U());}
            )
        .def("Uijkl", static_cast<const T& (electronic_structure<T>::*)(size_t, size_t, size_t, size_t) const>(&electronic_structure<T>::U), py::return_value_policy::reference);


    py::class_<TFIM<T>, model<T>>(m, (std::string("TFIM_")+label).c_str())
        .def(py::init())
        .def(py::init<size_t, real_type, real_type, bool>(), py::arg(), py::arg(), py::arg(), py::arg("open_boundary_conditions") = false)
        .def_property
            (
                "open_boundary_conditions", 
                static_cast<const bool& (TFIM<T>::*)() const>(&TFIM<T>::open_boundary_conditions),
                [](TFIM<T>& o, const bool& i){o.open_boundary_conditions() = i;}
            )
        .def_property
            (
                "N", 
                static_cast<size_t(TFIM<T>::*)() const>(&TFIM<T>::N),
                [](TFIM<T>& o, const size_t& i){o.N() = i;}
            )

        .def_property
            (
                "t", 
                static_cast<const real_type& (TFIM<T>::*)() const>(&TFIM<T>::t),
                [](TFIM<T>& o, const real_type& i){o.t() = i;}
            )
        .def_property
            (
                "J", 
                static_cast<const real_type& (TFIM<T>::*)() const>(&TFIM<T>::J),
                [](TFIM<T>& o, const real_type& i){o.J() = i;}
            );



    py::class_<spin_boson_base<T>, model<T>>(m, (std::string("spin_boson_base_")+label).c_str())
        .def_property
            (
                "spin_index", 
                static_cast<size_t(spin_boson_base<T>::*)() const>(&spin_boson_base<T>::spin_index),
                [](spin_boson_base<T>& o, const size_t& i){o.spin_index() = i;}
            )

        .def_property
            (
                "eps", 
                static_cast<const real_type& (spin_boson_base<T>::*)() const>(&spin_boson_base<T>::eps),
                [](spin_boson_base<T>& o, const real_type& i){o.eps() = i;}
            )
        .def_property
            (
                "delta", 
                static_cast<const real_type& (spin_boson_base<T>::*)() const>(&spin_boson_base<T>::delta),
                [](spin_boson_base<T>& o, const real_type& i){o.delta() = i;}
            )
        .def_property
            (
                "mode_dims", 
                static_cast<const std::vector<size_t>& (spin_boson_base<T>::*)() const>(&spin_boson_base<T>::mode_dims),
                [](spin_boson_base<T>& o, const std::vector<size_t>& i){o.mode_dims() = i;}
            )
        .def("mode_dim", static_cast<const size_t& (spin_boson_base<T>::*)(size_t) const>(&spin_boson_base<T>::mode_dim), py::return_value_policy::reference);


    py::class_<spin_boson_generic<T>, spin_boson_base<T>>(m, (std::string("spin_boson_generic_")+label).c_str())
        .def(py::init())
        .def(py::init<real_type, real_type, const linalg::matrix<T>&>())
        .def(py::init<size_t, real_type, real_type, const linalg::matrix<T>&>())
        .def(py::init([](real_type eps, real_type delta, py::buffer& bT)
                {
                    linalg::matrix<T, linalg::blas_backend> mT;
                    conv::copy_to_tensor(bT, mT);
                    return spin_boson_generic<T>(eps, delta, mT);
                }
            )
        )
        .def(py::init([](size_t spin_index, real_type eps, real_type delta, py::buffer& bT)
                {
                    linalg::matrix<T, linalg::blas_backend> mT;
                    conv::copy_to_tensor(bT, mT);
                    return spin_boson_generic<T>(spin_index, eps, delta, mT);
                }
            )
        )
        .def_property
            (
                "T", 
                static_cast<const linalg::matrix<T>& (spin_boson_generic<T>::*)() const>(&spin_boson_generic<T>::T), 
                [](spin_boson_generic<T>& o, py::buffer& i){conv::copy_to_tensor<T, 2>(i, o.T());}
            );

    py::class_<spin_boson_star<T>, spin_boson_base<T>>(m, (std::string("spin_boson_star_")+label).c_str())
        .def(py::init())
        .def(py::init<real_type, real_type, const std::vector<real_type>&, const std::vector<T>&>())
        .def(py::init([](real_type eps, real_type del, const std::vector<real_type>& w, const std::vector<real_type>& g)
                {
                    std::vector<T> cg(g.size());
                    for(size_t i = 0; i < g.size(); ++i){cg[i] = g[i];}
                    return spin_boson_star<T>(eps, del, w, cg);
                }
            )
        )
        .def(py::init<size_t, real_type, real_type, const std::vector<real_type>&, const std::vector<T>&>())
        .def(py::init([](size_t spin_index, real_type eps, real_type del, const std::vector<real_type>& w, const std::vector<real_type>& g)
                {
                    std::vector<T> cg(g.size());
                    for(size_t i = 0; i < g.size(); ++i){cg[i] = g[i];}
                    return spin_boson_star<T>(spin_index, eps, del, w, cg);
                }
            )
        )
        .def_property
            (
                "w", 
                static_cast<const std::vector<real_type>& (spin_boson_star<T>::*)() const>(&spin_boson_star<T>::w), 
                [](spin_boson_star<T>& o, const std::vector<real_type>& i){o.w() = i;}
            )
        .def_property
            (
                "g", 
                static_cast<const std::vector<T>& (spin_boson_star<T>::*)() const>(&spin_boson_star<T>::g), 
                [](spin_boson_star<T>& o, const std::vector<T>& i){o.g() = i;}
            );

    py::class_<spin_boson_chain<T>, spin_boson_base<T>>(m, (std::string("spin_boson_chain_")+label).c_str())
        .def(py::init())
        .def(py::init<real_type, real_type, const std::vector<real_type>&, const std::vector<T>&>())
        .def(py::init([](real_type eps, real_type del, const std::vector<real_type>& e, const std::vector<real_type>& t)
                {
                    std::vector<T> ct(t.size());
                    for(size_t i = 0; i <t.size(); ++i){ct[i] = t[i];}
                    return spin_boson_chain<T>(eps, del, e, ct);
                }
            )
        )
        .def_property
            (
                "e", 
                static_cast<const std::vector<real_type>& (spin_boson_chain<T>::*)() const>(&spin_boson_chain<T>::e), 
                [](spin_boson_chain<T>& o, const std::vector<real_type>& i){o.e() = i;}
            )
        .def_property
            (
                "t", 
                static_cast<const std::vector<T>& (spin_boson_chain<T>::*)() const>(&spin_boson_chain<T>::t), 
                [](spin_boson_chain<T>& o, const std::vector<T>& i){o.t() = i;}
            );

}

void initialise_models(py::module& m);

#endif  //PYTHON_BINDING_SOP_MODELS_HPP


