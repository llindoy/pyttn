#ifndef PYTHON_BINDING_MS_TTN_HPP
#define PYTHON_BINDING_MS_TTN_HPP

#include <ttns_lib/ttn/ms_ttn.hpp>
#include <ttns_lib/ttn/multiset_ttn_slice.hpp>
#include <ttns_lib/operators/site_operators/site_operator.hpp>
#include "../../utils.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py=pybind11;

template <typename T, typename backend>
void init_msttn(py::module &m, const std::string& label)
{
    using namespace ttns;
    using real_type = typename linalg::get_real_type<T>::type;
    using size_type = typename backend::size_type;

    using _msttn = ms_ttn<T, backend>;
    using _msttn_node = typename _msttn::node_type;
    using _msttn_node_data = multiset_node_data<T, backend>;
    using _msttn_slice = multiset_ttn_slice<T, backend, false>;
    //using _msttn_slice_real = multiset_ttn_slice<real_type, backend, false>;

    py::class_<_msttn_node>(m, (std::string("ms_ttn_node_")+label).c_str())
        .def(py::init())
        .def("data", static_cast<_msttn_node_data& (_msttn_node::*)()>(&_msttn_node::operator()))
        .def("data", static_cast<const _msttn_node_data& (_msttn_node::*)() const>(&_msttn_node::operator()))
        .def("is_root", &_msttn_node::is_root)
        .def("is_leaf", &_msttn_node::is_leaf)
        .def("complex_dtype", [](const _msttn_node&){return !std::is_same<T, real_type>::value;})
        .def("conj", &_msttn_node::conj)
        .def("__len__", &_msttn_node::size)
        .def("__str__", [](const _msttn_node& o){std::ostringstream oss; oss << o; return oss.str();})
        .def(
                "__iter__",
                [](_msttn_node& s){return py::make_iterator(s.begin(), s.end());},
                py::keep_alive<0, 1>()
            );

    //TODO: Figure out why the commented out assign functions don't compile
    using sobj_type = typename _msttn_slice::obj_type;
    py::class_<_msttn_slice>(m, (std::string("ms_ttn_slice_")+label).c_str())
        .def(py::init<sobj_type, size_type>())
        .def(py::init<const _msttn_slice&>())
        .def("assign", static_cast<_msttn_slice& (_msttn_slice::*)(const ttn<real_type, backend>&)>(&_msttn_slice::template operator=<real_type, backend>))
        .def("assign", static_cast<_msttn_slice& (_msttn_slice::*)(const ttn<T, backend>&)>(&_msttn_slice::template operator=<T, backend>))
        //.def("assign", static_cast<_msttn_slice& (_msttn_slice::*)(const _msttn_slice_real&)>(&_msttn_slice::template operator=<real_type, backend>))
        //.def("assign", static_cast<_msttn_slice& (_msttn_slice::*)(const _msttn_slice&)>(&_msttn_slice::template operator=<T, backend>))
        .def("nset", &_msttn_slice::nset);


    //expose the ttn node class.  This is our core tensor network object.
    py::class_<_msttn>(m, (std::string("ms_ttn_")+label).c_str())
        .def(py::init<const _msttn&>())
        .def(py::init<const ms_ttn<real_type, backend>&>())

        .def(py::init<size_t, const ntree<size_t>&, bool>(), py::arg(), py::arg(), py::arg("purification") = false)
        .def(py::init<size_t, const ntree<size_t>&, const ntree<size_t>&, bool>(), py::arg(), py::arg(), py::arg(), py::arg("purification") = false)
        .def(py::init<size_t, const std::string&, bool>(), py::arg(), py::arg(), py::arg("purification") = false)
        .def(py::init<size_t, const std::string&, const std::string, bool &>(), py::arg(), py::arg(), py::arg(), py::arg("purification") = false)

        .def("complex_dtype", [](const _msttn&){return !std::is_same<T, real_type>::value;})
        .def("assign", [](_msttn& o, const _msttn& i){o=i;})
        //.def("assign", &_msttn::template operator=<real_type, backend> )
        .def("__copy__", [](const _msttn& o){return _msttn(o);})
        .def("__deepcopy__", [](const _msttn& o, py::dict){return _msttn(o);}, py::arg("memo"))

        .def("slice", static_cast<_msttn_slice(_msttn::*)(size_t)>(&_msttn::slice))

        .def("assign_slice", static_cast<void (_msttn::*)(size_type, const ttn<real_type, backend>&)>(&_msttn::template set_slice<real_type, backend>))
        .def("assign_slice", static_cast<void (_msttn::*)(size_type, const ttn<T, backend>&)>(&_msttn::template set_slice<T, backend>))
        //.def("assign_slice", static_cast<void (_msttn::*)(size_type, const _msttn_slice_real&)>(&_msttn::template set_slice<real_type, backend>)) 
        //.def("assign_slice", static_cast<void (_msttn::*)(size_type, const _msttn_slice&)>(&_msttn::template set_slice<T, backend>))

        .def("bond_dimensions", &_msttn::bond_dimensions)
        .def("bond_dimensions", [](const _msttn& o){typename _msttn::hrank_info res;  o.bond_dimensions(res);   return res;})

        .def("resize", static_cast<void (_msttn::*)(const ntree<size_t, std::allocator<size_t>>&, size_t, bool)>(&_msttn::resize), py::arg(), py::arg(), py::arg("purification") = false)
        .def("resize", static_cast<void (_msttn::*)(const ntree<size_t, std::allocator<size_t>>&, const ntree<size_t, std::allocator<size_t>>&, size_t, bool)>(&_msttn::resize), py::arg(), py::arg(), py::arg(), py::arg("purification") = false)
        .def("resize", static_cast<void (_msttn::*)(const std::string&, size_t, bool)>(&_msttn::resize), py::arg(), py::arg(), py::arg("purification") = false)
        .def("resize", static_cast<void (_msttn::*)(const std::string&, const std::string&, size_t, bool)>(&_msttn::resize), py::arg(), py::arg(), py::arg(), py::arg("purification") = false)
        .def("set_seed", &_msttn::template set_seed<int>)
        .def("reset_orthogonality_centre", &_msttn::reset_orthogonality_centre)

        .def("set_state", &_msttn::template set_state<int>, py::arg(), py::arg("random_unoccupied_initialisation")=false)
        .def("set_state", &_msttn::template set_state<size_t>, py::arg(), py::arg("random_unoccupied_initialisation")=false)
        .def("set_state", &_msttn::template set_state<T, int>, py::arg(), py::arg(), py::arg("random_unoccupied_initialisation")=false)
        .def("set_state", &_msttn::template set_state<T, size_t>, py::arg(), py::arg(), py::arg("random_unoccupied_initialisation")=false)
        .def("set_state", &_msttn::template set_state<real_type, int>, py::arg(), py::arg(), py::arg("random_unoccupied_initialisation")=false)
        .def("set_state", &_msttn::template set_state<real_type, size_t>, py::arg(), py::arg(), py::arg("random_unoccupied_initialisation")=false)
        .def("set_state_purification", &_msttn::template set_state<int>, py::arg(), py::arg("random_unoccupied_initialisation")=false)
        .def("set_state_purification", &_msttn::template set_state<size_t>, py::arg(), py::arg("random_unoccupied_initialisation")=false)

        //.def("set_product", &_msttn::template set_product<real_type, backend>)
        //.def("set_product", &_msttn::template set_product<T, backend>)
        //.def("set_product", [](_msttn& self, std::vector<py::buffer>& ps)
        //                    {
        //                        std::vector<linalg::vector<T>> _ps(ps.size());
        //                        for(size_t i = 0; i < ps.size(); ++i)
        //                        {
        //                            conv::copy_to_tensor(ps[i], _ps[i]);
        //                        }
        //                        self.set_product(_ps);
        //                    }
        //    )
        //.def("set_identity_purification", &_msttn::set_identity_purification)
        //.def(
        //        "sample_product_state", 
        //        [](_msttn& o, const std::vector<std::vector<real_type>>& x)
        //        {
        //            std::vector<size_t> state;
        //            o.sample_product_state(state, x); 
        //            return state;
        //        }
        //    )        

        .def("__imul__", [](_msttn& a, const real_type& b){return a*=b;})
        .def("__imul__", [](_msttn& a, const T& b){return a*=b;})
        .def("__idiv__", [](_msttn& a, const real_type& b){return a*=b;})
        .def("__idiv__", [](_msttn& a, const T& b){return a*=b;})
        
        .def("conj", &_msttn::conj)
        .def("random", &_msttn::random)
        .def("zero", &_msttn::zero)

        .def("clear",&_msttn::clear)
        .def(
                "__iter__",
                [](_msttn& s){return py::make_iterator(s.begin(), s.end());},
                py::keep_alive<0, 1>()
            )

        .def_property
            (
                "nthreads", 
                static_cast<size_type (_msttn::*)() const>(&_msttn::nthreads),
                [](_msttn& o, const size_type& i){o.nthreads() = i;}
            )
        .def("mode_dimensions", [](const _msttn& o){return o.mode_dimensions();})
        .def("dim", [](const _msttn& o, size_t i){return o.dim(i);})
        .def("nmodes", [](const _msttn& o){return o.nmodes();})
        .def("is_purification", &_msttn::is_purification)
        .def("ntensors", [](const _msttn& o){return o.ntensors();})
        .def("nsites", [](const _msttn& o){return o.ntensors();})
        .def("nset", &_msttn::nset)
        .def("nelems", [](const _msttn& o){return o.nelems();})
        .def("__len__", [](const _msttn& o){return o.nmodes();})

        //.def("compute_maximum_bond_entropy", &_msttn::compute_maximum_bond_entropy)
        //.def("maximum_bond_entropy", [](const _msttn& o){return o.maximum_bond_entropy();})
        //.def("bond_entropy", &_msttn::bond_entropy)
        //.def("maximum_bond_dimension", [](const _msttn& o){return o.maximum_bond_dimension();})
        //.def("minimum_bond_dimension", [](const _msttn& o){return o.minimum_bond_dimension();})

        .def("has_orthogonality_centre", [](const _msttn& o){return o.has_orthogonality_centre();})
        .def("orthogonality_centre", [](const _msttn& o){return o.orthogonality_centre();})
        .def("is_orthogonalised", [](const _msttn& o){return o.is_orthogonalised();})

        .def("force_set_orthogonality_centre", static_cast<void (_msttn::*)(size_t)>(&_msttn::force_set_orthogonality_centre))
        .def("force_set_orthogonality_centre", static_cast<void (_msttn::*)(const std::list<size_t>&)>(&_msttn::force_set_orthogonality_centre))
        .def(
                "shift_orthogonality_centre", 
                &_msttn::shift_orthogonality_centre, py::arg(),  
                py::arg("tol") = 0, py::arg("nchi")=0
            )
        .def(
                "set_orthogonality_centre", 
                static_cast<void (_msttn::*)(size_t, double, size_t)>(&_msttn::set_orthogonality_centre), 
                py::arg(), py::arg("tol") = 0, py::arg("nchi")=0
            )
        .def(
                "set_orthogonality_centre", 
                static_cast<void (_msttn::*)(const std::list<size_t>&, double, size_t)>(&_msttn::set_orthogonality_centre), 
                py::arg(), py::arg("tol") = 0, py::arg("nchi")=0
            )
        .def(
                "orthogonalise", 
                &_msttn::orthogonalise, 
                py::arg("force")=false
            )
        .def(
                "truncate", 
                &_msttn::truncate, 
                py::arg("tol") = 0, py::arg("nchi")=0
            )

        .def("normalise", &_msttn::normalise)
        .def("norm", &_msttn::norm)

        .def("__str__", [](const _msttn& o){std::stringstream oss;   oss << o; return oss.str();})


        .def(
                "__setitem__", 
                [](_msttn& i, size_t ind, const _msttn_node_data& o){ i[ind]() = o;}
            )
        .def(
                "__getitem__", 
                [](_msttn& i, size_t ind) -> _msttn_node_data& { return i[ind]();}, 
                py::return_value_policy::reference
            )        
        .def(   
                "site_tensor", 
                static_cast<const _msttn_node_data& (_msttn::*)(size_t) const>(&_msttn::site_tensor),
                py::return_value_policy::reference
            )
        //.def(
        //        "set_site_tensor",
        //        [](_msttn& self, size_t i, const linalg::matrix<T>& mat)
        //        {
        //            CALL_AND_HANDLE(self.site_tensor(i).as_matrix() = mat, "Failed to set site tensor.");
        //        }
        //    )        
        //.def(
        //        "set_site_tensor",
        //        [](_msttn& self, size_t i, py::buffer& mat)
        //        {
        //            CALL_AND_HANDLE(conv::copy_to_tensor(mat, self.site_tensor(i).as_matrix()), "Failed to set site tensor.");
        //        }
        //    )


        //.def(
        //        "measure_without_collapse", 
        //        [](_msttn& o, size_t i)
        //        {
        //            std::vector<real_type> p0;
        //            o.measure_without_collapse(i, p0);
        //            return p0;
        //        }
        //    )
        //.def(
        //        "measure_all_without_collapse", 
        //        [](_msttn& o)
        //        {
        //            std::vector<std::vector<real_type>> p0;
        //            o.measure_all_without_collapse(p0);
        //            return p0;
        //        }
        //    )
        //.def(
        //        "collapse_basis", 
        //        [](_msttn& o, std::vector<linalg::matrix<T>>& U, bool truncate=true, real_type tol=real_type(0), size_t nchi=0)
        //        {
        //            std::vector<size_t> state;
        //            real_type p = o.collapse_basis(U, state, truncate, tol, nchi); 
        //            return std::make_pair(p, state);
        //        }, 
        //        py::arg(), py::arg("truncate")=true, py::arg("tol") = real_type(0), py::arg("nchi") = 0
        //    )
        //.def(
        //        "collapse_basis", 
        //        [](_msttn& o, std::vector<py::buffer>& U, bool truncate=true, real_type tol=real_type(0), size_t nchi=0)
        //        {
        //            std::vector<linalg::matrix<T>> Uop(U.size());
        //            for(size_t i = 0; i < U.size(); ++i)
        //            {
        //                conv::copy_to_tensor(U[i], Uop[i]);
        //            }
        //            std::vector<size_t> state;
        //            real_type p = o.collapse_basis(Uop, state, truncate, tol, nchi); 
        //            return std::make_pair(p, state);
        //        }, 
        //        py::arg(), py::arg("truncate")=true, py::arg("tol") = real_type(0), py::arg("nchi") = 0
        //    )
        //.def(
        //        "collapse", 
        //        [](_msttn& o, bool truncate=true, real_type tol=real_type(0), size_t nchi=0)
        //        {
        //            std::vector<size_t> state;
        //            real_type p = o.collapse(state, truncate, tol, nchi); 
        //            return std::make_pair(p, state);
        //        }, 
        //        py::arg("truncate")=true, py::arg("tol") = real_type(0), py::arg("nchi") = 0
        //    )

        //.def(
        //        "apply_one_body_operator", 
        //        [](_msttn& o, const linalg::matrix<T>& op, size_t index, bool shift_orthogonality){CALL_AND_RETHROW(return o.apply_one_body_operator(op, index, shift_orthogonality));},
        //        py::arg(), py::arg(), py::arg("shift_orthogonality") = true
        //    )
        //.def(
        //        "apply_one_body_operator", 
        //        [](_msttn& o, py::buffer& op, size_t index, bool shift_orthogonality)
        //        {
        //            linalg::matrix<T> mt;
        //            conv::copy_to_tensor(op, mt);
        //            CALL_AND_RETHROW(return o.apply_one_body_operator(mt, index, shift_orthogonality));
        //        },
        //        py::arg(), py::arg(), py::arg("shift_orthogonality") = true
        //    )
        //.def(
        //        "apply_one_body_operator", 
        //        [](_msttn& o, siteop& op, bool shift_orthogonality){CALL_AND_RETHROW(return o.apply_one_body_operator(op, shift_orthogonality));},
        //        py::arg(), py::arg("shift_orthogonality") = true
        //    )
        //.def(
        //        "apply_one_body_operator", 
        //        [](_msttn& o, siteop& op, size_t index, bool shift_orthogonality){CALL_AND_RETHROW(return o.apply_one_body_operator(op, index, shift_orthogonality));},
        //        py::arg(), py::arg(), py::arg("shift_orthogonality") = true
        //    )
        //Increment with additional operators as they are bound.

        //ttn& apply_one_body_operator(const Op<T, backend>& op, bool shift_orthogonality = true)
        //ttn& apply_operator(const Op<T, backend>& op, real_type tol = real_type(0), size_type nchi=0)
        ;
}

template <typename real_type, typename backend>
void initialise_msttn(py::module& m)
{
    using complex_type = linalg::complex<real_type>;

#ifdef BUILD_REAL_TTN
    init_msttn<real_type, backend>(m, "real");
#endif
    init_msttn<complex_type, backend>(m, "complex");
}

#endif  //PYTHON_BINDING_TTN_HPP


