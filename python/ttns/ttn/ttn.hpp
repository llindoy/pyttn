#ifndef PYTHON_BINDING_TTN_HPP
#define PYTHON_BINDING_TTN_HPP

#include <ttns_lib/ttn/ttn.hpp>
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

template <typename T>
void init_ttn(py::module &m, const std::string& label)
{
    using namespace ttns;
    using _ttn = ttn<T, linalg::blas_backend>;
    using _ttn_node = typename _ttn::node_type;
    using _ttn_node_data = ttn_node_data<T, linalg::blas_backend>;
    using real_type = typename linalg::get_real_type<T>::type;
    using siteop = site_operator<T, linalg::blas_backend>;

    py::class_<_ttn_node_data>(m, (std::string("ttn_data_")+label).c_str())
        .def(py::init())
        .def(py::init<const _ttn_node_data&>())
        //.def(py::init<_ttn_node_data&&>())
        .def("resize", &_ttn_node_data::resize)
        .def("reallocate", &_ttn_node_data::reallocate)

        .def("is_orthogonalised", &_ttn_node_data::reallocate)
        .def("complex_dtype", [](const _ttn_node_data&){return !std::is_same<T, real_type>::value;})

        .def("nmodes", &_ttn_node_data::nmodes)
        .def("hrank", &_ttn_node_data::hrank)
        .def("dimen", &_ttn_node_data::dimen)
        .def("dim", &_ttn_node_data::dim)
        .def("dims", &_ttn_node_data::dims)
        .def("set_dim", &_ttn_node_data::set_dim)

        .def("nelems", &_ttn_node_data::nelems)
        .def("nset", &_ttn_node_data::nset)
        .def("max_hrank", &_ttn_node_data::max_hrank)
        .def("max_dimen", &_ttn_node_data::max_dimen)
        .def("max_dim", &_ttn_node_data::max_dim)
        .def("max_dims", &_ttn_node_data::max_dims)
        .def("clear", &_ttn_node_data::clear)
        .def("__str__", [](const _ttn_node_data& o){std::ostringstream oss; oss << o; return oss.str();})

        .def("set_matrix", [](_ttn_node_data& i, const linalg::matrix<T>& mat){i.as_matrix() = mat;})
        .def("set_matrix", [](_ttn_node_data& i, py::buffer& mat){copy_pybuffer_to_tensor(mat, i.as_matrix());})
        //allow for setting of the elements
        .def(
                "as_matrix", 
                static_cast<const linalg::matrix<T>& (_ttn_node_data::*)() const>(&_ttn_node_data::as_matrix),
                py::return_value_policy::reference
            )
        .def(
                "as_matrix", 
                static_cast<linalg::matrix<T>& (_ttn_node_data::*)()>(&_ttn_node_data::as_matrix),
                py::return_value_policy::reference
            );

    py::class_<_ttn_node>(m, (std::string("ttn_node_")+label).c_str())
        .def(py::init())
        .def("data", static_cast<_ttn_node_data& (_ttn_node::*)()>(&_ttn_node::operator()))
        .def("data", static_cast<const _ttn_node_data& (_ttn_node::*)() const>(&_ttn_node::operator()))
        .def("is_root", &_ttn_node::is_root)
        .def("is_leaf", &_ttn_node::is_leaf)
        .def("complex_dtype", [](const _ttn_node&){return !std::is_same<T, real_type>::value;})
        .def("__len__", &_ttn_node::size)
        .def("__str__", [](const _ttn_node& o){std::ostringstream oss; oss << o; return oss.str();})
        .def(
                "__iter__",
                [](_ttn_node& s){return py::make_iterator(s.begin(), s.end());},
                py::keep_alive<0, 1>()
            );

    //expose the ttn node class.  This is our core tensor network object.
    py::class_<_ttn>(m, (std::string("ttn_")+label).c_str())
        .def(py::init())
        .def(py::init<const _ttn&>())
        .def(py::init<const ttn<real_type, linalg::blas_backend>&>())

        .def(py::init<const multiset_ttn_slice<real_type, linalg::blas_backend, true>&>())
        .def(py::init<const multiset_ttn_slice<T, linalg::blas_backend, true>&>())
        .def(py::init<const multiset_ttn_slice<real_type, linalg::blas_backend, false>&>())
        .def(py::init<const multiset_ttn_slice<T, linalg::blas_backend, false>&>())

        .def(py::init<const ntree<size_t>&, bool>(), py::arg(), py::arg("purification") = false)
        .def(py::init<const ntree<size_t>&, const ntree<size_t>&, bool>(), py::arg(), py::arg(), py::arg("purification") = false)
        .def(py::init<const std::string&, bool>(), py::arg(), py::arg("purification") = false)
        .def(py::init<const std::string&, const std::string, bool &>(), py::arg(), py::arg(), py::arg("purification") = false)

        .def("complex_dtype", [](const _ttn&){return !std::is_same<T, real_type>::value;})

        .def("assign", &_ttn::template operator=<T, linalg::blas_backend, true>)
        .def("assign", &_ttn::template operator=<T, linalg::blas_backend, false>)
        .def("assign", &_ttn::template operator=<real_type, linalg::blas_backend, true>)
        .def("assign", &_ttn::template operator=<real_type, linalg::blas_backend, false>)
        .def("assign", [](_ttn& o, const _ttn& i){o=i;})
        .def("assign", [](_ttn& o, const ttn<real_type, linalg::blas_backend>& i){o=i;})
        .def("__copy__", [](const _ttn& o){return _ttn(o);})
        .def("__deepcopy__", [](const _ttn& o, py::dict){return _ttn(o);}, py::arg("memo"))

        .def("bond_dimensions", &_ttn::bond_dimensions)
        .def("bond_dimensions", [](const _ttn& o){typename _ttn::hrank_info res;  o.bond_dimensions(res);   return res;})
        .def("reset_orthogonality", &_ttn::reset_orthogonality)
        .def("reset_orthogonality_centre", &_ttn::reset_orthogonality_centre)

        .def("resize", static_cast<void (_ttn::*)(const ntree<size_t, std::allocator<size_t>>&, size_t, bool)>(&_ttn::resize), py::arg(), py::arg("nset")=1, py::arg("purification") = false)
        .def("resize", static_cast<void (_ttn::*)(const ntree<size_t, std::allocator<size_t>>&, const ntree<size_t, std::allocator<size_t>>&, size_t, bool)>(&_ttn::resize), py::arg(), py::arg(), py::arg("nset")=1, py::arg("purification") = false)
        .def("resize", static_cast<void (_ttn::*)(const std::string&, size_t, bool)>(&_ttn::resize), py::arg(), py::arg("nset")=1, py::arg("purification") = false)
        .def("resize", static_cast<void (_ttn::*)(const std::string&, const std::string&, size_t, bool)>(&_ttn::resize), py::arg(), py::arg(), py::arg("nset")=1, py::arg("purification") = false)
        .def("set_seed", &_ttn::template set_seed<int>)


        .def("set_state", &_ttn::template set_state<int>)
        .def("set_state", &_ttn::template set_state<size_t>)
        .def("set_state_purification", &_ttn::template set_state<int>)
        .def("set_state_purification", &_ttn::template set_state<size_t>)

        .def("set_product", &_ttn::template set_product<real_type, linalg::blas_backend>)
        .def("set_product", &_ttn::template set_product<T, linalg::blas_backend>)
        .def("set_product", [](_ttn& self, std::vector<py::buffer>& ps)
                            {
                                std::vector<linalg::vector<T>> _ps(ps.size());
                                for(size_t i = 0; i < ps.size(); ++i)
                                {
                                    copy_pybuffer_to_tensor(ps[i], _ps[i]);
                                }
                                self.set_product(_ps);
                            }
            )
        .def("set_identity_purification", &_ttn::set_identity_purification)
        .def(
                "sample_product_state", 
                [](_ttn& o, const std::vector<std::vector<real_type>>& x)
                {
                    std::vector<size_t> state;
                    o.sample_product_state(state, x); 
                    return state;
                }
            )        

        .def("__imul__", [](_ttn& a, const real_type& b){return a*=b;})
        .def("__imul__", [](_ttn& a, const T& b){return a*=b;})
        .def("__idiv__", [](_ttn& a, const real_type& b){return a*=b;})
        .def("__idiv__", [](_ttn& a, const T& b){return a*=b;})
        
        .def("random", &_ttn::random)
        .def("zero", &_ttn::zero)

        .def("clear",&_ttn::clear)
        .def(
                "__iter__",
                [](_ttn& s){return py::make_iterator(s.begin(), s.end());},
                py::keep_alive<0, 1>()
            )

        .def("mode_dimensions", [](const _ttn& o){return o.mode_dimensions();})
        .def("dim", [](const _ttn& o, size_t i){return o.dim(i);})
        .def("nmodes", [](const _ttn& o){return o.nmodes();})
        .def("is_purification", &_ttn::is_purification)
        .def("ntensors", [](const _ttn& o){return o.ntensors();})
        .def("nsites", [](const _ttn& o){return o.ntensors();})
        .def("nset", &_ttn::nset)
        .def("nelems", [](const _ttn& o){return o.nelems();})
        .def("__len__", [](const _ttn& o){return o.nmodes();})

        .def("compute_maximum_bond_entropy", &_ttn::compute_maximum_bond_entropy)
        .def("maximum_bond_entropy", [](const _ttn& o){return o.maximum_bond_entropy();})
        .def("bond_entropy", &_ttn::bond_entropy)
        .def("maximum_bond_dimension", [](const _ttn& o){return o.maximum_bond_dimension();})
        .def("minimum_bond_dimension", [](const _ttn& o){return o.minimum_bond_dimension();})

        .def("has_orthogonality_centre", [](const _ttn& o){return o.has_orthogonality_centre();})
        .def("orthogonality_centre", [](const _ttn& o){return o.orthogonality_centre();})
        .def("is_orthogonalised", [](const _ttn& o){return o.is_orthogonalised();})

        .def("force_set_orthogonality_centre", static_cast<void (_ttn::*)(size_t)>(&_ttn::force_set_orthogonality_centre))
        .def("force_set_orthogonality_centre", static_cast<void (_ttn::*)(const std::list<size_t>&)>(&_ttn::force_set_orthogonality_centre))
        .def(
                "shift_orthogonality_centre", 
                &_ttn::shift_orthogonality_centre, py::arg(),  
                py::arg("tol") = 0, py::arg("nchi")=0
            )
        .def(
                "set_orthogonality_centre", 
                static_cast<void (_ttn::*)(size_t, double, size_t)>(&_ttn::set_orthogonality_centre), 
                py::arg(), py::arg("tol") = 0, py::arg("nchi")=0
            )
        .def(
                "set_orthogonality_centre", 
                static_cast<void (_ttn::*)(const std::list<size_t>&, double, size_t)>(&_ttn::set_orthogonality_centre), 
                py::arg(), py::arg("tol") = 0, py::arg("nchi")=0
            )
        .def(
                "orthogonalise", 
                &_ttn::orthogonalise, 
                py::arg("force")=false
            )
        .def(
                "truncate", 
                &_ttn::truncate, 
                py::arg("tol") = 0, py::arg("nchi")=0
            )

        .def("normalise", &_ttn::normalise)
        .def("norm", &_ttn::norm)

        .def("__str__", [](const _ttn& o){std::stringstream oss;   oss << o; return oss.str();})


        .def(
                "__setitem__", 
                [](_ttn& i, size_t ind, const _ttn_node_data& o){ i[ind]() = o;}
            )
        .def(
                "__getitem__", 
                [](const _ttn& i, size_t ind){ return i[ind]();}
            )        
        .def(   
                "site_tensor", 
                static_cast<const _ttn_node_data& (_ttn::*)(size_t) const>(&_ttn::site_tensor),
                py::return_value_policy::reference
            )
        .def(
                "set_site_tensor",
                [](_ttn& self, size_t i, const linalg::matrix<T>& mat)
                {
                    CALL_AND_HANDLE(self.site_tensor(i).as_matrix() = mat, "Failed to set site tensor.");
                }
            )        
        .def(
                "set_site_tensor",
                [](_ttn& self, size_t i, py::buffer& mat)
                {
                    CALL_AND_HANDLE(copy_pybuffer_to_tensor(mat, self.site_tensor(i).as_matrix()), "Failed to set site tensor.");
                }
            )


        .def(
                "measure_without_collapse", 
                [](_ttn& o, size_t i)
                {
                    std::vector<real_type> p0;
                    o.measure_without_collapse(i, p0);
                    return p0;
                }
            )
        .def(
                "measure_all_without_collapse", 
                [](_ttn& o)
                {
                    std::vector<std::vector<real_type>> p0;
                    o.measure_all_without_collapse(p0);
                    return p0;
                }
            )
        .def(
                "collapse_basis", 
                [](_ttn& o, std::vector<linalg::matrix<T>>& U, bool truncate=true, real_type tol=real_type(0), size_t nchi=0)
                {
                    std::vector<size_t> state;
                    real_type p = o.collapse_basis(U, state, truncate, tol, nchi); 
                    return std::make_pair(p, state);
                }, 
                py::arg(), py::arg("truncate")=true, py::arg("tol") = real_type(0), py::arg("nchi") = 0
            )
        .def(
                "collapse_basis", 
                [](_ttn& o, std::vector<py::buffer>& U, bool truncate=true, real_type tol=real_type(0), size_t nchi=0)
                {
                    std::vector<linalg::matrix<T>> Uop(U.size());
                    for(size_t i = 0; i < U.size(); ++i)
                    {
                        copy_pybuffer_to_tensor(U[i], Uop[i]);
                    }
                    std::vector<size_t> state;
                    real_type p = o.collapse_basis(Uop, state, truncate, tol, nchi); 
                    return std::make_pair(p, state);
                }, 
                py::arg(), py::arg("truncate")=true, py::arg("tol") = real_type(0), py::arg("nchi") = 0
            )
        .def(
                "collapse", 
                [](_ttn& o, bool truncate=true, real_type tol=real_type(0), size_t nchi=0)
                {
                    std::vector<size_t> state;
                    real_type p = o.collapse(state, truncate, tol, nchi); 
                    return std::make_pair(p, state);
                }, 
                py::arg("truncate")=true, py::arg("tol") = real_type(0), py::arg("nchi") = 0
            )

        .def(
                "apply_one_body_operator", 
                [](_ttn& o, const linalg::matrix<T>& op, size_t index, bool shift_orthogonality){CALL_AND_RETHROW(return o.apply_one_body_operator(op, index, shift_orthogonality));},
                py::arg(), py::arg(), py::arg("shift_orthogonality") = true
            )
        .def(
                "apply_one_body_operator", 
                [](_ttn& o, py::buffer& op, size_t index, bool shift_orthogonality)
                {
                    linalg::matrix<T> mt;
                    copy_pybuffer_to_tensor(op, mt);
                    CALL_AND_RETHROW(return o.apply_one_body_operator(mt, index, shift_orthogonality));
                },
                py::arg(), py::arg(), py::arg("shift_orthogonality") = true
            )
        .def(
                "apply_one_body_operator", 
                [](_ttn& o, siteop& op, bool shift_orthogonality){CALL_AND_RETHROW(return o.apply_one_body_operator(op, shift_orthogonality));},
                py::arg(), py::arg("shift_orthogonality") = true
            )
        .def(
                "apply_one_body_operator", 
                [](_ttn& o, siteop& op, size_t index, bool shift_orthogonality){CALL_AND_RETHROW(return o.apply_one_body_operator(op, index, shift_orthogonality));},
                py::arg(), py::arg(), py::arg("shift_orthogonality") = true
            )
        //Increment with additional operators as they are bound.

        //ttn& apply_one_body_operator(const Op<T, backend>& op, bool shift_orthogonality = true)
        //ttn& apply_operator(const Op<T, backend>& op, real_type tol = real_type(0), size_type nchi=0)
        ;



}

void initialise_ttn(py::module& m);

#endif  //PYTHON_BINDING_TTN_HPP


