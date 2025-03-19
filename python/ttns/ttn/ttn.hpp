/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2025 NPL Management Limited
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

#ifndef PYTHON_BINDING_TTN_HPP
#define PYTHON_BINDING_TTN_HPP

#include <ttns_lib/ttn/ttn.hpp>
#include <ttns_lib/ttn/multiset_ttn_slice.hpp>
#include <ttns_lib/operators/site_operators/site_operator.hpp>
#include <ttns_lib/operators/sop_operator.hpp>
#include <ttns_lib/ttn/sop_ttn_contraction.hpp>
#include "../../utils.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py = pybind11;

template <typename T, typename backend>
void init_ttn(py::module &m, const std::string &label)
{
    using namespace ttns;
    using _ttn = ttn<T, backend>;
    using _ttn_node = typename _ttn::node_type;
    using _ttn_node_data = ttn_node_data<T, backend>;
    using real_type = typename linalg::get_real_type<T>::type;
    using siteop = site_operator<T, backend>;
    using prodop = product_operator<T, backend>;
    using sop = sop_operator<T, backend>;

    using numpy_type = typename linalg::numpy_converter<T>::type;

    using conv = linalg::pybuffer_converter<backend>;

#ifdef PYTTN_BUILD_CUDA
    using otherbackend = typename other_backend<backend>::type;
#endif

    py::class_<_ttn_node_data>(m, (std::string("ttn_data_") + label).c_str())
        .def(py::init())
        .def(py::init<const _ttn_node_data &>())
        //.def(py::init<_ttn_node_data&&>())
        .def("resize", &_ttn_node_data::resize)
        .def("reallocate", &_ttn_node_data::reallocate)

        .def("is_orthogonalised", &_ttn_node_data::reallocate)
        .def("complex_dtype", [](const _ttn_node_data &)
             { return !std::is_same<T, real_type>::value; })

        .def("conj", &_ttn_node_data::conj)
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
        .def("__str__", [](const _ttn_node_data &o)
             {std::ostringstream oss; oss << o; return oss.str(); })

        .def("set_matrix", [](_ttn_node_data &i, const linalg::matrix<T, backend> &mat)
             { i.as_matrix() = mat; })
#ifdef PYTTN_BUILD_CUDA
        .def("set_matrix", [](_ttn_node_data &i, const linalg::matrix<T, otherbackend> &mat)
             { i.as_matrix() = mat; })
#endif
        .def("set_matrix", [](_ttn_node_data &i, py::buffer &mat)
             { conv::copy_to_tensor(mat, i.as_matrix()); })
        // allow for setting of the elements
        .def(
            "as_matrix",
            static_cast<const linalg::matrix<T, backend> &(_ttn_node_data::*)() const>(&_ttn_node_data::as_matrix),
            py::return_value_policy::reference)
        .def(
            "as_matrix",
            static_cast<linalg::matrix<T, backend> &(_ttn_node_data::*)()>(&_ttn_node_data::as_matrix),
            py::return_value_policy::reference)
        .def("backend", [](const _ttn_node_data &)
             { return backend::label(); });

    py::class_<_ttn_node>(m, (std::string("ttn_node_") + label).c_str())
        .def(py::init())
        .def("data", static_cast<_ttn_node_data &(_ttn_node::*)()>(&_ttn_node::operator()))
        .def("data", static_cast<const _ttn_node_data &(_ttn_node::*)() const>(&_ttn_node::operator()))
        .def("is_root", &_ttn_node::is_root)
        .def("is_leaf", &_ttn_node::is_leaf)
        .def("complex_dtype", [](const _ttn_node &)
             { return !std::is_same<T, real_type>::value; })
        .def("conj", &_ttn_node::conj)
        .def("__len__", &_ttn_node::size)
        .def("__str__", [](const _ttn_node &o)
             {std::ostringstream oss; oss << o; return oss.str(); })
        .def(
            "__iter__",
            [](_ttn_node &s)
            { return py::make_iterator(s.begin(), s.end()); },
            py::keep_alive<0, 1>())
        .def("backend", [](const _ttn_node &)
             { return backend::label(); });

    // expose the ttn node class.  This is our core tensor network object.
    py::class_<_ttn>(m, (std::string("ttn_") + label).c_str())
        .def(py::init())
        .def(py::init<const _ttn &>())
        .def(py::init<const ttn<real_type, backend> &>())

        .def(py::init<const multiset_ttn_slice<real_type, backend, true> &>())
        .def(py::init<const multiset_ttn_slice<T, backend, true> &>())
        .def(py::init<const multiset_ttn_slice<real_type, backend, false> &>())
        .def(py::init<const multiset_ttn_slice<T, backend, false> &>())

        .def(py::init<const ntree<size_t> &, bool>(), py::arg(), py::arg("purification") = false)
        .def(py::init<const ntree<size_t> &, const ntree<size_t> &, bool>(), py::arg(), py::arg(), py::arg("purification") = false)
        .def(py::init<const std::string &, bool>(), py::arg(), py::arg("purification") = false)
        .def(py::init<const std::string &, const std::string, bool &>(), py::arg(), py::arg(), py::arg("purification") = false, "For details see :class:`ttn_dtype`")

        .def("complex_dtype", [](const _ttn &)
             { return !std::is_same<T, real_type>::value; }, "For details see :class:`pyttn.ttn_dtype`")

        .def("assign", &_ttn::template operator= <T, backend, true>)
        .def("assign", &_ttn::template operator= <T, backend, false>)
        .def("assign", &_ttn::template operator= <real_type, backend, true>)
        .def("assign", &_ttn::template operator= <real_type, backend, false>, "For details see :meth:`pyttn.ttn_dtype.assign`")
        .def("assign", [](_ttn &o, const _ttn &i)
             { o = i; })
        .def("assign", [](_ttn &o, const ttn<real_type, backend> &i)
             { o = i; })
        .def("__copy__", [](const _ttn &o)
             { return _ttn(o); })
        .def("__deepcopy__", [](const _ttn &o, py::dict)
             { return _ttn(o); }, py::arg("memo"))

        .def("bonds", [](const _ttn &o)
             {std::vector<std::pair<int, int>> edges; o.get_edges(edges); return edges; })

        .def("bond_dimensions", &_ttn::bond_dimensions)
        .def("bond_dimensions", [](const _ttn &o)
             {typename _ttn::hrank_info res;  o.bond_dimensions(res);   return res; })
        .def("bond_dimensions", [](const _ttn &o, typename _ttn::hrank_info &res)
             {;  o.bond_dimensions(res); }, "For details see :meth:`pyttn.ttn_dtype.bond_dimensions`")

        .def("bond_capacities", &_ttn::bond_capacities)
        .def("bond_capacities", [](const _ttn &o)
             {typename _ttn::hrank_info res;  o.bond_capacities(res);   return res; })
        .def("bond_capacities", [](const _ttn &o, typename _ttn::hrank_info &res)
             {;  o.bond_capacities(res); }, "For details see :meth:`pyttn.ttn_dtype.bond_capacities`")

        //.def("reset_orthogonality", &_ttn::reset_orthogonality)
        .def("reset_orthogonality_centre", &_ttn::reset_orthogonality_centre)

        .def("resize", static_cast<void (_ttn::*)(const ntree<size_t, std::allocator<size_t>> &, size_t, bool)>(&_ttn::resize), py::arg(), py::arg("nset") = 1, py::arg("purification") = false)
        .def("resize", static_cast<void (_ttn::*)(const ntree<size_t, std::allocator<size_t>> &, const ntree<size_t, std::allocator<size_t>> &, size_t, bool)>(&_ttn::resize), py::arg(), py::arg(), py::arg("nset") = 1, py::arg("purification") = false)
        .def("resize", static_cast<void (_ttn::*)(const std::string &, size_t, bool)>(&_ttn::resize), py::arg(), py::arg("nset") = 1, py::arg("purification") = false)
        .def("resize", static_cast<void (_ttn::*)(const std::string &, const std::string &, size_t, bool)>(&_ttn::resize), py::arg(), py::arg(), py::arg("nset") = 1, py::arg("purification") = false, "For details see :meth:`pyttn.ttn_dtype.resize`")
        .def("set_seed", &_ttn::template set_seed<int>, "For details see :meth:`pyttn.ttn_dtype.set_seed`")

        .def("set_state", &_ttn::template set_state<int>, py::arg(), py::arg("random_primitive") = false, py::arg("random_internal") = true)
        .def("set_state", &_ttn::template set_state<size_t>, py::arg(), py::arg("random_primitive") = false, py::arg("random_internal") = true, "For details see :meth:`pyttn.ttn_dtype.set_state`")
        .def("set_state_purification", &_ttn::template set_state<int>, py::arg(), py::arg("random_primitive") = false, py::arg("random_internal") = true)
        .def("set_state_purification", &_ttn::template set_state<size_t>, py::arg(), py::arg("random_primitive") = false, py::arg("random_internal") = true, "For details see :meth:`pyttn.ttn_dtype.set_state_purification`")

        .def("set_product", &_ttn::template set_product<real_type, backend>, py::arg(), py::arg("random_internal") = true)
        .def("set_product", &_ttn::template set_product<T, backend>, py::arg(), py::arg("random_internal") = true)
        .def("set_product", [](_ttn &self, std::vector<py::buffer> &ps, bool random_internal)
             {
                                std::vector<linalg::vector<T, backend>> _ps(ps.size());
                                for(size_t i = 0; i < ps.size(); ++i)
                                {
                                    conv::copy_to_tensor(ps[i], _ps[i]);
                                }
                                self.set_product(_ps, random_internal); }, py::arg(), py::arg("random_internal") = true, "For details see :meth:`pyttn.ttn_dtype.set_product`")
        .def("set_identity_purification", &_ttn::set_identity_purification, py::arg("random_internal") = true, "For details see :meth:`pyttn.ttn_dtype.set_identity_purification`")
        .def("sample_product_state", [](_ttn &o, const std::vector<std::vector<real_type>> &x, bool random_internal)
             {
                    std::vector<size_t> state;
                    o.sample_product_state(state, x, random_internal); 
                    return state; }, py::arg(), py::arg("random_internal") = true, "For details see :meth:`pyttn.ttn_dtype.sample_product_state`")

        .def("__imul__", [](_ttn &a, const real_type &b)
             { return a *= b; })
        .def("__imul__", [](_ttn &a, const numpy_type &b)
             { return a *= T(b); }, "For details see :meth:`pyttn.ttn_dtype.__imul__`")
        .def("__idiv__", [](_ttn &a, const real_type &b)
             { return a /= b; })
        .def("__idiv__", [](_ttn &a, const numpy_type &b)
             { return a /= T(b); }, "For details see :meth:`pyttn.ttn_dtype.__idiv__`")

        .def("conj", &_ttn::conj, "For details see :meth:`pyttn.ttn_dtype.conj`")
        .def("random", &_ttn::random, "For details see :meth:`pyttn.ttn_dtype.random`")
        .def("zero", &_ttn::zero, "For details see :meth:`pyttn.ttn_dtype.zero`")

        .def("clear", &_ttn::clear, "For details see :meth:`pyttn.ttn_dtype.clear`")
        .def("__iter__", [](_ttn &s)
             { return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>(), "For details see :meth:`pyttn.ttn_dtype.__iter__`")

        .def("mode_dimensions", [](const _ttn &o)
             { return o.mode_dimensions(); }, "For details see :meth:`pyttn.ttn_dtype.mode_dimensions`")
        .def("dim", [](const _ttn &o, size_t i)
             { return o.dim(i); }, "For details see :meth:`pyttn.ttn_dtype.dim`")
        .def("nmodes", [](const _ttn &o)
             { return o.nmodes(); }, "For details see :meth:`pyttn.ttn_dtype.nmodes`")
        .def("is_purification", &_ttn::is_purification, "For details see :meth:`pyttn.ttn_dtype.is_purification`")
        .def("ntensors", [](const _ttn &o)
             { return o.ntensors(); }, "For details see :meth:`pyttn.ttn_dtype.ntensors`")
        .def("nsites", [](const _ttn &o)
             { return o.ntensors(); }, "For details see :meth:`pyttn.ttn_dtype.nsites`")
        .def("nset", &_ttn::nset, "For details see :meth:`pyttn.ttn_dtype.nset`")
        .def("nelems", [](const _ttn &o)
             { return o.nelems(); }, "For details see :meth:`pyttn.ttn_dtype.nelems`")
        .def("__len__", [](const _ttn &o)
             { return o.nmodes(); }, "For details see :meth:`pyttn.ttn_dtype.__len__`")

        .def("compute_maximum_bond_entropy", &_ttn::compute_maximum_bond_entropy, "For details see :meth:`pyttn.ttn_dtype.compute_maximum_bond_entropy`")
        .def("maximum_bond_entropy", [](const _ttn &o)
             { return o.maximum_bond_entropy(); }, "For details see :meth:`pyttn.ttn_dtype.maximum_bond_entropy`")
        .def("bond_entropy", &_ttn::bond_entropy, "For details see :meth:`pyttn.ttn_dtype.bond_entropy`")
        .def("maximum_bond_dimension", [](const _ttn &o)
             { return o.maximum_bond_dimension(); }, "For details see :meth:`pyttn.ttn_dtype.maximum_bond_dimension")
        .def("minimum_bond_dimension", [](const _ttn &o)
             { return o.minimum_bond_dimension(); }, "For details see :meth:`pyttn.ttn_dtype.minimum_bond_dimension`")

        .def("has_orthogonality_centre", [](const _ttn &o)
             { return o.has_orthogonality_centre(); }, "For details see :meth:`pyttn.ttn_dtype.has_orthogonality_centre`")
        .def("orthogonality_centre", [](const _ttn &o)
             { return o.orthogonality_centre(); }, "For details see :meth:`pyttn.ttn_dtype.orthogonality_centre`")
        .def("is_orthogonalised", [](const _ttn &o)
             { return o.is_orthogonalised(); }, "For details see :meth:`pyttn.ttn_dtype.is_orthogonalised`")

        .def("force_set_orthogonality_centre", static_cast<void (_ttn::*)(size_t)>(&_ttn::force_set_orthogonality_centre))
        .def("force_set_orthogonality_centre", static_cast<void (_ttn::*)(const std::list<size_t> &)>(&_ttn::force_set_orthogonality_centre), "For details see :meth:`pyttn.ttn_dtype.force_set_orthogoanlity_centre`")
        .def("shift_orthogonality_centre", &_ttn::shift_orthogonality_centre, py::arg(), py::arg("tol") = 0, py::arg("nchi") = 0, "For details see :meth:`pyttn.ttn_dtype.shift_orthogonality_centre`")
        .def("set_orthogonality_centre", static_cast<void (_ttn::*)(size_t, double, size_t)>(&_ttn::set_orthogonality_centre), py::arg(), py::arg("tol") = 0, py::arg("nchi") = 0)
        .def("set_orthogonality_centre", static_cast<void (_ttn::*)(const std::list<size_t> &, double, size_t)>(&_ttn::set_orthogonality_centre), py::arg(), py::arg("tol") = 0, py::arg("nchi") = 0, "For details see :meth:`pyttn.ttn_dtype.set_orthogonality_centre`")
        .def("orthogonalise", &_ttn::orthogonalise, py::arg("force") = false, "For details see :meth:`pyttn.ttn_dtype.orthogonalise`")
        .def("truncate", &_ttn::truncate, py::arg("tol") = 0, py::arg("nchi") = 0, "For details see :meth:`pyttn.ttn_dtype.truncate`")

        .def("normalise", &_ttn::normalise, "For details see :meth:`pyttn.ttn_dtype.normalise`")
        .def("norm", &_ttn::norm, "For details see :meth:`pyttn.ttn_dtype.norm`")

        .def("__str__", [](const _ttn &o)
             {std::stringstream oss;   oss << o; return oss.str(); })

        .def("__setitem__", [](_ttn &i, size_t ind, const _ttn_node_data &o)
             { i[ind]() = o; }, "For details see :meth:`pyttn.ttn_dtype.__setitem__`")
        .def("__getitem__", [](_ttn &i, size_t ind) -> _ttn_node_data &
             { return i[ind](); }, py::return_value_policy::reference, "For details see :meth:`pyttn.ttn_dtype.resize`")
        .def("site_tensor", static_cast<const _ttn_node_data &(_ttn::*)(size_t) const>(&_ttn::site_tensor), py::return_value_policy::reference, "For details see :meth:`pyttn.ttn_dtype.site_tensor`")
#ifdef PYTTN_BUILD_CUDA
        .def("set_site_tensor", [](_ttn &self, size_t i, const linalg::matrix<T, otherbackend> &mat)
             { CALL_AND_HANDLE(self.site_tensor(i).as_matrix() = mat, "Failed to set site tensor."); })
#endif
        .def("set_site_tensor", [](_ttn &self, size_t i, const linalg::matrix<T, backend> &mat)
             { CALL_AND_HANDLE(self.site_tensor(i).as_matrix() = mat, "Failed to set site tensor."); })
        .def("set_site_tensor", [](_ttn &self, size_t i, py::buffer &mat)
             { CALL_AND_HANDLE(conv::copy_to_tensor(mat, self.site_tensor(i).as_matrix()), "Failed to set site tensor."); }, "For details see :meth:`pyttn.ttn_dtype.set_site_tensor`")

        .def("measure_without_collapse", [](_ttn &o, size_t i)
             {
                    std::vector<real_type> p0;
                    o.measure_without_collapse(i, p0);
                    return p0; }, "For details see :meth:`pyttn.ttn_dtype.measure_without_collapse`")
    //.def(
    //        "measure_all_without_collapse",
    //        [](_ttn& o)
    //        {
    //            std::vector<std::vector<real_type>> p0;
    //            o.measure_all_without_collapse(p0);
    //            return p0;
    //        }
    //    )
    //
#ifdef PYTTN_BUILD_CUDA
        .def("collapse_basis", [](_ttn &o, std::vector<linalg::matrix<T, otherbackend>> &U, bool truncate = true, real_type tol = real_type(0), size_t nchi = 0)
             {
                    std::vector<linalg::matrix<T, backend>> Uop(U.size());
                    for(size_t i = 0; i < U.size(); ++i)
                    {
                        Uop[i] = U[i];
                    }
                    std::vector<size_t> state;
                    real_type p = o.collapse_basis(Uop, state, truncate, tol, nchi); 
                    return std::make_pair(p, state); }, py::arg(), py::arg("truncate") = true, py::arg("tol") = real_type(0), py::arg("nchi") = 0)
#endif
        .def("collapse_basis", [](_ttn &o, std::vector<linalg::matrix<T, backend>> &U, bool truncate = true, real_type tol = real_type(0), size_t nchi = 0)
             {
                    std::vector<size_t> state;
                    real_type p = o.collapse_basis(U, state, truncate, tol, nchi); 
                    return std::make_pair(p, state); }, py::arg(), py::arg("truncate") = true, py::arg("tol") = real_type(0), py::arg("nchi") = 0)
        .def("collapse_basis", [](_ttn &o, std::vector<py::buffer> &U, bool truncate = true, real_type tol = real_type(0), size_t nchi = 0)
             {
                    std::vector<linalg::matrix<T, backend>> Uop(U.size());
                    for(size_t i = 0; i < U.size(); ++i)
                    {
                        conv::copy_to_tensor(U[i], Uop[i]);
                    }
                    std::vector<size_t> state;
                    real_type p = o.collapse_basis(Uop, state, truncate, tol, nchi); 
                    return std::make_pair(p, state); }, py::arg(), py::arg("truncate") = true, py::arg("tol") = real_type(0), py::arg("nchi") = 0, "For details see :meth:`pyttn.ttn_dtype.collapse_basis`")
        .def("collapse", [](_ttn &o, bool truncate = true, real_type tol = real_type(0), size_t nchi = 0)
             {
                    std::vector<size_t> state;
                    real_type p = o.collapse(state, truncate, tol, nchi); 
                    return std::make_pair(p, state); }, py::arg("truncate") = true, py::arg("tol") = real_type(0), py::arg("nchi") = 0, "For details see :meth:`pyttn.ttn_dtype.collapse`")

        .def("apply_one_body_operator", [](_ttn &o, const linalg::matrix<T, backend> &op, size_t index, bool shift_orthogonality)
             { CALL_AND_RETHROW(return o.apply_one_body_operator(op, index, shift_orthogonality)); }, py::arg(), py::arg(), py::arg("shift_orthogonality") = true)
        .def("apply_one_body_operator", [](_ttn &o, py::buffer &op, size_t index, bool shift_orthogonality)
             {
                    linalg::matrix<T, backend> mt;
                    conv::copy_to_tensor(op, mt);
                    CALL_AND_RETHROW(return o.apply_one_body_operator(mt, index, shift_orthogonality)); }, py::arg(), py::arg(), py::arg("shift_orthogonality") = true)
        .def("apply_one_body_operator", [](_ttn &o, siteop &op, bool shift_orthogonality)
             { CALL_AND_RETHROW(return o.apply_one_body_operator(op, shift_orthogonality)); }, py::arg(), py::arg("shift_orthogonality") = true)
        .def("apply_one_body_operator", [](_ttn &o, siteop &op, size_t index, bool shift_orthogonality)
             { CALL_AND_RETHROW(return o.apply_one_body_operator(op, index, shift_orthogonality)); }, py::arg(), py::arg(), py::arg("shift_orthogonality") = true, "For details see :meth:`pyttn.ttn_dtype.rapply_one_body_operator`")
        .def("apply_product_operator", [](_ttn &o, prodop &op, bool shift_orthogonality)
             { CALL_AND_RETHROW(return o.apply_product_operator(op, shift_orthogonality)); }, py::arg(), py::arg("shift_orthogonality") = true, "For details see :meth:`pyttn.ttn_dtype.apply_product_operator`")
        .def("apply_operator", [](_ttn &o, siteop &op, bool shift_orthogonality)
             { CALL_AND_RETHROW(return o.apply_operator(op, shift_orthogonality)); }, py::arg(), py::arg("shift_orthogonality") = true)
        .def("apply_operator", [](_ttn &o, prodop &op, bool shift_orthogonality)
             { CALL_AND_RETHROW(return o.apply_operator(op, shift_orthogonality)); }, py::arg(), py::arg("shift_orthogonality") = true, "For details see :meth:`pyttn.ttn_dtype.apply_operator`")

        .def("__imatmul__", [](_ttn &o, siteop &op)
             { CALL_AND_RETHROW(return o.apply_one_body_operator(op)); })
        .def("__imatmul__", [](_ttn &o, prodop &op)
             { CALL_AND_RETHROW(return o.apply_operator(op)); })
        .def("__imatmul__", [](_ttn &o, sop &op)
             {
                    _ttn i;
                    using contr = sop_ttn_contraction_engine<T, backend>;
                    CALL_AND_RETHROW(contr::sop_ttn_contraction(op, o, i, numpy_type(1)));
                    o = i;
                    return o; }, "For details see :meth:`pyttn.ttn_dtype.__imatmul__`")
        .def("__rmatmul__", [](const _ttn &o, siteop &op)
             {
                    _ttn i(o);
                    CALL_AND_RETHROW(return i.apply_one_body_operator(op)); })
        .def("__rmatmul__", [](const _ttn &o, prodop &op)
             {
                    _ttn i(o);
                    CALL_AND_RETHROW(return i.apply_operator(op)); })
        .def("__rmatmul__", [](const _ttn &o, sop &op)
             {
                    _ttn i;
                    using contr = sop_ttn_contraction_engine<T, backend>;
                    CALL_AND_RETHROW(contr::sop_ttn_contraction(op, o, i));
                    return i; }, "For details see :meth:`pyttn.ttn_dtype.__rmatmul__`")
        .def("backend", [](const _ttn &)
             { return backend::label(); })
        .doc() = R"mydelim(
          The pybind11 wrapper class for handling a general tree tensor network
          )mydelim";

    m.def("apply_sop_to_ttn", [](const sop &Op, const _ttn &A, _ttn &B, numpy_type coeff, real_type cutoff)
          {
                using cont_eng = sop_ttn_contraction_engine<T, backend>;
                CALL_AND_RETHROW(cont_eng::sop_ttn_contraction(Op, A, B, T(coeff), cutoff)); }, py::arg(), py::arg(), py::arg(), py::arg("coeff") = numpy_type(1), py::arg("cutoff") = real_type(1e-12));
}

template <typename real_type, typename backend>
void initialise_ttn(py::module &m)
{
    using complex_type = linalg::complex<real_type>;

#ifdef BUILD_REAL_TTN
    init_ttn<real_type, backend>(m, "real");
#endif
    init_ttn<complex_type, backend>(m, "complex");
}

#endif // PYTHON_BINDING_TTN_HPP
