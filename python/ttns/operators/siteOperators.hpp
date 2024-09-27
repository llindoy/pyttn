#ifndef PYTHON_BINDING_TTNS_SITE_OPERATORS_HPP
#define PYTHON_BINDING_TTNS_SITE_OPERATORS_HPP

#include <ttns_lib/operators/site_operators/matrix_operators.hpp>
#include <ttns_lib/operators/site_operators/direct_product_operator.hpp>
#include <ttns_lib/operators/site_operators/sequential_product_operator.hpp>
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
void init_site_operators(py::module &m, const std::string& label)
{
    using namespace ttns;
    using prim = ops::primitive<T, linalg::blas_backend>;
    using ident = ops::identity<T, linalg::blas_backend>;
    using dmat = ops::dense_matrix_operator<T, linalg::blas_backend>;
    using adjmat = ops::adjoint_dense_matrix_operator<T, linalg::blas_backend>;
    using spmat = ops::sparse_matrix_operator<T, linalg::blas_backend>;
    using diagmat = ops::diagonal_matrix_operator<T, linalg::blas_backend>;
    using dirprodop = ops::direct_product_operator<T, linalg::blas_backend>;

    using size_type = typename prim::size_type;
    using real_type = typename prim::real_type;

    using matrix_type = linalg::matrix<T, linalg::blas_backend>;
    using matrix_ref = typename prim::matrix_ref;
    using const_matrix_ref = typename prim::const_matrix_ref;
    using vector_ref = typename prim::vector_ref;
    using const_vector_ref = typename prim::const_vector_ref;

    using siteop = site_operator<T, linalg::blas_backend>;
    using opdict = operator_dictionary<T, linalg::blas_backend>;
    //the base primitive operator type
    py::class_<siteop>(m, (std::string("site_operator_")+label).c_str())
        .def(py::init())
        .def(py::init<const siteop&>())
        .def(py::init<const ident&>())
        .def(py::init<const dmat&>())
        .def(py::init<const adjmat&>())
        .def(py::init<const spmat&>())
        .def(py::init<const diagmat&>())
        .def(py::init<const dirprodop&>())

        .def(py::init<const ident&, size_t>())
        .def(py::init<const dmat&, size_t>())
        .def(py::init<const adjmat&, size_t>())
        .def(py::init<const spmat&, size_t>())
        .def(py::init<const diagmat&, size_t>())
        .def(py::init<const dirprodop&, size_t>())

        .def(py::init<sOP&, const system_modes&, bool>(), py::arg(), py::arg(), py::arg("use_sparse")=true)
        .def(py::init<sOP&, const system_modes&, const opdict&, bool>(), py::arg(), py::arg(), py::arg(), py::arg("use_sparse")=true)

        .def("initialise", static_cast<void (siteop::*)(sOP&, const system_modes&, bool)>(&siteop::initialise), py::arg(), py::arg(), py::arg("use_sparse")=true)
        .def("initialise", static_cast<void (siteop::*)(sOP&, const system_modes&, const opdict&, bool)>(&siteop::initialise), py::arg(), py::arg(), py::arg(), py::arg("use_sparse")=true)

        .def("complex_dtype", [](const siteop&){return !std::is_same<T, real_type>::value;})

        .def("assign", [](siteop& op, const siteop& o){return op=o;})
        .def("assign", [](siteop& op, const ident& o){return op=o;})
        .def("assign", [](siteop& op, const dmat& o){return op=o;})
        .def("assign", [](siteop& op, const adjmat& o){return op=o;})
        .def("assign", [](siteop& op, const spmat& o){return op=o;})
        .def("assign", [](siteop& op, const diagmat& o){return op=o;})
        .def("assign", [](siteop& op, const dirprodop& o){return op=o;})

        .def("bind", [](siteop& op, const ident& o){return op.bind(o);})
        .def("bind", [](siteop& op, const dmat& o){return op.bind(o);})
        .def("bind", [](siteop& op, const adjmat& o){return op.bind(o);})
        .def("bind", [](siteop& op, const spmat& o){return op.bind(o);})
        .def("bind", [](siteop& op, const diagmat& o){return op.bind(o);})
        .def("bind", [](siteop& op, const dirprodop& o){return op.bind(o);})

        .def("__copy__", [](const siteop& o){return siteop(o);})
        .def("__deepcopy__", [](const siteop& o, py::dict){return siteop(o);}, py::arg("memo"))
        .def("size", &siteop::size)
        .def("is_identity", &siteop::is_identity)
        .def("is_resizable", &siteop::is_resizable)
        .def_property(
            "mode", 
            static_cast<size_t (siteop::*)() const>(&siteop::mode),
            [](siteop& o, size_t val){o.mode() = val;}
         )

        .def("resize", &siteop::resize)
        .def(
                "apply", 
                static_cast<void (siteop::*)(const_matrix_ref, matrix_ref)>(&siteop::apply)
            )
        .def(
                "apply", 
                static_cast<void (siteop::*)(const_matrix_ref, matrix_ref, real_type, real_type)>(&siteop::apply)
            )
        .def(
                "apply", 
                static_cast<void (siteop::*)(const_vector_ref, vector_ref)>(&siteop::apply)
            )
        .def(
                "apply", 
                static_cast<void (siteop::*)(const_vector_ref, vector_ref, real_type, real_type)>(&siteop::apply)
            )
        .def("__str__", &siteop::to_string);

    //the base primitive operator type
    py::class_<prim>(m, (std::string("primitive_")+label).c_str())
        .def("size", &prim::size)
        .def("size", &prim::size)
        .def("size", &prim::size)
        .def("is_identity", &prim::is_identity)
        .def("is_resizable", &prim::is_resizable)
        .def("resize", &prim::resize)
        .def("clone", &prim::clone)
        .def("complex_dtype", [](const prim&){return !std::is_same<T, real_type>::value;})
        .def("__str__", &prim::to_string);

    //a type for storing a trivial representation of the identity operator
    py::class_<ident, prim>(m, (std::string("identity_")+label).c_str())
        .def(py::init())
        .def(py::init<size_type>())
        .def("complex_dtype", [](const ident&){return !std::is_same<T, real_type>::value;});

    //a dense matrix representation of the operator
    py::class_<dmat, prim>(m, (std::string("matrix_")+label).c_str())
        .def(py::init())
        .def(py::init<matrix_type>())
        .def(py::init([](py::buffer& b)
                {
                    linalg::matrix<T, linalg::blas_backend> mat;
                    copy_pybuffer_to_tensor(b, mat);
                    return dmat(mat);
                }
            )
        )
        .def("complex_dtype", [](const dmat&){return !std::is_same<T, real_type>::value;})
        .def("matrix", &dmat::mat);

    //a dense matrix representation of the adjoint of an operator
    py::class_<adjmat, prim>(m, (std::string("adjoint_matrix_")+label).c_str())
        .def(py::init())
        .def(py::init<matrix_type>())
        .def(py::init([](py::buffer& b)
                {
                    linalg::matrix<T, linalg::blas_backend> mat;
                    copy_pybuffer_to_tensor(b, mat);
                    return adjmat(mat);
                }
            )
        )
        .def("complex_dtype", [](const adjmat&){return !std::is_same<T, real_type>::value;})
        .def("matrix", &adjmat::mat);

    //a csr matrix representation of an operator
    using csr_type = linalg::csr_matrix<T, linalg::blas_backend>;
    using index_type = typename csr_type::index_type;
    py::class_<spmat, prim>(m, (std::string("sparse_matrix_")+label).c_str())
        .def(py::init())
        .def(py::init<const std::vector<T>&, const std::vector<index_type>&, const std::vector<index_type>&, size_t>(), py::arg(), py::arg(), py::arg(), py::arg("ncols")=0)
        .def(py::init<const csr_type&>())
        .def("complex_dtype", [](const spmat&){return !std::is_same<T, real_type>::value;})
        .def("matrix", &spmat::mat);

    //a diagonal matrix representation of an operator
    using diag_type = linalg::diagonal_matrix<T, linalg::blas_backend>;
    py::class_<diagmat, prim>(m, (std::string("diagonal_matrix_")+label).c_str())
        .def(py::init())
        .def(py::init<const diag_type&>())        
        .def(py::init([](py::buffer& b)
                {
                    diag_type mat;
                    copy_pybuffer_to_diagonal_matrix(b, mat);
                    return diagmat(mat);
                }
            )
        )
        .def(py::init<const std::vector<T>&>())
        .def(py::init<const std::vector<T>&, size_t>())
        .def(py::init<const std::vector<T>&, size_t, size_t>())
        .def(py::init<const linalg::tensor<T, 1>&>())
        .def(py::init<const linalg::tensor<T, 1>&, size_t>())
        .def(py::init<const linalg::tensor<T, 1>&, size_t, size_t>())
        .def("complex_dtype", [](const diagmat&){return !std::is_same<T, real_type>::value;})
        .def("matrix", &diagmat::mat);

    //a special operator form for a direct product operator
    py::class_<dirprodop, prim>(m, (std::string("direct_product_")+label).c_str())
        .def(py::init())
        .def("complex_dtype", [](const dirprodop&){return !std::is_same<T, real_type>::value;});

    //a special operator form for a sequential product operator
    //using seqprodop = ops::sequential_product_operator<T, linalg::blas_backend>;
    //py::class_<seqprodop, prim>(m, (std::string("sequential_product_")+label).c_str())
    //    .def(py::init())
    //    .def("append", &seqprodop::append_operator<diagmat>)
    //    .def("append", &seqprodop::append_operator<dmat>)
    //    .def("append", &seqprodop::append_operator<adjmat>)
    //    .def("append", &seqprodop::append_operator<spmat>)
    //    .def("append", &seqprodop::append_operator<commmat>)
    //    .def("append", &seqprodop::append_operator<anti_commmat>)
    //    .def("append", &seqprodop::append_operator<dvrop>)
    //    .def("append", &seqprodop::append_operator<dirprodop>);
}

void initialise_site_operators(py::module& m);

#endif  //PYTHON_BINDING_TTNS_SITE_OPERATORS_HPP


