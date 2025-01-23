#ifndef PYTHON_BINDING_STRING_SOP_HPP
#define PYTHON_BINDING_STRING_SOP_HPP

#include <ttns_lib/sop/coeff_type.hpp>
#include <ttns_lib/sop/sSOP.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>


namespace py=pybind11;

template <typename real_type>
void init_sSOP(py::module& m)
{
    using complex_type = linalg::complex<real_type>;
    using namespace ttns;
    using namespace literal;
    //wrapper for the sOP type 
    py::class_<sOP>(m, "sOP")
        .def(py::init())
        .def(py::init<const std::string&, size_t>())
        .def(py::init<const std::string&, size_t, bool>())
        .def(py::init<const sOP&>())
        .def("clear", &sOP::clear, "Clear the sOPs mode and label information.")
        .def("assign", [](sOP& self, const sOP& o){self=o;})
        .def("__copy__",[](const sOP& o){return sOP(o);})
        .def("__deepcopy__", [](const sOP& o, py::dict){return sOP(o);}, py::arg("memo"))
        .def_property
            (
                "op", 
                static_cast<const std::string& (sOP::*)() const>(&sOP::op),
                [](sOP& o, const std::string& i){o.op() = i;},
                "The label of the operator"
            )
        .def_property
            (
                "mode", 
                static_cast<const size_t& (sOP::*)() const>(&sOP::mode),
                [](sOP& o, const size_t& i){o.mode() = i;},
                "The mode the operator acts on"
            )

        .def("__str__", [](const sOP& o){return static_cast<std::string>(o);})

        .def("__add__", [](const sOP& a, const sOP& b){return a+b;})
        .def("__add__", [](const sOP& a, const sPOP& b){return a+b;})
        .def("__add__", [](const sOP& a, const sNBO<real_type>& b){return a+b;})
        .def("__add__", [](const sOP& a, const sNBO<complex_type>& b){return a+b;})
        .def("__add__", [](const sOP& a, const sSOP<real_type>&  b){return a+b;})
        .def("__add__", [](const sOP& a, const sSOP<complex_type>& b){return a+b;}, R"mydelim(
              Functions for adding an sOP object from the left to any other spin operator to form a sSOP_real or sSOP_complex
              depending on the dtype used.  

              :Parameters:  - **a** (:class:`sOP`) - The left term in the expression
                            - **b** (class:`sOP` or :class:`sPOP` or :class:`sNBO_real` or :class:`sNBO_complex` or :class:`sSOP_real` or :class:`sSOP_complex`) - The right term in the expression

              :returns: The result of the sum
              :rtype: :class:`sSOP_real` or :class:`sSOP_complex`


              )mydelim")

        .def("__sub__", [](const sOP& a, const sOP& b){return a-b;})
        .def("__sub__", [](const sOP& a, const sPOP& b){return a-b;})
        .def("__sub__", [](const sOP& a, const sNBO<real_type>& b){return a-b;})
        .def("__sub__", [](const sOP& a, const sNBO<complex_type>& b){return a-b;})
        .def("__sub__", [](const sOP& a, const sSOP<real_type>& b){return a-b;})
        .def("__sub__", [](const sOP& a, const sSOP<complex_type>& b){return a-b;}, R"mydelim(
              Functions for subtracting a string operator from an sOP object to form a sSOP_real or sSOP_complex
              depending on the dtype used.  


              :Parameters:  - **a** (:class:`sOP`) - The left term in the expression sOP
                            - **b** (class:`sOP` or :class:`sPOP` or :class:`sNBO_real` or :class:`sNBO_complex` or :class:`sSOP_real` or :class:`sSOP_complex`) - The right term in the expression

              :returns: The result of a-b
              :rtype: :class:`sSOP_real` or :class:`sSOP_complex`



              )mydelim")

        .def("__div__", [](const sOP& a, const real_type& b){return a*(1.0/b);})
        .def("__div__", [](const sOP& a, const complex_type& b){return a*(1.0/b);}, R"mydelim(
              Functions for dividing a sOP by a scalar .

              :Parameters:  - **a** (:class:`sOP`) - The left term in the expression
                            - **b** (float or complex ) - The scalar to divide the sOP by

              :returns: The result of a/b
              :rtype: :class:`sNBO_real` or :class:`sNBO_complex`

              )mydelim")

        .def("__mul__", [](const sOP& a, const coeff<real_type>& b){return a*b;})
        .def("__mul__", [](const sOP& a, const coeff<complex_type>& b){return a*b;})
        .def("__mul__", [](const sOP& a, const real_type& b){return a*b;})
        .def("__mul__", [](const sOP& a, const complex_type& b){return a*b;})
        .def("__mul__", [](const sOP& a, const sOP& b){return a*b;})
        .def("__mul__", [](const sOP& a, const sPOP& b){return a*b;})
        .def("__mul__", [](const sOP& a, const sNBO<real_type>& b){return a*b;})
        .def("__mul__", [](const sOP& a, const sNBO<complex_type>& b){return a*b;}, R"mydelim(
              Functions for multiplying a sOP by a scalar or another expression.

              :Parameters:  - **a** (:class:`sOP`) - The left term in the expression
                            - **b** (float or complex or :class:`coeff_real` or :class:`coeff_complex` or :class:`sOP` or :class:`sPOP` or :class:`sNBO_real` or :class:`sNBO_complex`) - The term to multiply sOP by
              :type b: 

              :returns: The result of a*b
              :rtype: :class:`sNBO_real` or :class:`sNBO_complex`

              )mydelim")


        .def("__mul__", [](const sOP& a, const sSOP<real_type>& b){return a*b;})
        .def("__mul__", [](const sOP& a, const sSOP<complex_type>& b){return a*b;}, R"mydelim(
              Functions for multiplying a sOP by a sSOP.

              :Parameters:  - **a** (:class:`sOP`) - The left term in the expression
                            - **b** (:class:`sSOP_real` or :class:`sSOP_complex`) - The term to multiply sOP by

              :Returns: The result of a*b
              :Return Type: :class:`sSOP_real` or :class:`sSOP_complex`

              )mydelim")

        .def("__rmul__", [](const sOP& a, const coeff<real_type>& b){return a*b;})
        .def("__rmul__", [](const sOP& a, const coeff<complex_type>& b){return a*b;})
        .def("__rmul__", [](const sOP& a, const real_type& b){return a*b;})
        .def("__rmul__", [](const sOP& a, const complex_type& b){return a*b;}, R"mydelim(
              Functions for multiplying a sOP by a scalar from the right.

              :Parameters:  - **a** (:class:`sOP`) - The sOP
                            - **b** (float or complex or :class:`coeff_real` or :class:`coeff_complex`) - The term to multiply sOP by

              :Returns: The result of b*a
              :Return Type: :class:`sNBO_real` or :class:`sNBO_complex`

              )mydelim")
        .doc() = R"mydelim(
            The single site operator used for the string operator handling functionality of pyTTN.  This class allows for definition of 
            a string label for an operator and the mode that the operator acts upon. In addition to allowing for arbitrary string labels
            with the combination of user defined operator dictionaries.  This code supports several automatic dictionaries depending on
            the type of mode considered.  These are

            :Fermion Modes:
              - Annihilation operator :math:`\hat{c}` :  {"c", "a", "f"}
              - Creation operator :math:`\hat{c}^\dagger` :  {"cdag", "adag", "fdag", "cd", "ad", "fd"}
              - Number operator :math:`\hat{c}^\dagger\hat{c}` :  {"n", "cdagc", "adaga", "fdagf", "cdc", "ada", "fdf"}
              - Vacancy operator :math:`1-\hat{c}^\dagger\hat{c}` :  "v"

            :Bosonic Modes:
              - Annihilation operator :math:`\hat{c}` :  {"c", "a", "b"}
              - Creation operator :math:`\hat{c}^\dagger` :  {"cdag", "adag", "bdag", "cd", "ad", "bd"}
              - Number operator :math:`\hat{c}^\dagger\hat{c}` :  {"n", "cdagc", "adaga", "bdagb", "cdc", "ada", "bdb"}
              - Position operator :math:`hat{q}` : {"q", "x"}
              - Momentum opeartor :math:`hat{p}` : "p"

            :Spin Modes for arbitrary spin S:
              - :math:`\hat{S}_x` : {"sx", "x"}
              - :math:`\hat{S}_y` : {"sy", "y"}
              - :math:`\hat{S}_z` : {"sz", "z"}
              - :math:`\hat{S}_+` : {"s+", "sp"}
              - :math:`\hat{S}_-` : {"s-", "sm"}

            :Two Level System Modes:
              - :math:`\hat{\sigma}_x` : {"sx", "x", "sigmax"}
              - :math:`\hat{\sigma}_y` : {"sy", "y", "sigmay"}
              - :math:`\hat{\sigma}_z` : {"sz", "z", "sigmaz"}
              - :math:`\hat{\sigma}_+` : {"s+", "sp", "sigma+", "sigmap"}
              - :math:`\hat{\sigma}_-` : {"s-", "sm", "sigma-", "sigmam"}
        )mydelim";



    m.def("fermion_operator", &fermion_operator, R"mydelim(
      Create a new site operator string where the operator is a Fermionic operator.

      :param arg0: The operator label associated with this operator.   
      :type arg0: str
      :param arg1: The mode this operator acts upon.
      :type arg1: int

      For fermionic systems the following operator are supported:
        - Annihilation operator :math:`\hat{c}` :  {"c", "a", "f"}
        - Creation operator :math:`\hat{c}^\dagger` :  {"cdag", "adag", "fdag", "cd", "ad", "fd"}
        - Number operator :math:`\hat{c}^\dagger\hat{c}` :  {"n", "cdagc", "adaga", "fdagf", "cdc", "ada", "fdf"}
        - Vacancy operator :math:`1-\hat{c}^\dagger\hat{c}` :  {"v"}

      :returns: fermionic mode data object
      :rtype: mode_data
      )mydelim");

    m.def("fOP", &fermion_operator, R"mydelim(
      Create a new site operator string where the operator is a Fermionic operator.

      :param arg0: The operator label associated with this operator.   
      :type arg0: str
      :param arg1: The mode this operator acts upon.
      :type arg1: int

      For fermionic systems the following operator are supported:
        - Annihilation operator :math:`\hat{c}` :  {"c", "a", "f"}
        - Creation operator :math:`\hat{c}^\dagger` :  {"cdag", "adag", "fdag", "cd", "ad", "fd"}
        - Number operator :math:`\hat{c}^\dagger\hat{c}` :  {"n", "cdagc", "adaga", "fdagf", "cdc", "ada", "fdf"}
        - Vacancy operator :math:`1-\hat{c}^\dagger\hat{c}` :  {"v"}

      :returns: fermionic mode data object
      :rtype: mode_data
      )mydelim");

    //wrapper for the sPOP type 
    py::class_<sPOP>(m, "sPOP")
        .def(py::init())
        .def(py::init<const sOP&>())
        .def(py::init<const std::list<sOP>&>())
        .def(py::init<const sPOP&>())
        .def("assign", [](sPOP& self, const sPOP& o){self=o;})
        .def("__copy__",[](const sPOP& o){return sPOP(o);})
        .def("__deepcopy__", [](const sPOP& o, py::dict){return sPOP(o);}, py::arg("memo"))
        .def("clear", &sPOP::clear, "empty the internal buffer storing the sOPs in the sPOP")
        .def("insert_front", &sPOP::append, R"mydelim(

            :param o: Insert a sOP object at the front of this product.
            :type o: sOP
        )mydelim")
        .def("insert_back", &sPOP::prepend, R"mydelim(

            :param o: Insert a sOP object at the back of this product.
            :type o: sOP
        )mydelim")
        .def("size", &sPOP::size, R"mydelim(

            :returns: The number of individual sOP terms in this product.
            :rtype: int
        )mydelim")
        .def("nmodes", &sPOP::nmodes, R"mydelim(

            :returns: The number of modes that this sPOP acts on
            :rtype: int
        )mydelim")
        .def_property
            (
                "ops", 
                static_cast<const std::list<sOP>& (sPOP::*)() const>(&sPOP::ops),
                [](sPOP& o, const std::list<sOP>& i){o.ops() = i;},
                "A list of the individual sOP objects forming the sPOP."
            )

        .def(
                "__iter__",
                [](sPOP& s){return py::make_iterator(s.begin(), s.end());},
                py::keep_alive<0, 1>()
            )
        .def("__str__", [](const sPOP& o){return static_cast<std::string>(o);})
        .def("__imul__", [](sPOP& a, const sPOP& b){return a*=b;})
        .def("__imul__", [](sPOP& a, const sOP& b){return a*=b;}, R"mydelim(
              Inplace multiplication of a sPOP with a sPOP or sOP

              :Parameters:  - **b** (class:`sOP` or :class:`sPOP`) - The right term in the expression

              )mydelim")

        .def("__add__", [](const sPOP& a, const sOP& b){return a+b;})
        .def("__add__", [](const sPOP& a, const sPOP& b){return a+b;})
        .def("__add__", [](const sPOP& a, const sNBO<real_type>& b){return a+b;})
        .def("__add__", [](const sPOP& a, const sNBO<complex_type>& b){return a+b;})
        .def("__add__", [](const sPOP& a, const sSOP<real_type>&  b){return a+b;})
        .def("__add__", [](const sPOP& a, const sSOP<complex_type>& b){return a+b;}, R"mydelim(
              Functions for adding an sPOP object from the left to any other spin operator to form a sSOP_real or sSOP_complex
              depending on the dtype used.  

              :Parameters:  - **a** (:class:`sPOP`) - The sPOP
                            - **b** (class:`sOP` or :class:`sPOP` or :class:`sNBO_real` or :class:`sNBO_complex` or :class:`sSOP_real` or :class:`sSOP_complex`) - The right term in the expression

              :returns: The result of the sum
              :rtype: :class:`sSOP_real` or :class:`sSOP_complex`

              )mydelim")

        .def("__sub__", [](const sPOP& a, const sOP& b){return a-b;})
        .def("__sub__", [](const sPOP& a, const sPOP& b){return a-b;})
        .def("__sub__", [](const sPOP& a, const sNBO<real_type>& b){return a-b;})
        .def("__sub__", [](const sPOP& a, const sNBO<complex_type>& b){return a-b;})
        .def("__sub__", [](const sPOP& a, const sSOP<real_type>& b){return a-b;})
        .def("__sub__", [](const sPOP& a, const sSOP<complex_type>& b){return a-b;}, R"mydelim(
              Functions for subtracting a string operator from an sPOP object to form a sSOP_real or sSOP_complex
              depending on the dtype used.  

              :Parameters:  - **a** (:class:`sPOP`) - The sPOP
                            - **b** (class:`sOP` or :class:`sPOP` or :class:`sNBO_real` or :class:`sNBO_complex` or :class:`sSOP_real` or :class:`sSOP_complex`) - The right term in the expression

              :returns: The result of a-b
              :rtype: :class:`sSOP_real` or :class:`sSOP_complex`

              )mydelim")

        .def("__div__", [](const sOP& a, const real_type& b){return a*(1.0/b);})
        .def("__div__", [](const sOP& a, const complex_type& b){return a*(1.0/b);}, R"mydelim(
              Functions for dividing a sPOP by a scalar .

              :Parameters:  - **a** (:class:`sPOP`) - The sPOP
                            - **b** (float or complex ) - The scalar to divide the sPOP by

              :returns: The result of a/b
              :rtype: :class:`sNBO_real` or :class:`sNBO_complex`

              )mydelim")

        .def("__mul__", [](const sPOP& a, const coeff<real_type>& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const coeff<complex_type>& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const real_type& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const complex_type& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const sOP& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const sPOP& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const sNBO<real_type>& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const sNBO<complex_type>& b){return a*b;}, R"mydelim(
              Functions for multiplying a sPOP by a scalar or another expression.

              :Parameters:  - **a** (:class:`sPOP`) - The sPOP
                            - **b** (float or complex or :class:`coeff_real` or :class:`coeff_complex` or :class:`sOP` or :class:`sPOP` or :class:`sNBO_real` or :class:`sNBO_complex`) - The term to multiply sPOP by
              :type b: 

              :returns: The result of a*b
              :rtype: :class:`sNBO_real` or :class:`sNBO_complex`

              )mydelim")
        .def("__mul__", [](const sPOP& a, const sSOP<real_type>& b){return a*b;})
        .def("__mul__", [](const sPOP& a, const sSOP<complex_type>& b){return a*b;}, R"mydelim(
              Functions for multiplying a sPOP by a sSOP.

              :Parameters:  - **a** (:class:`sPOP`) - The sPOP
                            - **b** (:class:`sSOP_real` or :class:`sSOP_complex`) - The term to multiply sPOP by

              :Returns: The result of a*b
              :Return Type: :class:`sSOP_real` or :class:`sSOP_complex`

              )mydelim")

        .def("__rmul__", [](const sPOP& a, const coeff<real_type>& b){return a*b;})
        .def("__rmul__", [](const sPOP& a, const coeff<complex_type>& b){return a*b;})
        .def("__rmul__", [](const sPOP& a, const real_type& b){return a*b;})
        .def("__rmul__", [](const sPOP& a, const complex_type& b){return a*b;}, R"mydelim(
              Functions for multiplying a sPOP by a scalar from the right.

              :Parameters:  - **a** (:class:`sPOP`) - The sPOP
                            - **b** (float or complex or :class:`coeff_real` or :class:`coeff_complex`) - The term to multiply sPOP by

              :Returns: The result of b*a
              :Return Type: :class:`sNBO_real` or :class:`sNBO_complex`

              )mydelim")
        .doc() = R"mydelim(
            A product of sOP operators. This function handles the fact that in general sOP operators do not commute.

        )mydelim";

    {
        using coef = coeff<real_type>;
        //using complex_func_type = std::function<complex_type(real_type)>;
        using func_type = std::function<real_type(real_type)>;
        py::class_<coef>(m, "coeff_real")
            .def(py::init())
            .def(py::init<const real_type&>())
            .def(py::init<const func_type&>())
            .def(py::init<const coef&>())
            .def("assign", [](coef& self, const coef& o){self=o;})
            .def("assign", [](coef& self, const real_type& o){self=o;})
            .def("assign", [](coef& self, const func_type& o){self=o;})
            .def("__copy__",[](const coef& o){return coef(o);})
            .def("__deepcopy__", [](const coef& o, py::dict){return coef(o);}, py::arg("memo"))
            .def("clear", &coef::clear)
            .def("is_zero", &coef::is_zero, py::arg("tol")=real_type(1e-14))
            .def("is_positive", &coef::is_positive)
            .def("is_time_dependent", &coef::is_time_dependent)
            .def("__call__", &coef::operator())

            .def("__str__", [](const coef& o){std::ostringstream oss; oss << o; return oss.str();})
            .def("__iadd__", [](coef& a, const real_type& b){return a+=b;})
            .def("__iadd__", [](coef& a, const coef& b){return a+=b;})

            .def("__isub__", [](coef& a, const real_type& b){return a-=b;})
            .def("__isub__", [](coef& a, const coef& b){return a-=b;})

            .def("__imul__", [](coef& a, const real_type& b){return a*=b;})
            .def("__imul__", [](coef& a, const coef& b){return a*=b;})

            .def("__idiv__", [](coef& a, const real_type& b){return a/=b;})


            .def("__div__", [](const coef& a, const real_type& b){return a/b;})
            .def("__div__", [](const coef& a, const complex_type& b){return a/b;})

            .def("__add__", [](coef& a, const coef& b){return a+b;})
            .def("__sub__", [](coef& a, const coef& b){return a-b;})
            .def("__mul__", [](coef& a, const coef& b){return a*b;})
            .def("__add__", [](coef& a, const coeff<complex_type>& b){return a+b;})
            .def("__sub__", [](coef& a, const coeff<complex_type>& b){return a-b;})
            .def("__mul__", [](coef& a, const coeff<complex_type>& b){return a*b;})
            .def("__mul__", [](coef& a, const sOP& b){return a*b;})
            .def("__mul__", [](coef& a, const sPOP& b){return a*b;})
            .def("__mul__", [](coef& a, const sNBO<real_type>& b){return a*b;})
            .def("__mul__", [](coef& a, const sNBO<complex_type>& b){return a*b;})
            .def("__mul__", [](coef& a, const sSOP<real_type>& b){return a*b;})
            .def("__mul__", [](coef& a, const sSOP<complex_type>& b){return a*b;})

            .def("__rmul__", [](const coef& a, const real_type& b){return a*b;})
            .def("__rmul__", [](const coef& a, const complex_type& b){return a*b;});
    }
      
    {
        using coef = coeff<complex_type>;
        using func_type = std::function<complex_type(real_type)>;
        using real_func_type = std::function<real_type(real_type)>;
        py::class_<coef>(m, "coeff_complex")
            .def(py::init())
            .def(py::init<const real_type&>())
            .def(py::init<const complex_type&>())
            .def(py::init<const func_type&>())
            .def(py::init<const real_func_type&>())
            .def(py::init<const coef&>())
            .def(py::init<const coeff<real_type>&>())
            .def("assign", [](coef& self, const coef& o){self=o;})
            .def("assign", [](coef& self, const coeff<real_type>& o){self=o;})
            .def("assign", [](coef& self, const real_type& o){self=o;})
            .def("assign", [](coef& self, const complex_type& o){self=o;})
            .def("assign", [](coef& self, const func_type& o){self=o;})
            .def("assign", [](coef& self, const real_func_type& o){self=coeff<real_type>(o);})
            .def("__copy__",[](const coef& o){return coef(o);})
            .def("__deepcopy__", [](const coef& o, py::dict){return coef(o);}, py::arg("memo"))
            .def("clear", &coef::clear)
            .def("is_zero", &coef::is_zero, py::arg("tol")=real_type(1e-14))
            .def("is_positive", &coef::is_positive)
            .def("is_time_dependent", &coef::is_time_dependent)
            .def("__call__", &coef::operator())

            .def("__str__", [](const coef& o){std::ostringstream oss; oss << o; return oss.str();})
            .def("__iadd__", [](coef& a, const real_type& b){return a+=b;})
            .def("__isub__", [](coef& a, const real_type& b){return a-=b;})
            .def("__imul__", [](coef& a, const real_type& b){return a*=b;})
            .def("__idiv__", [](coef& a, const real_type& b){return a/=b;})

            .def("__iadd__", [](coef& a, const complex_type& b){return a+=b;})
            .def("__isub__", [](coef& a, const complex_type& b){return a-=b;})
            .def("__imul__", [](coef& a, const complex_type& b){return a*=b;})
            .def("__idiv__", [](coef& a, const complex_type& b){return a/=b;})

            .def("__iadd__", [](coef& a, const coef& b){return a+=b;})
            .def("__isub__", [](coef& a, const coef& b){return a-=b;})
            .def("__imul__", [](coef& a, const coef& b){return a*=b;})

            .def("__iadd__", [](coef& a, const coeff<real_type>& b){return a+=b;})
            .def("__isub__", [](coef& a, const coeff<real_type>& b){return a-=b;})
            .def("__imul__", [](coef& a, const coeff<real_type>& b){return a*=b;})

            .def("__div__", [](const coef& a, const real_type& b){return a/b;})
            .def("__div__", [](const coef& a, const complex_type& b){return a/b;})

            .def("__add__", [](const coef& a, const coef& b){return a+b;})
            .def("__sub__", [](const coef& a, const coef& b){return a-b;})
            .def("__mul__", [](const coef& a, const coef& b){return a*b;})
            .def("__add__", [](const coef& a, const coeff<real_type>& b){return a+b;})
            .def("__sub__", [](const coef& a, const coeff<real_type>& b){return a-b;})
            .def("__mul__", [](const coef& a, const coeff<real_type>& b){return a*b;})
            .def("__mul__", [](const coef& a, const sOP& b){return a*b;})
            .def("__mul__", [](const coef& a, const sPOP& b){return a*b;})
            .def("__mul__", [](const coef& a, const sNBO<real_type>& b){return a*b;})
            .def("__mul__", [](const coef& a, const sNBO<complex_type>& b){return a*b;})
            .def("__mul__", [](const coef& a, const sSOP<real_type>& b){return a*b;})
            .def("__mul__", [](const coef& a, const sSOP<complex_type>& b){return a*b;})

            .def("__rmul__", [](const coef& a, const real_type& b){return a*b;})
            .def("__rmul__", [](const coef& a, const complex_type& b){return a*b;});
    }

    {
        using NBO = sNBO<real_type>;
        //wrapper for the sPOP type 
        py::class_<NBO>(m, "sNBO_real")
            .def(py::init())
            .def(py::init<const sOP&>())
            .def(py::init<const sPOP&>())
            .def(py::init<const real_type&, const sPOP&>())
            .def(py::init<const real_type&, const sOP&>())
            .def(py::init<const coeff<real_type>&, const sPOP&>())
            .def(py::init<const coeff<real_type>&, const sOP&>())
            .def(py::init<const NBO&>())
            .def("assign", [](NBO& self, const NBO& o){self=o;})
            .def("__copy__",[](const NBO& o){return NBO(o);})
            .def("__deepcopy__", [](const NBO& o, py::dict){return NBO(o);}, py::arg("memo"))
            .def("clear", &NBO::clear)
            .def("insert_front", &NBO::append)
            .def("insert_back", &NBO::prepend)
            .def("nmodes", &NBO::nmodes)
            .def(
                    "__iter__",
                    [](NBO& s){return py::make_iterator(s.begin(), s.end());},
                    py::keep_alive<0, 1>()
                )
            .def_property
                (
                    "coeff", 
                    static_cast<const coeff<real_type>& (NBO::*)() const>(&NBO::coeff),
                    [](NBO& o, const coeff<real_type>& i){o.coeff() = i;}
                )
            .def_property
                (
                    "ops", 
                    static_cast<const std::list<sOP>& (NBO::*)() const>(&NBO::ops),
                    [](NBO& o, const std::list<sOP>& i){o.ops() = i;}
                )
            .def_property
                (
                    "pop", 
                    static_cast<const sPOP& (NBO::*)() const>(&NBO::pop),
                    [](NBO& o, const sPOP& i){o.pop() = i;}
                )

            .def("__str__", [](const NBO& o){return static_cast<std::string>(o);})
            .def("__imul__", [](NBO& a, const sOP& b){return a*=b;})
            .def("__imul__", [](NBO& a, const sPOP& b){return a*=b;})
            .def("__imul__", [](NBO& a, const NBO& b){return a*=b;})
            .def("__imul__", [](NBO& a, const real_type& b){return a*=b;})

            .def("__add__", [](const NBO& a, const sOP& b){return a+b;})
            .def("__add__", [](const NBO& a, const sPOP& b){return a+b;})
            .def("__add__", [](const NBO& a, const sNBO<real_type>& b){return a+b;})
            .def("__add__", [](const NBO& a, const sNBO<complex_type>& b){return a+b;})
            .def("__add__", [](const NBO& a, const sSOP<real_type>&  b){return a+b;})
            .def("__add__", [](const NBO& a, const sSOP<complex_type>& b){return a+b;})

            .def("__sub__", [](const NBO& a, const sOP& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sPOP& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sNBO<real_type>& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sNBO<complex_type>& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sSOP<real_type>& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sSOP<complex_type>& b){return a-b;})

            .def("__div__", [](const NBO& a, const real_type& b){return a*(1.0/b);})
            .def("__div__", [](const NBO& a, const complex_type& b){return a*(1.0/b);})

            .def("__mul__", [](const NBO& a, const coeff<real_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const coeff<complex_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const real_type& b){return a*b;})
            .def("__mul__", [](const NBO& a, const complex_type& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sOP& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sPOP& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sNBO<real_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sNBO<complex_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sSOP<real_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sSOP<complex_type>& b){return a*b;})

            .def("__rmul__", [](const NBO& a, const coeff<real_type>& b){return a*b;})
            .def("__rmul__", [](const NBO& a, const coeff<complex_type>& b){return a*b;})
            .def("__rmul__", [](const NBO& a, const real_type& b){return a*b;})
            .def("__rmul__", [](const NBO& a, const complex_type& b){return a*b;});
    }
    {
        using NBO = sNBO<complex_type>;
        //wrapper for the sPOP type 
        py::class_<NBO>(m, "sNBO_complex")
            .def(py::init())
            .def(py::init<const sOP&>())
            .def(py::init<const sPOP&>())
            .def(py::init<const complex_type&, const sPOP&>())
            .def(py::init<const complex_type&, const sOP&>())
            .def(py::init<const coeff<complex_type>&, const sPOP&>())
            .def(py::init<const coeff<complex_type>&, const sOP&>())
            .def(py::init<const NBO&>())
            .def(py::init<const sNBO<real_type>&>())
            .def("assign", [](NBO& self, const NBO& o){self=o;})
            .def("assign", [](NBO& self, const sNBO<real_type>& o){self=o;})
            .def("__copy__",[](const NBO& o){return NBO(o);})
            .def("__deepcopy__", [](const NBO& o, py::dict){return NBO(o);}, py::arg("memo"))
            .def("clear", &NBO::clear)
            .def("insert_front", &NBO::append)
            .def("insert_back", &NBO::prepend)
            .def("nmodes", &NBO::nmodes)
            .def(
                    "__iter__",
                    [](NBO& s){return py::make_iterator(s.begin(), s.end());},
                    py::keep_alive<0, 1>()
                )
            .def_property
                (
                    "coeff", 
                    static_cast<const coeff<complex_type>& (NBO::*)() const>(&NBO::coeff),
                    [](NBO& o, const coeff<complex_type>& i){o.coeff() = i;}
                )
            .def_property
                (
                    "ops", 
                    static_cast<const std::list<sOP>& (NBO::*)() const>(&NBO::ops),
                    [](NBO& o, const std::list<sOP>& i){o.ops() = i;}
                )
            .def_property
                (
                    "pop", 
                    static_cast<const sPOP& (NBO::*)() const>(&NBO::pop),
                    [](NBO& o, const sPOP& i){o.pop() = i;}
                )

            .def("__str__", [](const NBO& o){return static_cast<std::string>(o);})
            .def("__imul__", [](NBO& a, const sOP& b){return a*=b;})
            .def("__imul__", [](NBO& a, const sPOP& b){return a*=b;})
            .def("__imul__", [](NBO& a, const NBO& b){return a*=b;})
            .def("__imul__", [](NBO& a, const complex_type& b){return a*=b;})

            .def("__add__", [](const NBO& a, const sOP& b){return a+b;})
            .def("__add__", [](const NBO& a, const sPOP& b){return a+b;})
            .def("__add__", [](const NBO& a, const sNBO<real_type>& b){return a+b;})
            .def("__add__", [](const NBO& a, const sNBO<complex_type>& b){return a+b;})
            .def("__add__", [](const NBO& a, const sSOP<real_type>&  b){return a+b;})
            .def("__add__", [](const NBO& a, const sSOP<complex_type>& b){return a+b;})

            .def("__sub__", [](const NBO& a, const sOP& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sPOP& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sNBO<real_type>& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sNBO<complex_type>& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sSOP<real_type>& b){return a-b;})
            .def("__sub__", [](const NBO& a, const sSOP<complex_type>& b){return a-b;})

            .def("__div__", [](const NBO& a, const real_type& b){return a*(1.0/b);})
            .def("__div__", [](const NBO& a, const complex_type& b){return a*(1.0/b);})

            .def("__mul__", [](const NBO& a, const coeff<real_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const coeff<complex_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const real_type& b){return a*b;})
            .def("__mul__", [](const NBO& a, const complex_type& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sOP& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sPOP& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sNBO<real_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sNBO<complex_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sSOP<real_type>& b){return a*b;})
            .def("__mul__", [](const NBO& a, const sSOP<complex_type>& b){return a*b;})

            .def("__rmul__", [](const NBO& a, const coeff<real_type>& b){return a*b;})
            .def("__rmul__", [](const NBO& a, const coeff<complex_type>& b){return a*b;})
            .def("__rmul__", [](const NBO& a, const real_type& b){return a*b;})
            .def("__rmul__", [](const NBO& a, const complex_type& b){return a*b;});
    }
    {
        using _SOP = sSOP<real_type>;
        using container_type = typename _SOP::container_type;
        //wrapper for the sPOP type 
        py::class_<_SOP>(m, "sSOP_real")
            .def(py::init())
            .def(py::init<size_t>())
            .def(py::init<const std::string&>())
            .def(py::init<const sOP&>())
            .def(py::init<const sPOP&>())
            .def(py::init<const sNBO<real_type>&>())
            .def(py::init<const _SOP&>())
            .def("assign", [](_SOP& self, const _SOP& o){self=o;})
            .def("__copy__",[](const _SOP& o){return _SOP(o);})
            .def("__deepcopy__", [](const _SOP& o, py::dict){return _SOP(o);}, py::arg("memo"))
            .def("clear", &_SOP::clear)
            .def("reserve", &_SOP::reserve)
            .def("nmodes", &_SOP::nmodes)
            .def("nterms", &_SOP::nterms)
            .def("__len__",&_SOP::nterms)
            .def(
                    "__iter__",
                    [](_SOP& s){return py::make_iterator(s.begin(), s.end());},
                    py::keep_alive<0, 1>()
                )
            .def_property
                (
                    "label", 
                    static_cast<const std::string& (_SOP::*)() const>(&_SOP::label),
                    [](_SOP& o, const std::string& i){o.label() = i;}
                )
            .def_property
                (
                    "terms", 
                    static_cast<const container_type& (_SOP::*)() const>(&_SOP::terms),
                    [](_SOP& o, const container_type& i){o.terms() = i;}
                )
            .def(
                    "__setitem__", 
                    [](_SOP& self, size_t i, const sNBO<real_type>& v){self[i] = v;}
                )
            .def(
                    "__getitem__", 
                    static_cast<sNBO<real_type>& (_SOP::*)(size_t)>(&_SOP::operator[]),
                    py::return_value_policy::reference
                )
            .def("__str__", [](const _SOP& o){std::ostringstream oss; oss << o; return oss.str();})

            .def("__iadd__", [](_SOP& a, const sOP& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const sPOP& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const sNBO<real_type>& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const _SOP& b){return a+=b;})

            .def("__imul__", [](_SOP& a, const real_type& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sOP& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sPOP& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sNBO<real_type>& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const _SOP& b){return a*=b;})

            .def("__add__", [](const _SOP& a, const sOP& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sPOP& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sNBO<real_type>& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sNBO<complex_type>& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sSOP<real_type>&  b){return a+b;})
            .def("__add__", [](const _SOP& a, const sSOP<complex_type>& b){return a+b;})

            .def("__sub__", [](const _SOP& a, const sOP& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sPOP& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sNBO<real_type>& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sNBO<complex_type>& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sSOP<real_type>& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sSOP<complex_type>& b){return a-b;})

            .def("__div__", [](const _SOP& a, const real_type& b){return a*(1.0/b);})
            .def("__div__", [](const _SOP& a, const complex_type& b){return a*(1.0/b);})

            .def("__mul__", [](const _SOP& a, const coeff<real_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const coeff<complex_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const real_type& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const complex_type& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sOP& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sPOP& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sNBO<real_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sNBO<complex_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sSOP<real_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sSOP<complex_type>& b){return a*b;})

            .def("__rmul__", [](const _SOP& a, const coeff<real_type>& b){return a*b;})
            .def("__rmul__", [](const _SOP& a, const coeff<complex_type>& b){return a*b;})
            .def("__rmul__", [](const _SOP& a, const real_type& b){return a*b;})
            .def("__rmul__", [](const _SOP& a, const complex_type& b){return a*b;});
    }
    {
        using _SOP = sSOP<complex_type>;
        using container_type = typename _SOP::container_type;
        //wrapper for the sPOP type 
        py::class_<_SOP>(m, "sSOP_complex")
            .def(py::init())
            .def(py::init<size_t>())
            .def(py::init<const std::string&>())
            .def(py::init<const sOP&>())
            .def(py::init<const sPOP&>())
            .def(py::init<const sNBO<real_type>&>())
            .def(py::init<const sNBO<complex_type>&>())
            .def(py::init<const sSOP<real_type>&>())
            .def(py::init<const _SOP&>())
            .def("assign", [](_SOP& self, const sSOP<real_type>& o){self=o;})
            .def("assign", [](_SOP& self, const _SOP& o){self=o;})
            .def("__copy__",[](const _SOP& o){return _SOP(o);})
            .def("__deepcopy__", [](const _SOP& o, py::dict){return _SOP(o);}, py::arg("memo"))
            .def("clear", &_SOP::clear)
            .def("reserve", &_SOP::reserve)
            .def("nmodes", &_SOP::nmodes)
            .def("nterms", &_SOP::nterms)
            .def("__len__",&_SOP::nterms)
            .def(
                    "__iter__",
                    [](_SOP& s){return py::make_iterator(s.begin(), s.end());},
                    py::keep_alive<0, 1>()
                )
            .def_property
                (
                    "label", 
                    static_cast<const std::string& (_SOP::*)() const>(&_SOP::label),
                    [](_SOP& o, const std::string& i){o.label() = i;}
                )
            .def_property
                (
                    "terms", 
                    static_cast<const container_type& (_SOP::*)() const>(&_SOP::terms),
                    [](_SOP& o, const container_type& i){o.terms() = i;}
                )
            .def(
                    "__setitem__", 
                    [](_SOP& self, size_t i, const sNBO<complex_type>& v){self[i] = v;}
                )
            .def(
                    "__getitem__", 
                    static_cast<sNBO<complex_type>& (_SOP::*)(size_t)>(&_SOP::operator[]), 
                    py::return_value_policy::reference
                )
            .def("__str__", [](const _SOP& o){std::ostringstream oss; oss << o; return oss.str();})

            .def("__iadd__", [](_SOP& a, const sOP& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const sPOP& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const sNBO<real_type>& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const sNBO<complex_type>& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const sSOP<real_type>& b){return a+=b;})
            .def("__iadd__", [](_SOP& a, const _SOP& b){return a+=b;})

            .def("__imul__", [](_SOP& a, const real_type& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const complex_type& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sOP& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sPOP& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sNBO<real_type>& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sNBO<complex_type>& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const sSOP<real_type>& b){return a*=b;})
            .def("__imul__", [](_SOP& a, const _SOP& b){return a*=b;})

            .def("__add__", [](const _SOP& a, const sOP& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sPOP& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sNBO<real_type>& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sNBO<complex_type>& b){return a+b;})
            .def("__add__", [](const _SOP& a, const sSOP<real_type>& b){return a+b;})
            .def("__add__", [](const _SOP& a, const _SOP& b){return a+b;})

            .def("__sub__", [](const _SOP& a, const sOP& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sPOP& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sNBO<real_type>& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sNBO<complex_type>& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const sSOP<real_type>& b){return a-b;})
            .def("__sub__", [](const _SOP& a, const _SOP& b){return a-b;})

            .def("__div__", [](const _SOP& a, const real_type& b){return a*(1.0/b);})
            .def("__div__", [](const _SOP& a, const complex_type& b){return a*(1.0/b);})

            .def("__mul__", [](const _SOP& a, const coeff<real_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const coeff<complex_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const real_type& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const complex_type& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sOP& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sPOP& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sNBO<real_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sNBO<complex_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const sSOP<real_type>& b){return a*b;})
            .def("__mul__", [](const _SOP& a, const _SOP& b){return a*b;})

            .def("__rmul__", [](const _SOP& a, const coeff<real_type>& b){return a*b;})
            .def("__rmul__", [](const _SOP& a, const coeff<complex_type>& b){return a*b;})
            .def("__rmul__", [](const _SOP& a, const real_type& b){return a*b;})
            .def("__rmul__", [](const _SOP& a, const complex_type& b){return a*b;});
    }
}

void initialise_sSOP(py::module& m);
#endif
