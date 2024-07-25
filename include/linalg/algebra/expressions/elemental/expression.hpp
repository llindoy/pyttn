#ifndef LINALG_ALGEBRA_EXPRESSION_HPP
#define LINALG_ALGEBRA_EXPRESSION_HPP

#include "../../../linalg_forward_decl.hpp"
#include "result_type.hpp"
#include "storage_traits.hpp"


namespace linalg
{

namespace expression_templates
{

/////////////////////////////////////////////////////////////////////////////////
//              Wrapper for number types in the expression trees               //
/////////////////////////////////////////////////////////////////////////////////
template <typename T>
class literal_type<T, blas_backend>
{
static_assert(is_number<T>::value, "Failed to initialise literal type object.  The literal must be a number type.");
public:
    using value_type = T;   using backend_type = blas_backend;
    literal_type(T val) : m_value(val) {}
    inline operator value_type() const{return m_value;}
    template <typename ... Args> inline value_type operator()(Args&& ... /* args */){return m_value;}
private:
    value_type m_value;
};

#ifdef __NVCC__
template <typename T>
class literal_type<T, cuda_backend>
{
static_assert(is_number<T>::value, "Failed to initialise literal type object.  The literal must be a number type.");
public:
    using value_type = T;   using backend_type = cuda_backend;
    literal_type(T val) : m_value(val) {}
    inline __host__ __device__ operator value_type() const{return m_value;}
    template <typename ... Args> inline __host__ __device__ value_type operator()(Args&& ... args){return m_value;}
private:
    value_type m_value;
};
#endif


/////////////////////////////////////////////////////////////////////////////////
//   Helper objects for specialising the applicative functions of the binary   //
//                             expression objects.                             //
/////////////////////////////////////////////////////////////////////////////////
namespace internal
{
template <typename type, typename expr, typename backend> struct expression_applicative;

//generic blas wrapper
template <typename type, typename expr>
struct expression_applicative<type, expr, blas_backend>
{
    using size_type = typename blas_backend::size_type;
    template <typename res> static inline void apply(res& _res, const expr& _expr)
    {
        static_assert(std::is_base_of<type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
        typename res::pointer buffer = _res.buffer();    for(size_type i=0; i<_res.nelems(); ++i){buffer[i] = _expr[i];}
    }
    template <typename res> static inline void addition_assign(res& _res, const expr& _expr)
    {
        static_assert(std::is_base_of<type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
        typename res::pointer buffer = _res.buffer();    for(size_type i=0; i<_res.nelems(); ++i){buffer[i] += _expr[i];}
    }
    template <typename res> static inline void subtraction_assign(res& _res, const expr& _expr)
    {
        static_assert(std::is_base_of<type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
        typename res::pointer buffer = _res.buffer();    for(size_type i=0; i<_res.nelems(); ++i){buffer[i] -= _expr[i];}
    }
};

//blas wrapper for diagonal matrix return type
template <typename expr>
struct expression_applicative<diagonal_matrix_type, expr, blas_backend>
{
    using size_type = typename blas_backend::size_type;
    template <typename res> static inline void apply(res& _res, const expr& _expr)
    {
        static_assert(std::is_base_of<diagonal_matrix_type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
        typename res::pointer buffer = _res.buffer();    size_type incx = _res.incx();    for(size_type i=0; i<_res.nelems(); ++i){buffer[i*incx] = _expr[i];}
    }
    template <typename res> static inline void addition_assign(res& _res, const expr& _expr)
    {
        static_assert(std::is_base_of<diagonal_matrix_type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
        typename res::pointer buffer = _res.buffer();    size_type incx = _res.incx();    for(size_type i=0; i<_res.nelems(); ++i){buffer[i*incx] += _expr[i];}
    }
    template <typename res> static inline void subtraction_assign(res& _res, const expr& _expr)
    {
        static_assert(std::is_base_of<diagonal_matrix_type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
        typename res::pointer buffer = _res.buffer();    size_type incx = _res.incx();    for(size_type i=0; i<_res.nelems(); ++i){buffer[i*incx] -= _expr[i];}
    }
};

#ifdef __NVCC__

//generic cuda wrapper
template <typename type, typename expr>
struct expression_applicative<type, expr, cuda_backend>
{
    using size_type = typename cuda_backend::size_type;
    template <typename res> static inline void apply(res& _res, const expr& _expr)
    {
        static_assert(std::is_base_of<type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
        CALL_AND_HANDLE(cuda_backend::evaluate_expression_tree(_res.buffer(), _res.nelems(), _expr), "Failed to evaluate expression object.  The cuda backend evaluate expression tree raised an exception.");
    }
    template <typename res> static inline void addition_assign(res& _res, const expr& _expr)
    {
        static_assert(std::is_base_of<type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
        CALL_AND_HANDLE(cuda_backend::evaluate_add_expression_tree(_res.buffer(), _res.nelems(), _expr), "Failed to evaluate expression object.  The cuda backend evaluate expression tree raised an exception.");
    }
    template <typename res> static inline void subtraction_assign(res& _res, const expr& _expr)
    {
        static_assert(std::is_base_of<type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
        CALL_AND_HANDLE(cuda_backend::evaluate_sub_expression_tree(_res.buffer(), _res.nelems(), _expr), "Failed to evaluate expression object.  The cuda backend evaluate expression tree raised an exception.");
    }
};

//blas wrapper for diagonal matrix return type
template <typename expr>
struct expression_applicative<diagonal_matrix_type, expr, cuda_backend>
{
    using size_type = typename cuda_backend::size_type;
    template <typename res> static inline void apply(res& _res, const expr& _expr)
    {
        static_assert(std::is_base_of<diagonal_matrix_type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
        CALL_AND_HANDLE(cuda_backend::evaluate_expression_tree_strided(_res.buffer(), _res.nelems(), _res.incx(), _expr), "Failed to evaluate expression object.  The cuda backend evaluate expression tree raised an exception.");
    }
    template <typename res> static inline void addition_assign(res& _res, const expr& _expr)
    {
        static_assert(std::is_base_of<diagonal_matrix_type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
        CALL_AND_HANDLE(cuda_backend::evaluate_add_expression_tree_strided(_res.buffer(), _res.nelems(), _res.incx(), _expr), "Failed to evaluate expression object.  The cuda backend evaluate expression tree raised an exception.");
    }
    template <typename res> static inline void subtraction_assign(res& _res, const expr& _expr)
    {
        static_assert(std::is_base_of<diagonal_matrix_type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
        CALL_AND_HANDLE(cuda_backend::evaluate_sub_expression_tree_strided(_res.buffer(), _res.nelems(), _res.incx(), _expr), "Failed to evaluate expression object.  The cuda backend evaluate expression tree raised an exception.");
    }
};

#endif
}   //namespace internal

/////////////////////////////////////////////////////////////////////////////////
//               Unary expression type wrapper for blas backend                //
/////////////////////////////////////////////////////////////////////////////////
template <typename vtype, template <typename > class operation> 
class unary_expression<vtype, operation, blas_backend> 
{
private:
    using vtraits = storage_traits<vtype>;
    typename vtraits::type m_val;
    typename vtraits::eval_type m_eval;
public:
    using self_type = unary_expression<vtype, operation, blas_backend>;
    using backend_type = blas_backend;
    using size_type = typename backend_type::size_type;
    using op_type = operation<backend_type>;
    using value_type = typename result_type<self_type>::value_type;
    using result_type = typename result_type<self_type>::type;
    using eval_type = internal::expression_applicative<result_type, self_type, backend_type>;

    unary_expression(typename vtraits::type v) : m_val(v), m_eval(vtraits::data(m_val)) {}
    auto obj() const ->decltype(m_val){return m_val;}

    template <typename array_type> void operator()(array_type& res) const{CALL_AND_HANDLE(eval_type::apply(res,*this), "Failed to evaluate unary expression.");}
    template <typename array_type> void addition_assign(array_type& res) const{CALL_AND_HANDLE(eval_type::addition_assign(res, *this), "Failed to evaluate unary expression.");}
    template <typename array_type> void subtraction_assign(array_type& res) const{CALL_AND_HANDLE(eval_type::subtraction_assign(res, *this), "Failed to evaluate unary expression.");}
    value_type operator[](size_type i) const{return op_type::apply(m_eval, i);}
};  //class unary_expression

#ifdef __NVCC__
/////////////////////////////////////////////////////////////////////////////////
//               Unary expression type wrapper for cuda backend                //
/////////////////////////////////////////////////////////////////////////////////
template <typename vtype, template <typename > class operation> 
class unary_expression<vtype, operation, cuda_backend>
{
private:
    using vtraits = storage_traits<vtype>;
    typename vtraits::type m_val;
    typename vtraits::eval_type m_eval;
public:
    using self_type = unary_expression<vtype, operation, cuda_backend>;
    using backend_type = cuda_backend;
    using size_type = typename backend_type::size_type;
    using op_type = operation<backend_type>;
    using value_type = typename result_type<self_type>::value_type;
    using result_type = typename result_type<self_type>::type;
    using eval_type = internal::expression_applicative<result_type, self_type, backend_type>;

    unary_expression(typename vtraits::type v) : m_val(v), m_eval(vtraits::data(m_val)) {}
    auto obj() const ->decltype(m_val){return m_val;}

    template <typename array_type> void operator()(array_type& res) const{CALL_AND_HANDLE(eval_type::apply(res,*this), "Failed to evaluate unary expression.");}
    template <typename array_type> void addition_assign(array_type& res) const{CALL_AND_HANDLE(eval_type::addition_assign(res, *this), "Failed to evaluate unary expression.");}
    template <typename array_type> void subtraction_assign(array_type& res) const{CALL_AND_HANDLE(eval_type::subtraction_assign(res, *this), "Failed to evaluate unary expression.");}
    inline __device__ value_type operator[](size_type i) const{return op_type::apply(m_eval, i);}
};  //class unary_expression
#endif




/////////////////////////////////////////////////////////////////////////////////
//              Binary expression type wrapper for blas backend                //
/////////////////////////////////////////////////////////////////////////////////
template <typename ltype, typename rtype, template <typename > class operation> 
class binary_expression<ltype, rtype, operation, blas_backend>
{
private:
    using ltraits = storage_traits<ltype>;
    using rtraits = storage_traits<rtype>;
    typename ltraits::type m_lstore;
    typename rtraits::type m_rstore;
    typename ltraits::eval_type m_left;
    typename rtraits::eval_type m_right;
public:
    using self_type = binary_expression<ltype, rtype, operation, blas_backend>;
    using backend_type = blas_backend;
    using size_type = typename backend_type::size_type;
    using op_type = operation<backend_type>;
    using value_type = typename result_type<self_type>::value_type;
    using result_type = typename result_type<self_type>::type;
    using eval_type = internal::expression_applicative<result_type, self_type, backend_type>;

    binary_expression(typename ltraits::type l, typename rtraits::type r) : m_lstore(l), m_rstore(r), m_left(ltraits::data(l)), m_right(rtraits::data(r)) {}

    auto left() const ->decltype(m_lstore){return m_lstore;}
    auto right() const ->decltype(m_rstore){return m_rstore;}

    template <typename array_type> void operator()(array_type& res) const{CALL_AND_HANDLE(eval_type::apply(res,*this), "Failed to evaluate binary expression.");}
    template <typename array_type> void addition_assign(array_type& res) const{CALL_AND_HANDLE(eval_type::addition_assign(res, *this), "Failed to evaluate binary expression.");}
    template <typename array_type> void subtraction_assign(array_type& res) const{CALL_AND_HANDLE(eval_type::subtraction_assign(res, *this), "Failed to evaluate binary expression.");}
    value_type operator[](size_type i) const{return op_type::apply(m_left, m_right, i);}
};  //class binary_expression

#ifdef __NVCC__
/////////////////////////////////////////////////////////////////////////////////
//              Binary expression type wrapper for cuda backend                //
/////////////////////////////////////////////////////////////////////////////////
template <typename ltype, typename rtype, template <typename > class operation> 
class binary_expression<ltype, rtype, operation, cuda_backend>
{
private:
    using ltraits = storage_traits<ltype>;
    using rtraits = storage_traits<rtype>;
    typename ltraits::type m_lstore;
    typename rtraits::type m_rstore;
    typename ltraits::eval_type m_left;
    typename rtraits::eval_type m_right;
public:
    using self_type = binary_expression<ltype, rtype, operation, cuda_backend>;
    using backend_type = cuda_backend;
    using size_type = typename backend_type::size_type;
    using op_type = operation<backend_type>;
    using value_type = typename result_type<self_type>::value_type;
    using result_type = typename result_type<self_type>::type;
    using eval_type = internal::expression_applicative<result_type, self_type, backend_type>;

    binary_expression(typename ltraits::type l, typename rtraits::type r) : m_lstore(l), m_rstore(r), m_left(ltraits::data(l)), m_right(rtraits::data(r)) {}

    auto left() const ->decltype(m_lstore){return m_lstore;}
    auto right() const ->decltype(m_rstore){return m_rstore;}

    template <typename array_type> void operator()(array_type& res) const{CALL_AND_HANDLE(eval_type::apply(res,*this), "Failed to evaluate binary expression.");}
    template <typename array_type> void addition_assign(array_type& res) const{CALL_AND_HANDLE(eval_type::addition_assign(res, *this), "Failed to evaluate binary expression.");}
    template <typename array_type> void subtraction_assign(array_type& res) const{CALL_AND_HANDLE(eval_type::subtraction_assign(res, *this), "Failed to evaluate binary expression.");}
    inline __device__ value_type operator[](size_type i) const{return op_type::apply(m_left, m_right, i);}
};  //class binary_expression

#endif


/////////////////////////////////////////////////////////////////////////////////
// Top level expression tree object for wrapping an exposed binary expression. //
/////////////////////////////////////////////////////////////////////////////////
template <typename ltype, typename rtype, template <typename> class operation, typename backend, size_t _rank>
class expression_tree<binary_expression<ltype, rtype, operation, backend>, _rank, backend>
    : public expression_base<expression_tree<binary_expression<ltype, rtype, operation, backend>, _rank, backend>, false>
    , public result_type<binary_expression<ltype, rtype, operation, backend>>::type
{
public:
    using expr = binary_expression<ltype, rtype, operation, backend>;
    using rtraits = result_type<expr>;
    using type = typename rtraits::type;
    using shape_type = typename rtraits::shape_type;
    using const_shape_reference = typename rtraits::const_shape_reference;

    using value_type = typename rtraits::value_type;
    static constexpr size_t rank = rtraits::rank;
    static_assert(_rank == rank ,"Failed to construct expression_tree object.  The specified rank is not compatible with the result_type rank.");
    using self_type = expression_tree<expr, rank, backend>;
    using base_type = expression_base<self_type, false>;

private:
    expr m_expr;

    template <typename arr>
    using valid_result_array = typename std::conditional<std::is_base_of<type, arr>::value && std::is_same<backend, typename traits<arr>::backend_type>::value && std::is_same<value_type, typename traits<arr>::value_type>::value, std::true_type, std::false_type>::type;
    
public:
    expression_tree(const expr& _expr, shape_type _shape) : base_type(_shape), m_expr(_expr) {}
    expression_tree() = delete;
#ifdef __NVCC__
    __host__ __device__ ~expression_tree(){}
#else
    ~expression_tree(){}
#endif

    const expr& expression() const{return m_expr;}
    auto left() const -> decltype(m_expr.left()){return m_expr.left();}
    auto right() const -> decltype(m_expr.right()){return m_expr.right();}
    template <typename array_type> 
    typename std::enable_if<valid_result_array<array_type>::value && traits<array_type>::is_mutable, void>::type applicative(array_type& res) const{CALL_AND_HANDLE(m_expr(res), "Failed to evaluate expression object into result array.");}

    template <typename array_type> 
    typename std::enable_if<valid_result_array<array_type>::value && traits<array_type>::is_mutable, void>::type addition_applicative(array_type& res) const{CALL_AND_HANDLE(m_expr.addition_assign(res), "Failed to evaluate expression object into result array.");}

    template <typename array_type> 
    typename std::enable_if<valid_result_array<array_type>::value && traits<array_type>::is_mutable, void>::type subtraction_applicative(array_type& res) const{CALL_AND_HANDLE(m_expr.subtraction_assign(res), "Failed to evaluate expression object into result array.");}
};  //class expression_tree



/////////////////////////////////////////////////////////////////////////////////
//  Top level expression tree object for wrapping an exposed unary expression  //
/////////////////////////////////////////////////////////////////////////////////
template <typename vtype, template <typename> class operation, typename backend, size_t _rank>
class expression_tree<unary_expression<vtype, operation, backend>, _rank, backend>
    : public expression_base<expression_tree<unary_expression<vtype, operation, backend>, _rank, backend>, false>
    , public result_type<unary_expression<vtype, operation, backend>>::type
{
public:
    using expr = unary_expression<vtype, operation, backend>;

    using rtraits = result_type<expr>;
    using type = typename rtraits::type;
    using shape_type = typename rtraits::shape_type;
    using const_shape_reference = typename rtraits::const_shape_reference;
    using value_type = typename rtraits::value_type;
    static constexpr size_t rank = rtraits::rank;
    static_assert(_rank == rank ,"Failed to construct expression_tree object.  The specified rank is not compatible with the result_type rank.");
    
    using self_type = expression_tree<expr, rank, backend>;
    using base_type = expression_base<self_type, false>;

private:
    expr m_expr;
    
    template <typename arr>
    using valid_result_array = typename std::conditional<std::is_base_of<type, arr>::value && std::is_same<backend, typename traits<arr>::backend_type>::value && std::is_same<value_type, typename traits<arr>::value_type>::value, std::true_type, std::false_type>::type;

public:
    expression_tree(const expr& _expr, shape_type _shape) : base_type(_shape), m_expr(_expr) {}
    expression_tree() = delete;
#ifdef __NVCC__
    __host__ __device__ ~expression_tree(){}
#else
    ~expression_tree(){}
#endif

    const expr& expression() const{return m_expr;}
    auto obj() const -> decltype(m_expr.obj()){return m_expr.obj();}

    template <typename array_type> 
    typename std::enable_if<valid_result_array<array_type>::value && traits<array_type>::is_mutable, void>::type applicative(array_type& res) const{CALL_AND_HANDLE(m_expr(res), "Failed to evaluate expression object into result array.");}
    template <typename array_type> 
    typename std::enable_if<valid_result_array<array_type>::value && traits<array_type>::is_mutable, void>::type addition_applicative(array_type& res) const{CALL_AND_HANDLE(m_expr.addition_assign(res), "Failed to evaluate expression object into result array.");}
    template <typename array_type> 
    typename std::enable_if<valid_result_array<array_type>::value && traits<array_type>::is_mutable, void>::type subtraction_applicative(array_type& res) const{CALL_AND_HANDLE(m_expr.subtraction_assign(res), "Failed to evaluate expression object into result array.");}

};  //class expression_tree

}   //namespace expression_templates

/////////////////////////////////////////////////////////////////////////////////////////////////
//                          traits objects for the expression types                            //
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename expr, size_t D, typename backend>
struct traits<expression_templates::expression_tree<expr, D, backend> >
{
    static constexpr size_t rank = D;
    using rtraits = expression_templates::result_type<expr>;
    using shape_type = typename rtraits::shape_type;
    using const_shape_reference = typename rtraits::const_shape_reference;
    using value_type = typename rtraits::value_type;
    using backend_type = backend;
};

}   //namespace linalg



#endif  //LINALG_ALGEBRA_EXPRESSION_HPP//

