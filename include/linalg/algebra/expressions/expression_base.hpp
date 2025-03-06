#ifndef LINALG_ALGEBRA_EXPRESSION_BASE_HPP
#define LINALG_ALGEBRA_EXPRESSION_BASE_HPP

#include "../../linalg_forward_decl.hpp"

namespace linalg
{
namespace expression_templates
{

//crtp base class for arbitrary expressions.  This allows for much cleaner implementation of the assignments of expressions to tensors
template <typename derived>
class expression_base<derived, false> : public expression_type<traits<derived>::rank, false>
{
public:
    using backend_type = typename traits<derived>::backend_type;
    using value_type = typename traits<derived>::value_type;
    using shape_type = typename traits<derived>::shape_type;
    using const_shape_reference = typename traits<derived>::const_shape_reference;

    static constexpr size_t rank = traits<derived>::rank;

protected:
    shape_type m_shape;

public:
    expression_base() = delete;
    expression_base(const_shape_reference _shape) : m_shape(_shape){}
    
    const_shape_reference shape() const{return m_shape;}
    template <typename result_type, typename ... Args> void operator()(result_type& res, Args&& ... args) const{CALL_AND_RETHROW(static_cast<const derived*>(this)->applicative(res, std::forward<Args>(args)...));}
    template <typename result_type, typename ... Args> void add_assignment(result_type& res, Args&& ... args) const{CALL_AND_RETHROW(static_cast<const derived*>(this)->addition_applicative(res, std::forward<Args>(args)...));}
    template <typename result_type, typename ... Args> void subtract_assignment(result_type& res, Args&& ... args) const{CALL_AND_RETHROW(static_cast<const derived*>(this)->subtraction_applicative(res, std::forward<Args>(args)...));}
};



template <typename derived>
class expression_base<derived, true> : public expression_type<traits<derived>::rank, true>
{
public:
    using backend_type = typename traits<derived>::backend_type;
    using value_type = typename traits<derived>::value_type;
    using shape_type = typename traits<derived>::shape_type;
    using const_shape_reference = typename traits<derived>::const_shape_reference;

    static constexpr size_t rank = traits<derived>::rank;

protected:
    shape_type m_shape;

public:
    expression_base() = delete;
    expression_base(const_shape_reference _shape) : m_shape(_shape){}
    
    const_shape_reference shape() const{return m_shape;}
    template <typename result_type, typename ... Args> void operator()(result_type& res, Args&& ... args){CALL_AND_RETHROW(static_cast<derived*>(this)->applicative(res, std::forward<Args>(args)...));}
    template <typename result_type, typename ... Args> void add_assignment(result_type& res, Args&& ... args){CALL_AND_RETHROW(static_cast<derived*>(this)->addition_applicative(res, std::forward<Args>(args)...));}
    template <typename result_type, typename ... Args> void subtract_assignment(result_type& res, Args&& ... args){CALL_AND_RETHROW(static_cast<derived*>(this)->subtraction_applicative(res, std::forward<Args>(args)...));}
};


}
}   //namespace linalg


#endif  //LINALG_ALGEBRA_EXPRESSION_BASE_HPP//

