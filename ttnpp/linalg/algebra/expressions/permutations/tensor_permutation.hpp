#ifndef LINALG_ALGEBRA_TENSOR_PERMUTATION_EXPRESSION_HPP
#define LINALG_ALGEBRA_TENSOR_PERMUTATION_EXPRESSION_HPP

#include "../expression_base.hpp"

namespace linalg
{

namespace expression_templates
{

template <typename _tensor_type, bool _conj>
class tensor_permutation_3_expression : public expression_base<tensor_permutation_3_expression<_tensor_type, _conj>, false>
{
public:
    static_assert(_tensor_type::rank == 3, "Failed to initialise tensor_permutation_3_expression class with tensor_type template not corresponding to a rank 3 tensor.");

    using base_type = expression_base<tensor_permutation_3_expression<_tensor_type, _conj>, false>;
    using tensor_type = _tensor_type;
    using value_type = typename std::remove_cv<typename traits<tensor_type>::value_type>::type;
    using backend_type = typename traits<tensor_type>::backend_type;
    using size_type = typename backend_type::size_type;
    using shape_type = std::array<size_type, 3>;

protected:
    const tensor_type& m_arr;
    value_type m_alpha;
    size_type m_perm_ind;

public:
    tensor_permutation_3_expression() = delete;
    tensor_permutation_3_expression(const tensor_type& arr, size_type permind, const value_type& alpha = value_type(1.0)) : base_type(shape_type{{0,0,0}}), m_arr(arr), m_alpha(alpha), m_perm_ind(permind)
    {
        ASSERT(permind < 3, "Failed to construct tensor_permutation_3_expression object.  The selected permutation index is out of bounds.");
        switch(permind)
        {
            case(0):
            {
                this->m_shape[0] = m_arr.shape(1);
                this->m_shape[1] = m_arr.shape(2);
                this->m_shape[2] = m_arr.shape(0);
                break;
            }
            case(1):
            {
                this->m_shape[0] = m_arr.shape(0);
                this->m_shape[1] = m_arr.shape(2);
                this->m_shape[2] = m_arr.shape(1);
                break;
            }
            case(2):
            {
                this->m_shape[0] = m_arr.shape(0);
                this->m_shape[1] = m_arr.shape(1);
                this->m_shape[2] = m_arr.shape(2);
                break;
            }
        };
    }

    const tensor_type& tensor() const{return m_arr;}
    value_type coeff() const{return m_alpha;}
    size_type permutation_index() const{return m_perm_ind;}

    //function for evaluating the transpose_expression.  This function can only be applied to rank 2 dense tensors
    template <typename array_type>
    typename std::enable_if<array_type::rank == 3 && is_dense<array_type>::value, void>::type applicative(array_type& res) const
    {
        ASSERT(res.buffer() != m_arr.buffer(), "Inplace rank 3 tensor reordering is not supported.");
        switch(m_perm_ind)
        {
            //perform the reordering as a matrix transpose
            case(0):
            {   
                CALL_AND_HANDLE(backend_type::transpose(_conj, m_arr.shape(0), m_arr.shape(1)*m_arr.shape(2), m_alpha, m_arr.buffer(), res.buffer()), "Failed to evaluate rank 3 tensor reordering.  Error when evaluating reordering as transpose.");
                break;
            }
            case(1):
            {
                CALL_AND_HANDLE(backend_type::batched_transpose(_conj, m_arr.shape(1), m_arr.shape(2), m_alpha, m_arr.buffer(), res.buffer(), m_arr.shape(0)), "Failed to evaluate rank 3 tensor reordering.  Error when evaluating reordering as batched transpose.");
                break;
            }
            //perform the reordering as a direct copy
            case(2):
            {
                if(_conj)
                {
                    if(m_alpha == value_type(1.0)){CALL_AND_HANDLE(backend_type::complex_conjugate(m_arr.shape(0)*m_arr.shape(1)*m_arr.shape(2), m_arr.buffer(), res.buffer()), "Failed to evaluate rank 3 tensor reordering error when evaluating it using conjugate call.");}
                    else{CALL_AND_HANDLE(res = m_alpha*conj(m_arr), "Failed to evaluate rank 3 tensor reordering error when evaluating it using scalar multiplication call.");}
                }
                else
                {
                    if(m_alpha == value_type(1.0)){CALL_AND_HANDLE(backend_type::copy(m_arr.buffer(), m_arr.shape(0)*m_arr.shape(1)*m_arr.shape(2), res.buffer()), "Failed to evaluate rank 3 tensor reordering.  Error when evaluating reordering as scalar multiplication call.");}
                    else{CALL_AND_HANDLE(res = m_alpha*m_arr, "Failed to evaluate rank 3 tensor reordering.  Error when evaluating reordering as scalar multiplication call.");}
                }
            }
        };
    }
};

}   //namespace expression_templates
template <typename tensor_type, bool conj>
struct traits<expression_templates::tensor_permutation_3_expression<tensor_type, conj>>
{
    using value_type = typename traits<tensor_type>::value_type;
    using backend_type = typename traits<tensor_type>::backend_type;
    using shape_type = std::array<typename backend_type::size_type, 3>;
    using const_shape_reference = const shape_type&;
    static constexpr size_t rank = 3;
};

}   //namespace linalg

#endif //LINALG_ALGEBRA_TENSOR_PERMUTATION_EXPRESSION_HPP//

