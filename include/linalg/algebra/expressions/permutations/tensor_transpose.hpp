#ifndef LINALG_ALGEBRA_TENSOR_PERMUTATION_EXPRESSION_HPP
#define LINALG_ALGEBRA_TENSOR_PERMUTATION_EXPRESSION_HPP

#include "../expression_base.hpp"

#include <vector>
#include <set>
#include <algorithm>

namespace linalg
{

namespace expression_templates
{

template <typename _tensor_type>
class tensor_transpose_expression<_tensor_type, typename std::enable_if<is_dense<_tensor_type>::value, void>::type> : public expression_base<tensor_transpose_expression<_tensor_type, void>, false>
 
{
public:
    using base_type = expression_base<tensor_transpose_expression<_tensor_type>, false>;
    using tensor_type = _tensor_type;
    using value_type = typename std::remove_cv<typename traits<tensor_type>::value_type>::type;
    using backend_type = typename traits<tensor_type>::backend_type;
    using size_type = typename backend_type::size_type;
    using shape_type = std::array<size_type, traits<tensor_type>::rank>;

protected:
    const tensor_type& m_arr;
    std::vector<size_type> m_order;

public:
    tensor_transpose_expression() = delete;
    template <typename int_type>
    tensor_transpose_expression(const tensor_type& arr, const std::vector<int_type>& order) : base_type(arr.shape()), m_arr(arr)
    {   
        ASSERT(order.size() == traits<tensor_type>::rank, "Insufficient indices for tensor transpose.");
        std::set<int_type> s(order.begin(), order.end());
        ASSERT(s.size() == order.size(), "Failed to compute tensor transpose, repeated index.")
        int_type irank = traits<tensor_type>::rank;

        m_order.resize(order.size());
        for(size_type i = 0; i < m_order.size(); ++i)
        {
            ASSERT(order[i] < irank, "Failed to compute tensor transpose, index out of bounds.");
            m_order[i] = order[i];
            this->m_shape[i] = m_arr.shape(m_order[i]);
        }
    }

    const tensor_type& tensor() const{return m_arr;}
    const std::vector<size_type>& permutation_order() const{return m_order;}

    //function for evaluating the transpose_expression.  This function can only be applied to rank 2 dense tensors
    template <typename array_type>
    typename std::enable_if<is_dense<array_type>::value, void>::type applicative(array_type& res) const
    {
        ASSERT(res.buffer() != m_arr.buffer(), "Inplace generic tensor reordering is not supported.");
        CALL_AND_HANDLE(backend_type::tensor_transpose(m_arr.buffer(), m_arr.shape(), m_arr.stride(), res.buffer(), res.shape(), res.stride(), m_order), "Failed to compute tensor transpose.");
    }
};

}   //namespace expression_templates
template <typename tensor_type>
struct traits<expression_templates::tensor_transpose_expression<tensor_type>>
{
    using value_type = typename traits<tensor_type>::value_type;
    using backend_type = typename traits<tensor_type>::backend_type;
    static constexpr size_t rank = traits<tensor_type>::rank;
    using shape_type = std::array<typename backend_type::size_type, rank>;
    using const_shape_reference = const shape_type&;
};

}   //namespace linalg

#endif //LINALG_ALGEBRA_TENSOR_PERMUTATION_EXPRESSION_HPP//

