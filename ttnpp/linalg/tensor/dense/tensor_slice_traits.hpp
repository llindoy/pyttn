#ifndef LINALG_TENSOR_SLICE_TRAITS_HPP
#define LINALG_TENSOR_SLICE_TRAITS_HPP

#include "../../linalg_forward_decl.hpp"

/**
 * @cond INTERNAL
 * \class Tensor_Slice_traits
 * This class is used by the Tensor class to allow efficient contiguous tensor views through the 
 * operator[].  This will store all of the information to all deferring of the evaluation of the 
 * operator[] until the result is used.
 */


namespace linalg{

//traits object for constructing the D-1 dimensional slice from a D dimensional tensor object
template <typename ArrType, typename U, size_t D>
struct tensor_slice_traits
{
    using ctraits = traits<ArrType>;
    using size_type = typename ctraits::size_type;

    static constexpr size_type container_rank = ctraits::rank;
    static constexpr size_type rank = D;
    static constexpr bool is_mutable = (!std::is_const<U>::value) && ctraits::is_mutable;

    static_assert(D <= container_rank, "Invalid tensor_slice_traits object.  The dimension of the slice object must be less than the tensor slice dimension");
    using container_pointer = typename std::add_pointer<typename std::conditional<is_mutable, ArrType, typename std::add_const<ArrType>::type>::type>::type;
    using const_container_pointer = typename std::add_pointer<typename std::add_const<ArrType>::type>::type;

    using value_type =  typename traits<ArrType>::value_type;
    using pointer = typename std::add_pointer<value_type>::type;   using const_pointer = typename std::add_pointer<typename std::add_const<value_type>::type>::type;

    using slice_type = tensor_slice<ArrType, U, D-1>;
    using const_slice_type = tensor_slice<ArrType, typename std::add_const<U>::type, D-1>;
    
    template <bool _mutable = is_mutable> static inline typename std::enable_if<_mutable, slice_type>::type make(container_pointer a, size_type i){return slice_type(a, a->buffer(), i);}
    static inline const_slice_type make(const_container_pointer a, size_type i){return const_slice_type(a, a->buffer(), i);}

    template <bool _mutable = is_mutable> static inline typename std::enable_if<_mutable, slice_type>::type make(container_pointer a, pointer p, size_type i){return slice_type(a, p, i);}
    static inline const_slice_type make(const_container_pointer a, const_pointer p, size_type i){return const_slice_type(a, p, i);}
};

template <typename ArrType, typename U>
struct tensor_slice_traits<ArrType, U, 1>
{
    using ctraits = traits<ArrType>;
    using size_type = typename ctraits::size_type;

    static constexpr size_type container_rank = ctraits::rank;
    static constexpr size_type rank = 1;
    static constexpr bool is_mutable = (!std::is_const<U>::value)&&ctraits::is_mutable;

    static_assert(1 <= container_rank, "Invalid tensor_slice_traits object.  The dimension of the slice object must be less than the tensor slice dimension");
    using container_type = ArrType;
    using container_pointer = typename std::add_pointer<typename std::conditional<is_mutable, ArrType, typename std::add_const<ArrType>::type>::type>::type;
    using const_container_pointer = typename std::add_pointer<typename std::add_const<ArrType>::type>::type;


    using slice_type = typename std::add_lvalue_reference<U>::type;
    using const_slice_type = typename std::add_const<typename std::add_lvalue_reference<U>::type>::type;

    
};

}

///@endcond

#endif  //LINALG_TENSOR_SLICE_TRAITS_HPP//

