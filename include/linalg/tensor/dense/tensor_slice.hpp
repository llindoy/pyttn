#ifndef LINALG_TENSOR_SLICE_HPP
#define LINALG_TENSOR_SLICE_HPP

#include "../../linalg_forward_decl.hpp"

/**
 * @cond INTERNAL
 */

namespace linalg
{

//////////////////////////////////////////////////////////////////////////////////////////
//CRTP base class for the arbitrary rank, reduced dimensional slice of a tensor.  This  //
//provide the majority of the functionality required for the implementations.           //
//////////////////////////////////////////////////////////////////////////////////////////
template <typename tensor_slice_impl>
class tensor_slice_base : public tensor_details<tensor_slice_impl>, public dense_tensor_type<traits<tensor_slice_impl>::rank>
{
public:
    using value_type =  typename traits<tensor_slice_impl>::value_type;
    using backend_type = typename traits<tensor_slice_impl>::backend_type;
    using size_type = typename backend_type::size_type;   
    using pointer = typename std::add_pointer<value_type>::type;   using const_pointer = typename std::add_pointer<typename std::add_const<value_type>::type>::type;

    using container_type = typename traits<tensor_slice_impl>::container_type;
    static constexpr size_type rank = traits<tensor_slice_impl>::rank;
    using slice_traits = tensor_slice_traits<container_type, value_type, rank>;

    static constexpr size_type container_rank = slice_traits::container_rank;
    static constexpr size_type first_index = (container_rank - rank);

    using shape_type = std::array<size_type, rank>;
    using container_pointer = typename slice_traits::container_pointer;
    using self_type = tensor_slice_base<tensor_slice_impl>;
    using detail_type = tensor_details<tensor_slice_impl>;
public:
    using memfill = memory::filler<value_type, backend_type>;
    template <typename srcbck> using memtransfer = memory::transfer<srcbck, backend_type>;

protected:
    container_pointer m_tensor;
    pointer m_buffer;
    size_type m_i;             //parameters for indexing the 

public:
    tensor_slice_base(container_pointer tensor, pointer _buffer, size_type i) : m_tensor(tensor), m_buffer(_buffer+i*tensor->stride(first_index-1)), m_i(i){}
    tensor_slice_base(tensor_slice_impl&& impl) : m_tensor(std::move(impl.m_tensor)), m_buffer(std::move(impl.m_buffer)), m_i(std::move(impl.m_i)){}

    tensor_slice_base(const tensor_slice_base& o) = default;
    tensor_slice_base(tensor_slice_base&& o) = default;
    /**
     *  Value assignment operator.  This sets all elements of the tensor to a specific value.
     *  \param val The value that the tensor will be set to.
     */ 
    template <typename U> 
    inline typename std::enable_if<std::is_convertible<U, value_type>::value,  self_type&>::type operator=(const U& _val)
    {
        ASSERT(m_buffer != nullptr, "Unable to fill tensor object.  The buffer has not been allocated");
        CALL_AND_HANDLE(fill_impl(_val),"Failed to value assign each element of the array.  Failed to fill the buffer.");
        return *this;
    }
    
    //copy assignment operator.  
    inline self_type& operator=(const self_type& src){if(this != &src){CALL_AND_RETHROW(copy_assign_impl(src));}  return *this;}
    template <typename Container> inline other_copy_assignable_type<Container, self_type> operator=(const Container& src){CALL_AND_RETHROW(return copy_assign_impl(src));}
    template <typename Container> inline other_move_assignable_type<Container, self_type> operator=(Container&& src){CALL_AND_RETHROW(return move_assign_impl(std::forward<Container>(src)));}

    template <typename Container>
    inline typename std::enable_if<is_buffer_copyable_dense<Container, self_type>::value, self_type&>::type set_buffer(const Container& src)
    {
        ASSERT(size() == src.size(), "Failed to copy buffer from input container.  The two objects do not have the same size.");
        using srcbck = typename Container::backend_type;
        CALL_AND_HANDLE(memtransfer<srcbck>::copy(src.buffer(), src.size(), m_buffer),"Copy assignment operator failed.  Error when copying the buffer.");      
        return *this;
    }

public:
    ///////////////////////////////////////////////////////////////////////////////////////////
    //                   Addition assignment from generic tensor base types                  //
    ///////////////////////////////////////////////////////////////////////////////////////////
    inline self_type& operator+=(const self_type& src){if(this != &src){CALL_AND_RETHROW(addition_assign_impl(src));}  return *this;}
    template <typename Container> inline other_copy_assignable_type<Container, self_type> operator+=(const Container& src){CALL_AND_RETHROW(return addition_assign_impl(src));}
    template <typename Container> inline other_move_assignable_type<Container, self_type> operator+=(Container&& src){CALL_AND_RETHROW(return addition_assign_impl(std::forward<Container>(src)));}

    ///////////////////////////////////////////////////////////////////////////////////////////
    //                 Subtraction assignment from generic tensor base types                 //
    ///////////////////////////////////////////////////////////////////////////////////////////
    inline self_type& operator-=(const self_type& src){if(this != &src){CALL_AND_RETHROW(subtraction_assign_impl(src));}  return *this;}
    template <typename Container> inline other_copy_assignable_type<Container, self_type> operator-=(const Container& src){CALL_AND_RETHROW(return subtraction_assign_impl(src));}
    template <typename Container> inline other_move_assignable_type<Container, self_type> operator-=(Container&& src){CALL_AND_RETHROW(return subtraction_assign_impl(std::forward<Container>(src)));}

private:
    template <typename Container>
    inline typename std::enable_if<!is_expression<Container>::value && is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container& src)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
        using srcbck = typename traits<Container>::backend_type;
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == src.shape(i), "Failed to copy assign tensor_slice object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(memtransfer<srcbck>::copy(src.buffer(), size(), m_buffer),"Failed to copy assign tensor_slice_base object.  Error when copying the buffer.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container& src)
    {
        static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate copy_assign_impl.");
        static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == src.shape(i), "Failed to copy assign tensor_slice object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(backend_type::copy_real_to_complex(src.buffer(), size(), m_buffer),"Copy operator failed.  Error when copying the buffer.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
        shape_type _shape; CALL_AND_HANDLE(_shape = expr.shape(), "Failed to copy assign tensor_slice object.  Failed to get the shape of the expression object.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == _shape[i], "Failed to copy assign tensor_slice object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(expr(*this), "Failed to copy assign tensor_slice object.  Failed to evaluate the expression into the tensor object.");
        return *this;
    }


    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type move_assign_impl(Container&& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
        shape_type _shape; CALL_AND_HANDLE(_shape = expr.shape(), "Failed to copy assign tensor_slice object.  Failed to get the shape of the expression object.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == _shape[i], "Failed to copy assign tensor_slice object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(expr(*this), "Failed to copy assign tensor_slice object.  Failed to evaluate the expression into the tensor object.");
        return *this;
    }

private:
    template <typename Container>
    inline typename std::enable_if<!is_expression<Container>::value && is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type 
    addition_assign_impl(const Container& src)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise addition assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
        static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate addition_assign_impl.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == src.shape(i), "Failed to addition assign tensor_slice object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(backend_type::addition_assign(src.buffer(), src.size(), m_buffer),"Addition assignment operator failed.  Error when additioning the buffer.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type 
    addition_assign_impl(const Container& src)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise addition assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
        static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate addition_assign_impl.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == src.shape(i), "Failed to addition assign tensor_slice object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(backend_type::addition_assign_real_to_complex(src.buffer(), src.size(), m_buffer),"Addition assignment operator failed.  Error when copying the buffer.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type 
    addition_assign_impl(const Container& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise addition assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
        shape_type _shape; CALL_AND_HANDLE(_shape = expr.shape(), "Failed to addition assign tensor_slice object.  Failed to get the shape of the expression object.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == _shape[i], "Failed to addition assign tensor_slice object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(expr.add_assignment(*this), "Addition assignment failed.  Failed to evaluate the expression into the tensor object.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type addition_assign_impl(Container&& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise subtract assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
        shape_type _shape; CALL_AND_HANDLE(_shape = expr.shape(), "Failed to subtract assign tensor_slice object.  Failed to get the shape of the expression object.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == _shape[i], "Failed to subtract assign tensor_slice object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(expr.add_assignment(*this), "Failed to subtract assign tensor_slice object.  Failed to evaluate the expression into the tensor object.");
        return *this;
    }


private:
    template <typename Container>
    inline typename std::enable_if<!is_expression<Container>::value && is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type 
    subtraction_assign_impl(const Container& src)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise subtraction assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
        static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate addition_assign_impl.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == src.shape(i), "Failed to subtraction assign tensor_slice object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(backend_type::subtraction_assign(src.buffer(), src.size(), m_buffer),"Subtraction assignment operator failed.  Error when subtractioning the buffer.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type 
    subtraction_assign_impl(const Container& src)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise subtraction assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
        static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate subtraction_assign_impl.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == src.shape(i), "Failed to subtraction assign tensor_slice object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(backend_type::subtraction_assign_real_to_complex(src.buffer(), src.size(), m_buffer),"Subtraction assignment operator failed.  Error when subtractioning the buffer.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type 
    subtraction_assign_impl(const Container& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise subtraction assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
        shape_type _shape; CALL_AND_HANDLE(_shape = expr.shape(), "Failed to subtraction assign tensor_slice object.  Failed to get the shape of the expression object.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == _shape[i], "Failed to subtraction assign tensor_slice object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(expr.subtract_assignment(*this), "Subtraction assignment failed.  Failed to evaluate the expression into the tensor object.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type subtraction_assign_impl(Container&& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise subtract assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
        shape_type _shape; CALL_AND_HANDLE(_shape = expr.shape(), "Failed to subtract assign tensor_slice object.  Failed to get the shape of the expression object.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == _shape[i], "Failed to subtract assign tensor_slice object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(expr.subtract_assignment(*this), "Failed to subtract assign tensor_slice object.  Failed to evaluate the expression into the tensor object.");
        return *this;
    }

public:

    template <typename Container>
    inline typename std::enable_if<is_buffer_copyable_dense<Container, tensor_slice_impl>::value, self_type&>::type copy_buffer(const Container& src)
    {
        ASSERT(size() == src.size(), "Failed to copy buffer from input container.  The two objects do not have the same size.");
        using srcbck = typename Container::backend_type;
        CALL_AND_HANDLE(memtransfer<srcbck>::copy(src.buffer(), src.size(), m_buffer),"Copy assignment operator failed.  Error when copying the buffer.");
        return *this;
    }

public:
    inline const size_type& capacity() const{ return m_tensor->stride(first_index-1);}
    inline const size_type& size() const{ return m_tensor->stride(first_index-1);}
    inline const size_type& nelems() const {return m_tensor->stride(first_index-1);}
    inline const shape_type shape() const{shape_type _shape; for(size_type i=first_index; i < container_rank; ++i){_shape[i-first_index] = m_tensor->shape(i);}  return _shape;}

    inline const size_type& size(size_type i) const{CALL_AND_HANDLE(return m_tensor->shape(i+first_index), "Failed to return size of tensor slice object. Failed when accessing underlying tensor objects sizes.");}
    inline const size_type& dims(size_type i) const{CALL_AND_HANDLE(return m_tensor->shape(i+first_index), "Failed to return size of tensor slice object. Failed when accessing underlying tensor objects sizes.");}
    inline const size_type& shape(size_type i) const{CALL_AND_HANDLE(return m_tensor->shape(i+first_index), "Failed to return size of tensor slice object. Failed when accessing underlying tensor objects sizes.");}
    inline const size_type& slice_index() const{return m_i;}

    inline const size_type& stride(size_type i) const{CALL_AND_HANDLE(return m_tensor->stride(i+first_index), "Failed to return size of tensor slice object. Failed when accessing underlying tensor objects sizes.");}

    inline bool same_shape(const shape_type& _shape) const{return _shape == shape();}

    inline pointer buffer(){return m_buffer;}
    inline const_pointer buffer()const{return m_buffer;}
    inline pointer data(){return m_buffer;}
    inline const_pointer data()const{return m_buffer;}

public:    

    template <typename ... Args>
    inline reinterpreted_tensor<typename std::add_const<value_type>::type, sizeof...(Args), backend_type> reinterpret_shape(Args&& ... args) const
    {
        using reinterpreted_type = reinterpreted_tensor<typename std::add_const<value_type>::type, sizeof...(Args), backend_type>;
        CALL_AND_HANDLE(return reinterpreted_type(size(), m_buffer, std::forward<Args>(args)...), "Failed to reinterpret shape of tensor_slice object.");
    }

    template <typename ... Args>
    inline reinterpreted_tensor<value_type, sizeof...(Args), backend_type> reinterpret_shape(Args&& ... args)
    {
        using reinterpreted_type = reinterpreted_tensor<value_type, sizeof...(Args), backend_type>;
        CALL_AND_HANDLE(return reinterpreted_type(size(), m_buffer, std::forward<Args>(args)...), "Failed to reinterpret shape of tensor_slice object.");
    }

    template <size_t vD>
    inline reinterpreted_tensor<typename std::add_const<value_type>::type, vD, backend_type> reinterpret_shape(const std::array<size_type, vD>& _size) const
    {
        using reinterpreted_type = reinterpreted_tensor<typename std::add_const<value_type>::type, vD, backend_type>;
        CALL_AND_HANDLE(return reinterpreted_type(size(), m_buffer, _size), "Failed to reinterpret the shape of the tensor_slice object.");
    }

    template <size_t vD>
    inline reinterpreted_tensor<value_type, vD, backend_type> reinterpret_shape(const std::array<size_type, vD>& _size)
    {
        using reinterpreted_type = reinterpreted_tensor<value_type, vD, backend_type>;
        CALL_AND_HANDLE(return reinterpreted_type(size(), m_buffer, _size), "Failed to reinterpret the shape of the tensor_slice object.");
    }

public:
    template <typename U> 
    inline typename std::enable_if<std::is_convertible<U, value_type>::value, self_type&>::type fill_value(const U& u){CALL_AND_HANDLE(fill_impl(u), "Failed to fill buffer with value.  Error when calling fill impl.");   return *this;}
    inline self_type& fill_zeros(){CALL_AND_HANDLE(fill_impl(value_type(0.0)), "Failed to fill buffer to zero.");   return *this;}
    inline self_type& fill_ones(){CALL_AND_HANDLE(fill_impl(value_type(1.0)), "Failed to fill buffer to one.");   return *this;}

private:
    template <bool _mutable = traits<self_type>::is_mutable, typename U = value_type> typename std::enable_if<_mutable && std::is_convertible<U, value_type>::value, void>::type 
    fill_impl(const U& v){CALL_AND_HANDLE(memfill::fill(m_buffer, size(), value_type(v)), "Failed to set buffer to value.  Error when calling the memfill object fill function.");}

public:
    //Inplace scalar multiplication/division functions
    template <typename Vt> inline value_update_type<Vt, self_type> operator*=(const Vt& v){CALL_AND_HANDLE(backend_type::scal(size(), value_type(v), m_buffer, 1), "Failed to perform operator*= on tensor slice object.  scal call failed.");      return *this;}
    template <typename Vt> inline value_update_type<Vt, self_type> operator/=(const Vt& v){CALL_AND_HANDLE(backend_type::scal(size(), value_type(1.0/v), m_buffer, 1), "Failed to perform operator/= on tensor slice object.  scal call failed.");  return *this;}
};  //class tensor_slice_base


///////////////////////////////////////////////////////////////////////////////////////
// D dimensional implementation of the reduced dimensional tensor slice object for   //
//                              use with the blas backend                            //
///////////////////////////////////////////////////////////////////////////////////////
template <template <typename, size_t, typename> class ArrType, typename T, size_t D1, size_t D2, typename pref>
class tensor_slice<ArrType<T, D1, blas_backend>, pref, D2> : public tensor_slice_base<tensor_slice<ArrType<T, D1, blas_backend>, pref, D2> >
{
public:
    using array_type = ArrType<T, D1, blas_backend>;
    static_assert(is_dense_tensor<array_type>::value, "Failed to instantiate tensor_slice object.  The input array type is not a valid dense tensor type.");
    using self_type = tensor_slice<array_type, pref, D2>;
    using slice_base = tensor_slice_base<self_type>;
    using slice_traits = tensor_slice_traits<array_type, pref, D2>;
    using const_slice_traits = tensor_slice_traits<array_type, typename std::add_const<pref>::type, D2>;
    using size_type = typename slice_base::size_type;

    using value_type = pref;
    using reference = typename std::add_lvalue_reference<pref>::type;
    using const_reference = typename std::add_lvalue_reference<typename std::add_const<pref>::type>::type;
protected:
    using slice_base::m_tensor;
    using slice_base::m_buffer;

public:
    template <typename ... Args> tensor_slice(Args&& ... args) try : slice_base(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct tensor slice object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(slice_base::operator=(std::forward<Args>(args)...));   return *this;}

    //accessor operator[] for returning slices
    inline typename slice_traits::slice_type operator[](size_type i) {return slice_traits::make(m_tensor, m_buffer, i);}
    inline typename const_slice_traits::slice_type operator[](size_type i) const{return const_slice_traits::make(m_tensor, m_buffer, i);}
    inline typename slice_traits::slice_type slice(size_type i) {ASSERT(internal::compare_bounds(i, m_buffer, slice_base::shape(0)), "Unable to return slice of array.  Slice index out of bounds."); return slice_traits::make(m_tensor, i);}
    inline typename const_slice_traits::slice_type slice(size_type i) const{ASSERT(internal::compare_bounds(i, m_buffer, slice_base::shape(0)), "Unable to return slice of array.  Slice index out of bounds."); return const_slice_traits::make(m_tensor, i);}
    
    //accessor which accesses the tensor as a 1d array
    inline reference operator()(size_type index){return m_buffer[index];}
    inline const_reference operator()(size_type index) const {return m_buffer[index];}
    inline reference at(size_type i){ASSERT(internal::compare_bounds(i, slice_base::size()), "Unable to access tensor element using at.  Index out of bounds.");    return m_buffer[i];}
    inline const_reference at(size_type i) const{ASSERT(internal::compare_bounds(i, slice_base::size()), "Unable to access tensor element using at.  Index out of bounds.");    return m_buffer[i];}

    //general accessor functions
    template <typename ... Inds>
    inline reference operator()(Inds... indices)
    {
        static_assert(sizeof...(Inds)==D2, "Failed to access element of tensor object.  The input index list does not have the correct size.");  using pack_type = typename internal::check_integral<Inds...>::pack_type;
        return m_buffer[get_index<pack_type>(indices...)];
    }

    template <typename ... Inds>
    inline const_reference operator()(Inds... indices) const
    {
        static_assert(sizeof...(Inds)==D2, "Failed to access element of tensor object.  The input index list does not have the correct size.");  using pack_type = typename internal::check_integral<Inds...>::pack_type;
        return m_buffer[get_index<pack_type>(indices...)];
    }

    template <typename ... Inds>
    inline reference at(Inds... indices)
    {
        static_assert(sizeof...(Inds)==D2, "Failed to access element of tensor object.  The input index list does not have the correct size.");
        using pack_type = typename internal::check_integral<Inds...>::pack_type;    size_type index;
        CALL_AND_HANDLE(index = get_index_bounds_check<pack_type>(indices...), "Unable to access tensor element.  Failed to determine flattened index.");
        return m_buffer[index];
    }

    template <typename ... Inds>
    inline const_reference at(Inds... indices) const
    {
        static_assert(sizeof...(Inds)==D2, "Failed to access element of tensor object.  The input index list does not have the correct size.");
        using pack_type = typename internal::check_integral<Inds...>::pack_type;    size_type index;
        CALL_AND_HANDLE(index = get_index_bounds_check<pack_type>(indices...), "Unable to access tensor element.  Failed to determine flattened index.");
        return m_buffer[index];
    }

private:
    ///@cond INTERNAL
    //get the index in the array corresponding to the parameter pack.
    template <typename IntegerType, typename ... Args>
    inline size_type get_index_bounds_check(IntegerType i, Args... args) const
    {
        ASSERT(internal::compare_bounds(i, slice_base::shape(D2-sizeof...(args)-1)), "Unable to get flattened index.  One of the unflattened indices was out of bounds.");
        CALL_AND_HANDLE(return i*slice_base::stride(D2-sizeof...(args)-1) + get_index(args...),"Unable to get flattened index.  Error on iterated get_index call.");
    }
    template <typename IntegerType>
    inline size_type get_index_bounds_check(IntegerType i) const{ASSERT(internal::compare_bounds(i, slice_base::shape(D2-1)), "Unable to get flattened index.  Final unflattened index was out of bounds."); return i; }

    template <typename IntegerType, typename ... Args>  inline size_type get_index(IntegerType i, Args... args) const{return i*slice_base::stride(D2-sizeof...(args)-1) + get_index(args...);}
    template <typename IntegerType>  inline size_type get_index(IntegerType i) const{return i;}

    ///@endcond
};


///////////////////////////////////////////////////////////////////////////////////////
// 1 dimensional implementation of the reduced dimensional tensor slice object for   //
//                              use with the blas backend                            //
///////////////////////////////////////////////////////////////////////////////////////
template <template <typename, size_t, typename> class ArrType, typename T, size_t D1, typename pref>
class tensor_slice<ArrType<T, D1, blas_backend>, pref, 1> : public tensor_slice_base<tensor_slice<ArrType<T, D1, blas_backend>, pref, 1> >
{
public:
    using array_type = ArrType<T, D1, blas_backend>;
    static_assert(is_dense_tensor<array_type>::value, "Failed to instantiate tensor_slice object.  The input array type is not a valid dense tensor type.");
    using self_type = tensor_slice<array_type, pref, 1>;
    using slice_base = tensor_slice_base<self_type>;
    using size_type = typename slice_base::size_type;

    using value_type = pref;
    using reference = typename std::add_lvalue_reference<pref>::type;
    using const_reference = typename std::add_lvalue_reference<typename std::add_const<pref>::type>::type;
protected:
    using slice_base::m_tensor;
    using slice_base::m_buffer;

public:
    template <typename ... Args> tensor_slice(Args&& ... args) try : slice_base(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct tensor slice object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(slice_base::operator=(std::forward<Args>(args)...));   return *this;}

    inline reference operator[](size_type i){return m_buffer[i]; }
    inline reference operator()(size_type i){return m_buffer[i]; }
    inline const_reference operator[](size_type i) const {return m_buffer[i]; }
    inline const_reference operator()(size_type i) const {return m_buffer[i]; }

    inline reference slice(size_type i){ASSERT(internal::compare_bounds(i, slice_base::size()), "Failed to access element. Index out of bounds.");  return m_buffer[i]; }
    inline reference at(size_type i){ASSERT(internal::compare_bounds(i, slice_base::size()), "Failed to access element. Index out of bounds.");  return m_buffer[i]; }
    inline const_reference slice(size_type i) const{ASSERT(internal::compare_bounds(i, slice_base::size()), "Failed to access element. Index out of bounds.");  return m_buffer[i]; }
    inline const_reference at(size_type i) const{ASSERT(internal::compare_bounds(i, slice_base::size()), "Failed to access element. Index out of bounds.");  return m_buffer[i]; }
};



#ifdef __NVCC__
template <template <typename, size_t, typename> class ArrType, typename T, size_t D1, size_t D2, typename pref>
class tensor_slice<ArrType<T, D1, cuda_backend>, pref, D2> : public tensor_slice_base<tensor_slice<ArrType<T, D1, cuda_backend>, pref, D2> >
{
public:
    using array_type = ArrType<T, D1, cuda_backend>;
    static_assert(is_dense_tensor<array_type>::value, "Failed to instantiate tensor_slice object.  The input array type is not a valid dense tensor type.");
    using self_type = tensor_slice<array_type, pref, D2>;
    using slice_base = tensor_slice_base<self_type>;
    using slice_traits = tensor_slice_traits<array_type, pref, D2>;

    using pointer = typename slice_base::pointer;    using const_pointer = typename slice_base::const_pointer;
    using const_slice_traits = tensor_slice_traits<array_type, typename std::add_const<pref>::type, D2>;
    using size_type = typename slice_base::size_type;

protected:
    using slice_base::m_tensor;
    using slice_base::m_buffer;

public:
    template <typename ... Args> tensor_slice(Args&& ... args) try : slice_base(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct tensor slice object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(slice_base::operator=(std::forward<Args>(args)...));   return *this;}

    //accessor operator[] for returning slices
    inline typename slice_traits::slice_type operator[](size_type i) {return slice_traits::make(m_tensor, i);}
    inline typename const_slice_traits::slice_type operator[](size_type i) const{return const_slice_traits::make(m_tensor, i);}
    inline typename slice_traits::slice_type slice(size_type i) {ASSERT(internal::compare_bounds(i, slice_base::shape(0)), "Unable to return slice of array.  Slice index out of bounds."); return slice_traits::make(m_tensor, i);}
    inline typename const_slice_traits::slice_type slice(size_type i) const{ASSERT(internal::compare_bounds(i, slice_base::shape(0)), "Unable to return slice of array.  Slice index out of bounds."); return const_slice_traits::make(m_tensor, i);}

    __host__ __device__ pointer buffer(){return slice_base::m_buffer;}
    __host__ __device__ const_pointer buffer()const{return slice_base::m_buffer;}
    __host__ __device__ pointer data(){return slice_base::m_buffer;}
    __host__ __device__ const_pointer data()const{return slice_base::m_buffer;}
};

template <template <typename, size_t, typename> class ArrType, typename T, size_t D1, typename pref>
class tensor_slice<ArrType<T, D1, cuda_backend>, pref, 1> : public tensor_slice_base<tensor_slice<ArrType<T, D1, cuda_backend>, pref, 1> >
{
public:
    using array_type = ArrType<T, D1, cuda_backend>;
    static_assert(is_dense_tensor<array_type>::value, "Failed to instantiate tensor_slice object.  The input array type is not a valid dense tensor type.");
    using self_type = tensor_slice<array_type, pref, 1>;
    using slice_base = tensor_slice_base<self_type>;
    using slice_traits = tensor_slice_traits<array_type, pref, 1>;
    using size_type = typename slice_base::size_type;

    using pointer = typename slice_base::pointer;    using const_pointer = typename slice_base::const_pointer;
public:
    template <typename ... Args> tensor_slice(Args&& ... args) try : slice_base(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct tensor slice object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(slice_base::operator=(std::forward<Args>(args)...));   return *this;}

    __host__ __device__ pointer buffer(){return slice_base::m_buffer;}
    __host__ __device__ const_pointer buffer()const{return slice_base::m_buffer;}
    __host__ __device__ pointer data(){return slice_base::m_buffer;}
    __host__ __device__ const_pointer data()const{return slice_base::m_buffer;}
};

#endif  //__NVCC__

}   //namespace linalg

///@endcond

#endif  //LINALG_TENSOR_SLICE_HPP//

