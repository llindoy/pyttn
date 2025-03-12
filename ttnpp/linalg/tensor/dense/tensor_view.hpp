#ifndef LINALG_TENSOR_VIEW_HPP
#define LINALG_TENSOR_VIEW_HPP

#include "../../linalg_forward_decl.hpp"

namespace linalg
{
///////////////////////////////////////////////////////////////////////////////////////
//                  The CRTP base class of the tensor view object                    //
///////////////////////////////////////////////////////////////////////////////////////
template <typename ViewRef>
class tensor_view_base : public view_type, public dense_tensor_type<traits<ViewRef>::rank>, public tensor_details<ViewRef>
{
public:
    using value_type = typename traits<ViewRef>::value_type;
    using pointer = typename std::add_pointer<value_type>::type;    using const_pointer = typename std::add_pointer<typename std::add_const<value_type>::type>::type;
    using backend_type = typename traits<ViewRef>::backend_type;
    using size_type = typename backend_type::size_type;   

    static constexpr size_type rank = traits<ViewRef>::rank;   ///< Returns the rank (dimensionality) of the tensor

    using shape_type = std::array<size_type, rank>;
    using self_type = tensor_view_base<ViewRef>;

    //the classes used for memory allocation and transfer
    using memfill = memory::filler<value_type, backend_type>;
    template <typename srcbck> using memtransfer = memory::transfer<srcbck, backend_type>;

    using detail_type = tensor_details<ViewRef>;
protected:
    pointer m_buffer;                                   ///< The 1-dimensional array used to store the tensor
    shape_type m_shape;                                 ///< A static array of size D that stores the shape of the tensor
    shape_type m_stride;                                ///< A D-dimensional static array storing 
    size_type m_totsize;
    
public:
    template <typename ... Dims>
    tensor_view_base(size_type mcapacity, pointer _buffer, Dims&& ... args) : m_buffer(_buffer), m_shape{0}, m_stride{0} 
    {
        static_assert(sizeof...(Dims) == rank, "Failed to compile tensor_view_base class.  The number of input dimensions is not equal to the rank of the tensor.");;
        CALL_AND_HANDLE(init_shape<0>(std::forward<Dims>(args)...),"Failed to construct tensor object.  Array buffer initialisation failed.");
        ASSERT(m_totsize == mcapacity, "Failed to construct tensor_view_base_object.  The input size is not compatible with the input tensor object.");
    }

    tensor_view_base(size_type mcapacity,  pointer _buffer, const shape_type& shape) : m_buffer(_buffer), m_shape(shape), m_stride{0}, m_totsize{1}
    {
        for(size_t i=0; i<rank; ++i){m_totsize*=m_shape[i];}    init_stride();  
        ASSERT(m_totsize == mcapacity, "Failed to construct tensor_view_base_object.  The input size is not compatible with the input tensor object.");
    }

    template <typename ... Dims>
    tensor_view_base(size_type mcapacity, bool check_size, pointer _buffer, Dims&& ... args) : m_buffer(_buffer), m_shape{0}, m_stride{0} 
    {
        static_assert(sizeof...(Dims) == rank, "Failed to compile tensor_view_base class.  The number of input dimensions is not equal to the rank of the tensor.");;
        CALL_AND_HANDLE(init_shape<0>(std::forward<Dims>(args)...),"Failed to construct tensor object.  Array buffer initialisation failed.");
        if(check_size)
        {
            ASSERT(m_totsize == mcapacity, "Failed to construct tensor_view_base_object.  The input size is not compatible with the input tensor object.");
        }
        else
        {
            ASSERT(m_totsize <= mcapacity, "Failed to construct tensor_view_base_object.  The input size is not compatible with the input tensor object.");
        }
    }

    tensor_view_base(size_type mcapacity, bool check_size, pointer _buffer, const shape_type& shape) : m_buffer(_buffer), m_shape(shape), m_stride{0}, m_totsize{1}
    {
        for(size_t i=0; i<rank; ++i){m_totsize*=m_shape[i];}    init_stride();  
        if(check_size)
        {
            ASSERT(m_totsize == mcapacity, "Failed to construct tensor_view_base_object.  The input size is not compatible with the input tensor object.");
        }
        else
        {
            ASSERT(m_totsize <= mcapacity, "Failed to construct tensor_view_base_object.  The input size is not compatible with the input tensor object.");
        }
    }

    tensor_view_base(const tensor_view_base& o) = default;
    tensor_view_base(tensor_view_base&& o) = default;

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
    inline self_type& operator+=(const self_type& src){if(this != &src){CALL_AND_RETHROW(addition_assign_impl(src));}  return *this;}
    template <typename Container> inline other_copy_assignable_type<Container, self_type> operator+=(const Container& src){CALL_AND_RETHROW(return addition_assign_impl(src));}
    template <typename Container> inline other_move_assignable_type<Container, self_type> operator+=(Container&& src){CALL_AND_RETHROW(return addition_assign_impl(std::forward<Container>(src)));}

    inline self_type& operator-=(const self_type& src){if(this != &src){CALL_AND_RETHROW(subtraction_assign_impl(src));}  return *this;}
    template <typename Container> inline other_copy_assignable_type<Container, self_type> operator-=(const Container& src){CALL_AND_RETHROW(return subtraction_assign_impl(src));}
    template <typename Container> inline other_move_assignable_type<Container, self_type> operator-=(Container&& src){CALL_AND_RETHROW(return subtraction_assign_impl(std::forward<Container>(src)));}

private:
    template <typename Container>
    inline typename std::enable_if<is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container& src)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for tensor_view object.  The specified tensor view is not mutable.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == src.shape(i), "Failed to copy assign tensor_view object.  The two tensor objects do not have the same shape.");}
        using srcbck = typename traits<Container>::backend_type;
        CALL_AND_HANDLE(memtransfer<srcbck>::copy(src.buffer(), size(), m_buffer),"Failed to copy assign tensor_view_base object.  Error when copying the buffer.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container& src)
    {
        static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate copy_assign_impl.");
        static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for tensor_view object.  The specified tensor view is not mutable.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == src.shape(i), "Failed to copy assign tensor_view object.  The two tensor objects do not have the same shape.");}
        CALL_AND_HANDLE(backend_type::copy_real_to_complex(src.buffer(), size(), m_buffer),"Copy operator failed.  Error when copying the buffer.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for tensor_view object.  The specified tensor view is not mutable.");
        shape_type _shape; CALL_AND_HANDLE(_shape = expr.shape(), "Failed to copy assign tensor_view object.  Failed to get the shape of the expression object.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == _shape[i], "Failed to copy assign tensor_view object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(expr(*this), "Failed to copy assign tensor_view object.  Failed to evaluate the expression into the tensor object.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type move_assign_impl(Container&& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for tensor_view object.  The specified tensor view is not mutable.");
        shape_type _shape; CALL_AND_HANDLE(_shape = expr.shape(), "Failed to copy assign tensor_view object.  Failed to get the shape of the expression object.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == _shape[i], "Failed to copy assign tensor_view object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(expr(*this), "Failed to copy assign tensor_view object.  Failed to evaluate the expression into the tensor object.");
        return *this;
    }

private:
    template <typename Container>
    inline typename std::enable_if<!is_expression<Container>::value && is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type 
    addition_assign_impl(const Container& src)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise addition assignment operator for tensor_view object.  The specified tensor slice is not mutable.");
        static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate addition_assign_impl.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == src.shape(i), "Failed to addition assign tensor_view object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(backend_type::addition_assign(src.buffer(), src.size(), m_buffer),"Addition assignment operator failed.  Error when additioning the buffer.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type 
    addition_assign_impl(const Container& src)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise addition assignment operator for tensor_view object.  The specified tensor slice is not mutable.");
        static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate addition_assign_impl.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == src.shape(i), "Failed to addition assign tensor_view object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(backend_type::addition_assign_real_to_complex(src.buffer(), src.size(), m_buffer),"Addition assignment operator failed.  Error when copying the buffer.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type 
    addition_assign_impl(const Container& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise addition assignment operator for tensor_view object.  The specified tensor slice is not mutable.");
        shape_type _shape; CALL_AND_HANDLE(_shape = expr.shape(), "Failed to addition assign tensor_view object.  Failed to get the shape of the expression object.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == _shape[i], "Failed to addition assign tensor_view object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(expr.add_assignment(*this), "Addition assignment failed.  Failed to evaluate the expression into the tensor object.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type addition_assign_impl(Container&& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise addition assignment operator for tensor_view object.  The specified tensor slice is not mutable.");
        shape_type _shape; CALL_AND_HANDLE(_shape = expr.shape(), "Failed to addition assign tensor_view object.  Failed to get the shape of the expression object.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == _shape[i], "Failed to addition assign tensor_view object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(expr.add_assignment(*this), "Failed to addition assign tensor_view object.  Failed to evaluate the expression into the tensor object.");
        return *this;
    }


private:
    template <typename Container>
    inline typename std::enable_if<!is_expression<Container>::value && is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type 
    subtraction_assign_impl(const Container& src)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise subtraction assignment operator for tensor_view object.  The specified tensor slice is not mutable.");
        static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate addition_assign_impl.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == src.shape(i), "Failed to subtraction assign tensor_view object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(backend_type::subtraction_assign(src.buffer(), src.size(), m_buffer),"Subtraction assignment operator failed.  Error when subtractioning the buffer.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type 
    subtraction_assign_impl(const Container& src)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise subtraction assignment operator for tensor_view object.  The specified tensor slice is not mutable.");
        static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate subtraction_assign_impl.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == src.shape(i), "Failed to subtraction assign tensor_view object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(backend_type::subtraction_assign_real_to_complex(src.buffer(), src.size(), m_buffer),"Subtraction assignment operator failed.  Error when subtractioning the buffer.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type 
    subtraction_assign_impl(const Container& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise subtraction assignment operator for tensor_view object.  The specified tensor slice is not mutable.");
        shape_type _shape; CALL_AND_HANDLE(_shape = expr.shape(), "Failed to subtraction assign tensor_view object.  Failed to get the shape of the expression object.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == _shape[i], "Failed to subtraction assign tensor_view object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(expr.subtract_assignment(*this), "Subtraction assignment failed.  Failed to evaluate the expression into the tensor object.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type subtraction_assign_impl(Container&& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise subtract assignment operator for tensor_view object.  The specified tensor slice is not mutable.");
        shape_type _shape; CALL_AND_HANDLE(_shape = expr.shape(), "Failed to subtract assign tensor_view object.  Failed to get the shape of the expression object.");
        for(size_type i=0; i<rank; ++i){ASSERT(shape(i) == _shape[i], "Failed to subtract assign tensor_view object.  The two slice object do not have the same shape.");}
        CALL_AND_HANDLE(expr.subtract_assignment(*this), "Failed to subtract assign tensor_view object.  Failed to evaluate the expression into the tensor object.");
        return *this;
    }

public:
    //methods for accessing the size of the tensor object.
    inline const size_type& capacity() const{ return m_totsize;}
    inline const size_type& size() const{ return m_totsize;}
    inline const size_type& nelems() const {return m_totsize;}
    inline const shape_type& shape() const{ return m_shape;}
    inline const shape_type& stride() const{ return m_stride;}

    inline const size_type& size(size_type i) const{    ASSERT(i < rank, "Unable to access tensor size element.  Index out of bounds.");  return m_shape[i];}
    inline const size_type& dims(size_type i) const{    ASSERT(i < rank, "Unable to access tensor size element.  Index out of bounds.");  return m_shape[i];}
    inline const size_type& shape(size_type i) const{   ASSERT(i < rank, "Unable to access tensor size element.  Index out of bounds.");  return m_shape[i];}
    inline const size_type& stride(size_type i) const{  ASSERT(i < rank, "Unable to access tensor size element.  Index out of bounds.");  return m_stride[i];}

    inline bool same_shape(const shape_type& _shape) const{return _shape == m_shape;}
public:    
    template <typename Itype, typename ... Args>
    inline typename std::enable_if<std::is_integral<Itype>::value, reinterpreted_tensor<typename std::add_const<value_type>::type, sizeof...(Args)+1, backend_type>>::type reinterpret_shape(const Itype& i, Args&& ... args) const
    {
        using reinterpreted_type = reinterpreted_tensor<typename std::add_const<value_type>::type, sizeof...(Args)+1, backend_type>;
        CALL_AND_HANDLE(return reinterpreted_type(size(), true, m_buffer, i, std::forward<Args>(args)...), "Failed to reinterpret the shape of the tensor_view object.");
    }

    template <typename Itype, typename ... Args>
    inline typename std::enable_if<std::is_integral<Itype>::value, reinterpreted_tensor<value_type, sizeof...(Args)+1, backend_type>>::type reinterpret_shape(const Itype& i, Args&& ... args)
    {
        using reinterpreted_type = reinterpreted_tensor<value_type, sizeof...(Args)+1, backend_type>;
        CALL_AND_HANDLE(return reinterpreted_type(size(), true, m_buffer, i, std::forward<Args>(args)...), "Failed to reinterpret the shape of the tensor_view object.");
    }

    template <size_t vD>
    inline reinterpreted_tensor<typename std::add_const<value_type>::type, vD, backend_type> reinterpret_shape(const std::array<size_type, vD>& _size) const
    {
        using reinterpreted_type = reinterpreted_tensor<typename std::add_const<value_type>::type, vD, backend_type>;
        CALL_AND_HANDLE(return reinterpreted_type(size(), true, m_buffer, _size), "Failed to reinterpret the shape of the tensor_view object.");
    }

    template <size_t vD>
    inline reinterpreted_tensor<value_type, vD, backend_type> reinterpret_shape(const std::array<size_type, vD>& _size)
    {
        using reinterpreted_type =reinterpreted_tensor<value_type, vD, backend_type>;
        CALL_AND_HANDLE(return reinterpreted_type(size(), true, m_buffer, _size), "Failed to reinterpret the shape of the tensor_view object.");
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
    template <typename Vt> inline value_update_type<Vt, self_type> operator*=(const Vt& v){CALL_AND_HANDLE(backend_type::scal(size(), value_type(v), m_buffer, 1), "Failed to perform operator*= on tensor view object.  scal call failed.");      return *this;}
    template <typename Vt> inline value_update_type<Vt, self_type> operator/=(const Vt& v){CALL_AND_HANDLE(backend_type::scal(size(), value_type(1.0/v), m_buffer, 1), "Failed to perform operator/= on reinterpreted object.  scal call failed.");  return *this;}

    inline pointer buffer(){return m_buffer;}
    inline const_pointer buffer()const{return m_buffer;}
    inline pointer data(){return m_buffer;}
    inline const_pointer data()const{return m_buffer;}

private:
    /** These two functions can and should probably be changed to ensure no additional runtime overhead*/
    //initialise the remaining dimensions of the array if required.
    template <size_type d>
    inline void init_size()
    {
        m_totsize = 1;
        if(d!=rank){for(size_type i=d; i<rank; ++i){m_shape[i] = m_shape[d-1];}}
        for(size_type i=0; i<rank; ++i){m_totsize*=m_shape[i];}
    }

    //initialize the stride array storing how far we need to skip through the array
    inline void init_stride()
    {
        size_type D = rank;
        m_stride[D-1] = 1;
        for(size_type i=1; i<D; ++i){m_stride[D-i-1] = m_shape[D-i]*m_stride[D-i];}
    }

    //Function for initialising the array
    template <size_type d, typename U, typename ... Args>
    typename std::enable_if<std::is_integral<U>::value, void>::type init_shape(U i, Args&& ... args){ m_shape[d] = i; CALL_AND_HANDLE(init_shape<d+1>(std::forward<Args>(args)...), "Failed to evaluate intermediate init call.");}

    //if we get a value_type in the initialiser we set the value to contain buffer size.
    template <size_type d> void init_shape(){init_size<d>(); init_stride();}
 
};  //class tensor



///////////////////////////////////////////////////////////////////////////////////////
// D dimensional implementation of the tensor view object for use with the  //
//                                    blas backend                                   //
///////////////////////////////////////////////////////////////////////////////////////
template <typename T, size_t D>
class tensor_view<T, D, blas_backend> : public tensor_view_base<tensor_view<T, D, blas_backend> >
{
public:
    using self_type = tensor_view<T, D, blas_backend>;
    using base_type = tensor_view_base<self_type>;
    using size_type = typename base_type::size_type;

    using value_type = T;
    using reference = typename std::add_lvalue_reference<T>::type;
    using const_reference = typename std::add_lvalue_reference<typename std::add_const<T>::type>::type;

    using const_slice_traits = tensor_slice_traits<self_type, typename std::add_const<T>::type, D>;
    using slice_traits = tensor_slice_traits<self_type, T, D>;
protected:
    using base_type::m_totsize;
    using base_type::m_shape;
    using base_type::m_stride;
    using base_type::m_buffer;

public:
    template <typename ... Args> tensor_view(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct tensor view object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));   return *this;}

public:
    //accessor operator[] for returning slices
    inline typename slice_traits::slice_type operator[](size_type i) {return slice_traits::make(this, i);}
    inline typename const_slice_traits::slice_type operator[](size_type i) const{return const_slice_traits::make(this, i);}
    inline typename slice_traits::slice_type slice(size_type i) {ASSERT(internal::compare_bounds(i, m_shape[0]), "Unable to return slice of array.  Slice index out of bounds."); return slice_traits::make(this, i);}
    inline typename const_slice_traits::slice_type slice(size_type i) const{ASSERT(internal::compare_bounds(i, m_shape[0]), "Unable to return slice of array.  Slice index out of bounds."); return const_slice_traits::make(this, i);}
    
    //accessor which accesses the tensor as a 1d array
    inline reference operator()(size_type index){return m_buffer[index];}
    inline const_reference operator()(size_type index) const {return m_buffer[index];}
    inline reference at(size_type i){ASSERT(internal::compare_bounds(i, m_totsize), "Unable to access tensor element using at.  Index out of bounds.");    return m_buffer[i];}
    inline const_reference at(size_type i) const{ASSERT(internal::compare_bounds(i, m_totsize), "Unable to access tensor element using at.  Index out of bounds.");    return m_buffer[i];}

    //general accessor functions
    template <typename ... Inds>
    inline reference operator()(Inds... indices)
    {
        static_assert(sizeof...(Inds)==D, "Failed to access element of tensor object.  The input index list does not have the correct size.");  using pack_type = typename internal::check_integral<Inds...>::pack_type;
        return m_buffer[get_index<pack_type>(indices...)];
    }

    template <typename ... Inds>
    inline const_reference operator()(Inds... indices) const
    {
        static_assert(sizeof...(Inds)==D, "Failed to access element of tensor object.  The input index list does not have the correct size.");  using pack_type = typename internal::check_integral<Inds...>::pack_type;
        return m_buffer[get_index<pack_type>(indices...)];
    }

    template <typename ... Inds>
    inline reference at(Inds... indices)
    {
        static_assert(sizeof...(Inds)==D, "Failed to access element of tensor object.  The input index list does not have the correct size.");
        using pack_type = typename internal::check_integral<Inds...>::pack_type;    size_type index;
        CALL_AND_HANDLE(index = get_index_bounds_check<pack_type>(indices...), "Unable to access tensor element.  Failed to determine flattened index.");
        return m_buffer[index];
    }

    template <typename ... Inds>
    inline const_reference at(Inds... indices) const
    {
        static_assert(sizeof...(Inds)==D, "Failed to access element of tensor object.  The input index list does not have the correct size.");
        using pack_type = typename internal::check_integral<Inds...>::pack_type;    size_type index;
        CALL_AND_HANDLE(index = get_index_bounds_check<pack_type>(indices...), "Unable to access tensor element.  Failed to determine flattened index.");
        return m_buffer[index];
    }
private:
    ///@cond INTERNAL - we might want to move this elsewhere - this should be common to all dense tensor types.
    //get the index in the array corresponding to the parameter pack.
    template <typename IntegerType, typename ... Args>
    inline size_type get_index_bounds_check(IntegerType i, Args... args) const
    {
        ASSERT(internal::compare_bounds(i, m_shape[D-sizeof...(args)-1]), "Unable to get flattened index.  One of the unflattened indices was out of bounds.");
        CALL_AND_HANDLE(return i*m_stride[D-sizeof...(args)-1] + get_index_bounds_check<IntegerType>(args...), "Unable to get flattened index.  Error on iterated get_index call.");
    }
    template <typename IntegerType>
    inline size_type get_index_bounds_check(IntegerType i) const{ASSERT(internal::compare_bounds(i, m_shape[D-1]), "Unable to get flattened index.  Final unflattened index was out of bounds."); return i; }

    template <typename IntegerType, typename ... Args>  inline size_type get_index(IntegerType i, Args... args) const{return i*m_stride[D-sizeof...(args)-1] + get_index<IntegerType>(args...);}
    template <typename IntegerType>  inline size_type get_index(IntegerType i) const{return i;}
    ///@endcond
};

///////////////////////////////////////////////////////////////////////////////////////
// 1 dimensional implementation of the tensor view object for use with the  //
//                                    blas backend                                   //
///////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class tensor_view<T, 1, blas_backend> : public tensor_view_base<tensor_view<T, 1, blas_backend> >
{
public:
    using self_type = tensor_view<T, 1, blas_backend>;
    using base_type = tensor_view_base<self_type>;
    using size_type = typename base_type::size_type;

    using value_type = T;
    using reference = typename std::add_lvalue_reference<T>::type;
    using const_reference = typename std::add_lvalue_reference<typename std::add_const<T>::type>::type;
protected:
    using base_type::m_buffer;
    using base_type::m_totsize;

public:
    template <typename ... Args> tensor_view(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct tensor view object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));   return *this;}

    inline reference operator[](size_type i){return m_buffer[i]; }
    inline reference operator()(size_type i){return m_buffer[i]; }
    inline const_reference operator[](size_type i) const {return m_buffer[i]; }
    inline const_reference operator()(size_type i) const {return m_buffer[i]; }

    inline reference slice(size_type i){ASSERT(internal::compare_bounds(i, m_totsize), "Failed to access slice. Index out of bounds.");  return m_buffer[i]; }
    inline reference at(size_type i){ASSERT(internal::compare_bounds(i, m_totsize), "Failed to access slice. Index out of bounds.");  return m_buffer[i]; }
    inline const_reference slice(size_type i) const{ASSERT(internal::compare_bounds(i, m_totsize), "Failed to access slice. Index out of bounds.");  return m_buffer[i]; }
    inline const_reference at(size_type i) const{ASSERT(internal::compare_bounds(i, m_totsize), "Failed to access slice. Index out of bounds.");  return m_buffer[i]; }
};


#ifdef PYTTN_BUILD_CUDA

///////////////////////////////////////////////////////////////////////////////////////
// D dimensional implementation of the tensor view object for use with the  //
//                                    cuda backend                                   //
///////////////////////////////////////////////////////////////////////////////////////
template <typename T, size_t D>
class tensor_view<T, D, cuda_backend > : public tensor_view_base<tensor_view<T, D, cuda_backend > >
{
public:
    using self_type = tensor_view<T, D, cuda_backend >;
    using value_type = T;
    using base_type = tensor_view_base<self_type>;
    using size_type = typename base_type::size_type;
    using const_slice_traits = tensor_slice_traits<self_type, typename std::add_const<T>::type, D>;
    using slice_traits = tensor_slice_traits<self_type, T, D>;
    using pointer = typename base_type::pointer;    using const_pointer = typename base_type::const_pointer;
protected:
    using base_type::m_totsize;
    using base_type::m_shape;
    using base_type::m_stride;
    using base_type::m_buffer;

public:
    template <typename ... Args> tensor_view(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct tensor view object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));   return *this;}

public:
    //accessor operator[] for returning slices
    inline typename slice_traits::slice_type operator[](size_type i) {return slice_traits::make(this, i);}
    inline typename const_slice_traits::slice_type operator[](size_type i) const{return const_slice_traits::make(this, i);}
    inline typename slice_traits::slice_type slice(size_type i) {ASSERT(internal::compare_bounds(i, m_shape[0]), "Unable to return slice of array.  Slice index out of bounds."); return slice_traits::make(this, i);}
    inline typename const_slice_traits::slice_type slice(size_type i) const{ASSERT(internal::compare_bounds(i, m_shape[0]), "Unable to return slice of array.  Slice index out of bounds."); return const_slice_traits::make(this, i);}
   
    __host__ __device__ pointer buffer(){return m_buffer;}
    __host__ __device__ const_pointer buffer()const{return m_buffer;}
    __host__ __device__ pointer data(){return m_buffer;}
    __host__ __device__ const_pointer data()const{return m_buffer;}
};

///////////////////////////////////////////////////////////////////////////////////////
//       1 dimensional implementation of the tensor view object for use with the     //
//                                    cuda backend                                   //
///////////////////////////////////////////////////////////////////////////////////////
template <typename T>
class tensor_view<T, 1, cuda_backend > : public tensor_view_base<tensor_view<T, 1, cuda_backend > >
{
public:
    using self_type = tensor_view<T, 1, cuda_backend >;
    using base_type = tensor_view_base<self_type>;
    using size_type = typename base_type::size_type;
    using value_type = T;

    using pointer = typename base_type::pointer;    using const_pointer = typename base_type::const_pointer;
protected:
    using base_type::m_buffer;
    using base_type::m_totsize;

public:
    template <typename ... Args> tensor_view(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct tensor view object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));   return *this;}

    __host__ __device__ pointer buffer(){return m_buffer;}
    __host__ __device__ const_pointer buffer()const{return m_buffer;}
    __host__ __device__ pointer data(){return m_buffer;}
    __host__ __device__ const_pointer data()const{return m_buffer;}
};

#endif  //PYTTN_BUILD_CUDA

///////////////////////////////////////////////////////////////////////////////////////
//                      SPECIAL TENSOR VIEW IMPLEMENTATIONS                          //
///////////////////////////////////////////////////////////////////////////////////////
//classes for interpreting a tensor object as an arbitrary rank tensor object.
template <typename T, size_t D, typename backend> class reinterpreted_tensor : public tensor_view<T, D, backend> 
{
public:
    using value_type = T;
    using backend_type = backend;
    using base_type = tensor_view<T, D, backend>;
    using self_type = reinterpreted_tensor<T, D, backend>;
    template <typename ... Args> reinterpreted_tensor(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct reinterpreted tensor object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));   return *this;}
};

//classes for interpreting a matrix object as a hermitian matrix object.  This does not actually restrict the buffer to store hermitian matrices but functions which 
//can will interpret it as if it was a hermitian matrix.
template <typename T, typename backend> class hermitian_matrix : public tensor_view<T, 2, backend>, public hermitian_type
{
public:
    using value_type = T;
    using backend_type = backend;
    using base_type = tensor_view<T, 2, backend>;
    using self_type = hermitian_matrix<T, backend>;
    template <typename ... Args> hermitian_matrix(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct hermitian matrix object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));   return *this;}
};

//classes for interpreting a matrix object as a hermitian matrix object.  This does not actually restrict the buffer to store hermitian matrices but functions which 
//can will interpret it as if it was an upper hessenberg matrix.  Not that for this to work it will treat the matrix in column major format (as otherwise it is a lower
//hessenberg matrix).
template <typename T, typename backend> class upper_hessenberg_matrix : public tensor_view<T, 2, backend>, public upper_hessenberg_type
{
public:
    using value_type = T;
    using backend_type = backend;
    using base_type = tensor_view<T, 2, backend>;
    using self_type = upper_hessenberg_matrix<T, backend>;

    template <typename ... Args> upper_hessenberg_matrix(Args&& ... args) try : base_type(std::forward<Args>(args)...), m_ordering(MATRIX_ORDERING::ROW_MAJOR) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct upper hessenberg matrix object.");}

    template <typename ... Args> upper_hessenberg_matrix(MATRIX_ORDERING morder, Args&& ... args) try : base_type(std::forward<Args>(args)...), m_ordering(morder) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct upper hessenberg matrix object.");}

    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));   return *this;}

    MATRIX_ORDERING& order(){return m_ordering;}
    const MATRIX_ORDERING& order() const{return m_ordering;}
protected:
    MATRIX_ORDERING m_ordering;
};


///////////////////////////////////////////////////////////////////////////////////////
//           FUNCTIONS FOR REINTERPRETING TENSORS AS THESE SPECIAL VIEWS             //
///////////////////////////////////////////////////////////////////////////////////////
//hermitian matrix view
template <typename array_type>
typename std::enable_if<is_dense_matrix<array_type>::value, hermitian_matrix<typename std::add_const<typename traits<array_type>::value_type>::type, typename traits<array_type>::backend_type>>::type
hermitian_view(const array_type& array)
{
    using result_type = hermitian_matrix<typename std::add_const<typename traits<array_type>::value_type>::type, typename traits<array_type>::backend_type>;
    ASSERT(array.size(0) == array.size(1), "Failed to construct hermitian view of dense tensor.  The tensor must be a square matrix.");
    CALL_AND_HANDLE(return result_type(array.size(), array.buffer(), array.size(0), array.size(1)), "Failed to return hermitian view of dense tensor.");
}

template <typename array_type>
typename std::enable_if<is_dense_matrix<array_type>::value && !traits<array_type>::is_mutable, hermitian_matrix<typename std::add_const<typename traits<array_type>::value_type>::type, typename traits<array_type>::backend_type>>::type
hermitian_view(array_type& array)
{
    using result_type = hermitian_matrix<typename std::add_const<typename traits<array_type>::value_type>::type, typename traits<array_type>::backend_type>;
    ASSERT(array.size(0) == array.size(1), "Failed to construct hermitian view of dense tensor.  The tensor must be a square matrix.");
    CALL_AND_HANDLE(return result_type(array.size(), array.buffer(), array.size(0), array.size(1)), "Failed to return hermitian view of dense tensor.");
}

template <typename array_type>
typename std::enable_if<is_dense_matrix<array_type>::value && traits<array_type>::is_mutable, hermitian_matrix<typename traits<array_type>::value_type, typename traits<array_type>::backend_type>>::type
hermitian_view(array_type& array)
{
    using result_type = hermitian_matrix<typename traits<array_type>::value_type, typename traits<array_type>::backend_type>;
    ASSERT(array.size(0) == array.size(1), "Failed to construct hermitian view of dense tensor.  The tensor must be a square matrix.");
    CALL_AND_HANDLE(return result_type(array.size(), array.buffer(), array.size(0), array.size(1)), "Failed to return hermitian view of dense tensor.");
}


//upper hessenberg matrix view
template <typename array_type>
typename std::enable_if<is_dense_matrix<array_type>::value, upper_hessenberg_matrix<typename std::add_const<typename traits<array_type>::value_type>::type, typename traits<array_type>::backend_type>>::type
upper_hessenberg_view(const array_type& array, MATRIX_ORDERING morder = MATRIX_ORDERING::ROW_MAJOR)
{
    using result_type = upper_hessenberg_matrix<typename std::add_const<typename traits<array_type>::value_type>::type, typename traits<array_type>::backend_type>;
    ASSERT(array.size(0) == array.size(1), "Failed to construct upper hessenberg view of dense tensor.  The tensor must be a square matrix.");
    CALL_AND_HANDLE(return result_type(morder, array.size(), array.buffer(), array.size(0), array.size(1)), "Failed to return upper_hessenberg view of dense tensor.");
}

template <typename array_type>
typename std::enable_if<is_dense_matrix<array_type>::value && !traits<array_type>::is_mutable, upper_hessenberg_matrix<typename std::add_const<typename traits<array_type>::value_type>::type, typename traits<array_type>::backend_type>>::type
upper_hessenberg_view(array_type& array, MATRIX_ORDERING morder = MATRIX_ORDERING::ROW_MAJOR)
{
    using result_type = upper_hessenberg_matrix<typename std::add_const<typename traits<array_type>::value_type>::type, typename traits<array_type>::backend_type>;
    ASSERT(array.size(0) == array.size(1), "Failed to construct upper hessenberg view of dense tensor.  The tensor must be a square matrix.");
    CALL_AND_HANDLE(return result_type(morder, array.size(), array.buffer(), array.size(0), array.size(1)), "Failed to return upper_hessenberg view of dense tensor.");
}

template <typename array_type>
typename std::enable_if<is_dense_matrix<array_type>::value && traits<array_type>::is_mutable, upper_hessenberg_matrix<typename traits<array_type>::value_type, typename traits<array_type>::backend_type>>::type
upper_hessenberg_view(array_type& array, MATRIX_ORDERING morder = MATRIX_ORDERING::ROW_MAJOR)
{
    using result_type = upper_hessenberg_matrix<typename traits<array_type>::value_type, typename traits<array_type>::backend_type>;
    ASSERT(array.size(0) == array.size(1), "Failed to construct upper hessenberg view of dense tensor.  The tensor must be a square matrix.");
    CALL_AND_HANDLE(return result_type(morder, array.size(), array.buffer(), array.size(0), array.size(1)), "Failed to return upper_hessenberg view of dense tensor.");
}

//reinterpreted tensor shape view
template <typename array_type, typename ... Args>
typename std::enable_if<is_dense_tensor<array_type>::value, reinterpreted_tensor<typename std::add_const<typename traits<array_type>::value_type>::type, sizeof...(Args), typename traits<array_type>::backend_type>>::type
reinterpret_shape(const array_type& array, Args&& ... args)
{
    using result_type = reinterpreted_tensor<typename std::add_const<typename traits<array_type>::value_type>::type, sizeof...(Args), typename traits<array_type>::backend_type>;
    CALL_AND_HANDLE(return result_type(array.size(), true, array.buffer(), std::forward<Args>(args)...), "Failed to return reinterpreted tensor view of dense tensor.");
}

template <typename array_type, typename ... Args>
typename std::enable_if<is_dense_tensor<array_type>::value && !traits<array_type>::is_mutable, reinterpreted_tensor<typename std::add_const<typename traits<array_type>::value_type>::type, sizeof...(Args), typename traits<array_type>::backend_type>>::type
reinterpret_shape(array_type& array, Args&& ... args)
{
    using result_type = reinterpreted_tensor<typename std::add_const<typename traits<array_type>::value_type>::type, sizeof...(Args), typename traits<array_type>::backend_type>;
    CALL_AND_HANDLE(return result_type(array.size(), true, array.buffer(), std::forward<Args>(args)...), "Failed to return reinterpreted tensor view of dense tensor.");
}

template <typename array_type, typename ... Args>
typename std::enable_if<is_dense_tensor<array_type>::value && traits<array_type>::is_mutable, reinterpreted_tensor<typename traits<array_type>::value_type, sizeof...(Args), typename traits<array_type>::backend_type>>::type
reinterpret_shape(array_type& array, Args&& ... args)
{
    using result_type = reinterpreted_tensor<typename traits<array_type>::value_type, sizeof...(Args), typename traits<array_type>::backend_type>;
    CALL_AND_HANDLE(return result_type(array.size(), true, array.buffer(), std::forward<Args>(args)...), "Failed to return reinterpreted tensor view of dense tensor.");
}


//reinterpreted tensor capacity view
template <typename array_type, typename ... Args>
typename std::enable_if<is_dense_tensor<array_type>::value, reinterpreted_tensor<typename std::add_const<typename traits<array_type>::value_type>::type, sizeof...(Args), typename traits<array_type>::backend_type>>::type
reinterpret_capacity(const array_type& array, Args&& ... args)
{
    using result_type = reinterpreted_tensor<typename std::add_const<typename traits<array_type>::value_type>::type, sizeof...(Args), typename traits<array_type>::backend_type>;
    CALL_AND_HANDLE(return result_type(array.capacity(), false, array.buffer(), std::forward<Args>(args)...), "Failed to return reinterpreted tensor view of dense tensor.");
}

template <typename array_type, typename ... Args>
typename std::enable_if<is_dense_tensor<array_type>::value && !traits<array_type>::is_mutable, reinterpreted_tensor<typename std::add_const<typename traits<array_type>::value_type>::type, sizeof...(Args), typename traits<array_type>::backend_type>>::type
reinterpret_capacity(array_type& array, Args&& ... args)
{
    using result_type = reinterpreted_tensor<typename std::add_const<typename traits<array_type>::value_type>::type, sizeof...(Args), typename traits<array_type>::backend_type>;
    CALL_AND_HANDLE(return result_type(array.capacity(), false, array.buffer(), std::forward<Args>(args)...), "Failed to return reinterpreted tensor view of dense tensor.");
}

template <typename array_type, typename ... Args>
typename std::enable_if<is_dense_tensor<array_type>::value && traits<array_type>::is_mutable, reinterpreted_tensor<typename traits<array_type>::value_type, sizeof...(Args), typename traits<array_type>::backend_type>>::type
reinterpret_capacity(array_type& array, Args&& ... args)
{
    using result_type = reinterpreted_tensor<typename traits<array_type>::value_type, sizeof...(Args), typename traits<array_type>::backend_type>;
    CALL_AND_HANDLE(return result_type(array.capacity(), false, array.buffer(), std::forward<Args>(args)...), "Failed to return reinterpreted tensor view of dense tensor.");
}


}   //namespace linalg

#endif  //LINALG_TENSOR_VIEW_HPP
