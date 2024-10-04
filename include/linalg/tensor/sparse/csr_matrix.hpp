#ifndef LINALG_CSR_MATRIX_HPP
#define LINALG_CSR_MATRIX_HPP

#include <vector>
#include <tuple>
#include <algorithm>
#include "../../linalg_forward_decl.hpp"

namespace linalg
{

template <typename backend>
class csr_topology_type
{
public:
    using backend_type = backend;
    using size_type = typename backend_type::size_type;   
    using index_type = typename backend_type::index_type;
    using index_pointer = typename std::add_pointer<index_type>::type;    using const_index_pointer = typename std::add_pointer<typename std::add_const<index_type>::type >::type;
    using shape_type = std::array<size_type, 2>;
    using self_type = csr_topology_type<backend_type>;

    template <typename U> friend class csr_matrix_base;
    template <typename U> friend class csr_topology_type;
protected:
    using iallocator = memory::allocator<index_type, backend_type>;    template <typename srcbck> using memtransfer = memory::transfer<srcbck, backend_type>;

    index_pointer m_colind;                             ///< The column indices of the values stored in the matrix
    index_pointer m_rowptr;                             ///< The cumulative number of elements in each row padded with an initial 0
    size_type m_nnz;                                    ///< The total number of non-zero elements in the csr matrix
    size_type m_max_nnz;                                ///< The maximum number of non-zero elements that the allocated buffers can store
    size_type m_max_rows;                               ///< The maximum number of rows that the matrix could have
    shape_type m_shape;                                 ///< A static array of size 2 that stores the shape of the csr_matrix

public:
    csr_topology_type() : m_colind{nullptr}, m_rowptr{nullptr}, m_nnz{0}, m_max_nnz{0}, m_max_rows{0}, m_shape{{0,0}} {}
    csr_topology_type(size_type nnz, size_type nrows, size_type ncols) : m_colind{nullptr}, m_rowptr{nullptr}, m_nnz{0}, m_max_nnz{nnz}, m_max_rows{nrows}, m_shape{{nrows,ncols}}
    {
        CALL_AND_HANDLE(m_colind = iallocator::allocate(m_max_nnz),"Failed to construct csr topology object.  Column index buffer allocation failed.");
        CALL_AND_HANDLE(m_rowptr = iallocator::allocate(m_max_rows+1),"Failed to construct csr topology object.  Row pointer buffer allocation failed.");
        m_rowptr[0] = 0;
    }

    //copy constructor
    csr_topology_type(const self_type& src){CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy construct csr topology object.");}
    template <typename bck, typename = typename std::enable_if<!std::is_same<bck, backend_type>::value, void>::type>
    csr_topology_type(const csr_topology_type<bck>& src){CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy construct csr topology object.");}

    //move constructor
    csr_topology_type(self_type&& src) noexcept : m_colind{src.m_colind}, m_rowptr{src.m_rowptr}, m_nnz{std::move(src.m_nnz)}, m_max_nnz{std::move(src.m_max_nnz)}, m_max_rows{std::move(src.m_max_rows)}, m_shape(std::move(src.m_shape))
    {
        src.m_colind = nullptr; src.m_rowptr = nullptr; src.m_nnz = 0; src.m_max_nnz = 0; src.m_max_rows = 0; for(size_type i=0; i<2; ++i){src.m_shape[i] = 0;}
    }

    ~csr_topology_type()
    {
        try{if(m_colind != nullptr){iallocator::deallocate(m_colind);}}catch(...){}
        try{if(m_rowptr != nullptr){iallocator::deallocate(m_rowptr);}}catch(...){}
    }

    //copy assignment
    self_type& operator=(const self_type& src){if(&src != this){CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy assign csr topology base type.");} return *this;}
    template <typename bck, typename = typename std::enable_if<!std::is_same<bck, backend_type>::value, void>::type>
    inline self_type& operator=(const csr_topology_type<bck>& src){CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy assign csr topology base type.");   return *this;}

    //need to implement move assignment operators


    inline self_type& swap(self_type& src) noexcept 
    {
        using std::swap;
        if(this != &src)
        {
            swap(m_nnz, src.m_nnz);     swap(m_max_nnz, src.m_max_nnz);   swap(m_max_rows, src.m_max_rows);     swap(m_shape, src.m_shape);
            index_pointer temp = src.m_colind;  src.m_colind = m_colind;  m_colind = temp;     
            temp = src.m_rowptr;  src.m_rowptr = m_rowptr;  m_rowptr = temp;     
        }
        return *this;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    //             Functions for allocating and deallocating csr topology buffers            //
    ///////////////////////////////////////////////////////////////////////////////////////////
    inline void deallocate()
    {
        if(m_rowptr != nullptr){CALL_AND_HANDLE(iallocator::deallocate(m_rowptr), "Failed to deallocate csr topology object.  Error when deallocating rowptr buffer.");}    m_rowptr = nullptr;
        if(m_colind != nullptr){CALL_AND_HANDLE(iallocator::deallocate(m_colind), "Failed to deallocate csr topology object.  Error when deallocating colind buffer.");}    m_colind = nullptr;
        m_nnz = 0; m_max_nnz = 0; m_max_rows = 0; m_shape[0] = 0; m_shape[1] = 0; 
    }

    inline void allocate(size_type nnz, size_type nrows)
    {
        ASSERT(m_rowptr == nullptr && m_colind == nullptr, "Unable to allocate csr topology object.  It has already been allocated.");
        CALL_AND_HANDLE(m_colind = iallocator::allocate(nnz),"Failed to allocate csr matrix object.  Column index buffer allocation failed.");
        CALL_AND_HANDLE(m_rowptr = iallocator::allocate(nrows+1),"Failed to allocate csr matrix object.  Row pointer buffer allocation failed.");
        m_max_nnz = nnz;    m_max_rows = nrows;
    }

    inline void reallocate(size_type nnz, size_type nrows)
    {
        CALL_AND_HANDLE(deallocate(), "Failed to reallocate csr topology object.  Error when deallocating previously allocated csr topology object.");
        CALL_AND_HANDLE(allocate(nnz, nrows), "Failed to reallocate csr topology object.  Error when allocate new buffers.");
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    //                  Functions for resizing the csr matrix topology object                //
    ///////////////////////////////////////////////////////////////////////////////////////////
    inline void resize(size_type nrows, size_type ncols)
    {
        if(nrows > m_max_rows)
        {
            if(m_rowptr != nullptr){CALL_AND_HANDLE(iallocator::deallocate(m_rowptr), "Failed to resize the csr topology object.  Failed to deallocate the previous rowptr buffer object.");}
            CALL_AND_HANDLE(m_rowptr = iallocator::allocate(nrows+1), "Failed to resize the csr topology object.  Failed to allocate the new rowptr buffer when an increase in size is required.");     
            m_max_rows = nrows;
        }
        m_shape[0] = nrows; m_shape[1] = ncols;
    }
    inline bool resize(size_type nnz)
    {
        m_nnz = nnz;
        if(nnz > m_max_nnz)
        {
            if(m_colind != nullptr){CALL_AND_HANDLE(iallocator::deallocate(m_colind), "Failed to resize the csr topology object.  Failed to deallocate the previous colind buffer object.");}
            CALL_AND_HANDLE(m_colind = iallocator::allocate(nnz), "Failed to resize the csr topology object.  Failed to allocate the new colind buffer when an increase in size is required.");     
            m_max_nnz = nnz;
            return true;
        }
        return false;
    }
    inline void resize(size_type nnz, size_type nrows, size_type ncols){CALL_AND_RETHROW(resize(nnz));  CALL_AND_RETHROW(resize(nrows, ncols));}

    ///////////////////////////////////////////////////////////////////////////////////////////
    //Shrink all of the buffers so that their total capacity is the same as there size.  This//
    //                       keeps all of the values in the buffers.                         //
    ///////////////////////////////////////////////////////////////////////////////////////////
    inline void shrink_to_fit()
    {
        if(m_nnz < m_max_nnz)
        {
            index_pointer temp;    CALL_AND_HANDLE(temp = iallocator::allocate(m_nnz), "Failed to resize the csr topology object.  Failed to allocate the new colind buffer when an increase in size is required.");     
            CALL_AND_HANDLE(memtransfer<backend_type>::copy(m_colind, m_nnz, temp), "Failed to shrink the csr topology object.  Failed when copying the current colind buffer into the new colind buffer.");
            CALL_AND_HANDLE(iallocator::deallocate(m_colind), "Failed to shrink the csr topology object.  Failed to deallocate the previous colind buffer object.");
            m_colind = temp;    temp = nullptr;

            m_max_nnz = m_nnz;
        }
        if(m_shape[0] < m_max_rows)
        {
            //here we actually need to grow the rowptr array
            index_pointer temp;    CALL_AND_HANDLE(temp = iallocator::allocate(m_shape[0]+1), "Failed to resize the csr topology object.  Failed to allocate the new rowptr buffer when an increase in size is required.");     
            CALL_AND_HANDLE(memtransfer<backend_type>::copy(m_rowptr, m_shape[0]+1, temp), "Failed to shrink the csr topology object.  Failed when copying the current rowptr buffer into the new rowptr buffer.");
            CALL_AND_HANDLE(iallocator::deallocate(m_rowptr), "Failed to shrink the csr topology object.  Failed to deallocate the previous rowptr buffer object.");
            m_rowptr = temp;    temp = nullptr;
            m_max_rows = m_shape[0];
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    //      Functions for setting the values in the buffers of the csr topology object       //
    ///////////////////////////////////////////////////////////////////////////////////////////
    void set_colind(const std::vector<index_type>& colind)
    {
        ASSERT(colind.size() == m_nnz, "Failed to set csr topology colind from vector.  The two buffers are not the same size.");
        //check that this input colind array is correctly formated (with each row being sorted).
        for(size_type i=0; i<m_shape[0](); ++i)
        {
            index_type pval = colind[m_rowptr[i]];
            for(index_type j=m_rowptr[i]+1; j<m_rowptr[i+1]; ++j){ASSERT(colind[j] > pval, "Failed to set csr topology colind from vector.  The colind array is not correctly formatted.");}
        }
        CALL_AND_HANDLE(memtransfer<blas_backend>::copy(&colind[0], m_nnz, m_colind), "Failed to set csr topology colind from vector.  Failed when copying the current colind buffer into the new colind buffer.");
    }
    void set_rowptr(const std::vector<index_type>& rowptr)
    {
        ASSERT(rowptr.size() == m_shape[0]+1, "Failed to set csr topology rowptr from vector.  The two buffers are not the same size.");
        for(index_type i = 0; i<rowptr.size()-1; ++i){ASSERT(rowptr[i] <= rowptr[i+1], "Failed to set csr topology rowptr from vector.  An element of the rowptr array is smaller than its proceeding value.");}
        CALL_AND_HANDLE(memtransfer<blas_backend>::copy(&rowptr[0], m_nnz, m_rowptr), "Failed to set csr topology rowptr from vector.  Failed when copying the current rowptr buffer into the new rowptr buffer.");
    }

    bool is_allocated() const{return m_rowptr != nullptr;}


#ifdef CEREAL_LIBRARY_FOUND
    ///////////////////////////////////////////////////////////////////////////////////////////
    //         functions for serialising and deserialising the csr topology object.          //
    ///////////////////////////////////////////////////////////////////////////////////////////
    using ibuffer_reader_type = internal::buffer_reader_wrapper<index_type, backend_type>;
    using ibuffer_writer_type = internal::buffer_writer_wrapper<index_type, backend_type>;

    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("colind", ibuffer_writer_type{m_colind, m_max_nnz})), "Failed to serialise csr topology object.  Error when serialising the colinds buffer.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("rowptr", ibuffer_writer_type{m_rowptr, m_max_rows+1})), "Failed to serialise csr topology object.  Error when serialising the rowptr buffer.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("shape", m_shape)), "Failed to serialise csr topology object.  Failed to serialise the csr topology shape.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("nnz", m_nnz)), "Failed to serialise csr topology object.  Error when serialising the number of non-zero elements in the matrix.");
    }
    
    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("colind", ibuffer_reader_type{&m_colind, &m_max_nnz})), "Failed to deserialise csr topology object.  Error when deserialising the colinds buffer.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("rowptr", ibuffer_reader_type{&m_rowptr, &m_max_rows})), "Failed to deserialise csr topology object.  Error when deserialising the rowptr buffer.");
        m_max_rows = m_max_rows-1;
        CALL_AND_HANDLE(ar(cereal::make_nvp("shape", m_shape)), "Failed to deserialise csr topology object.  Failed to deserialise the csr topology shape.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("nnz", m_nnz)), "Failed to deserialise csr topology object.  Error when deserialising the number of non-zero elements in the matrix.");
    }
#endif

    ///////////////////////////////////////////////////////////////////////////////////////////
    //         functions for serialising and deserialising the csr topology object.          //
    ///////////////////////////////////////////////////////////////////////////////////////////
    bool compare_topology(const self_type& other) const
    {
        if(m_nnz != other.m_nnz){return false;}
        if(m_shape != other.m_shape){return false;}
        bool is_equal; CALL_AND_HANDLE(is_equal = backend_type::is_equal(m_colind, other.m_colind, m_nnz), "Failed to compare two csr topology objects.  Failed to determine if the column indices arrays are equal."); if(!is_equal){return false;}
        CALL_AND_HANDLE(is_equal = backend_type::is_equal(m_rowptr, other.m_rowptr, m_shape[0]+1), "Failed to compare two csr topology objects.  Failed to determine if the column indices arrays are equal."); if(!is_equal){return false;}
        return true;
    }

private:
    template <typename bck>
    inline bool copy_assign_impl(const csr_topology_type<bck>& src)
    {
        bool resized_buffers = src.m_nnz > m_max_nnz;
        CALL_AND_HANDLE(resize(src.m_nnz, src.m_shape[0], src.m_shape[1]), "Failed to copy assign csr topology object.  Failed to resize buffers so that they could fit the src matrix."); 
        CALL_AND_HANDLE(memtransfer<bck>::copy(src.m_colind, m_nnz, m_colind), "Failed to copy assign the csr topology object.  Failed when copying the src colind buffer into the int m_colind.");
        CALL_AND_HANDLE(memtransfer<bck>::copy(src.m_rowptr, m_shape[0]+1, m_rowptr), "Failed to copy assign the csr topology object.  Failed when copying the src rowptr buffer into the int m_rowptr.");
        return resized_buffers;
    }
};


template <typename backend>
bool operator==(const csr_topology_type<backend>& a, const csr_topology_type<backend>& b){CALL_AND_RETHROW(return a.compare_topology(b));}


template <typename csr_impl>
class csr_matrix_base : public csr_matrix_type
{
public:
    using backend_type = typename traits<csr_impl>::backend_type;
    using topology_type = csr_topology_type<backend_type>;

    using value_type = typename traits<csr_impl>::value_type;
    using real_type = typename linalg::get_real_type<value_type>::type;
    using size_type = typename topology_type::size_type;   
    using index_type = typename topology_type::index_type;
    using self_type = csr_matrix_base<csr_impl>;

    using pointer = typename std::add_pointer<value_type>::type;    using const_pointer = typename std::add_pointer<typename std::add_const<value_type>::type >::type;
    using index_pointer = typename topology_type::index_pointer;    using const_index_pointer = typename topology_type::const_index_pointer;

    static constexpr size_type rank = traits<csr_impl>::rank;       ///< Returns the rank (dimensionality) of the csr_matrix
    using shape_type = std::array<size_type, rank>;

    //the classes used for memory allocation and transfer
    using allocator = memory::allocator<value_type, backend_type>;  using memfill = memory::filler<value_type, backend_type>;
    template <typename srcbck> using memtransfer = memory::transfer<srcbck, backend_type>;

    using coo_type = std::vector<std::tuple<index_type, index_type, value_type>>;
protected:
    pointer m_vals;                                     ///< The 1-dimensional array used to store the values present in the matrix
    topology_type m_topo;
    
public:
    //Default constructor
    csr_matrix_base() : m_vals(nullptr), m_topo() {}
    csr_matrix_base(size_type _nnz, size_type _nrows, size_type _ncols = 0) try : m_vals(nullptr), m_topo(_nnz, _nrows, _ncols)
    {CALL_AND_HANDLE(m_vals = allocator::allocate(_nnz),"Failed to construct csr matrix object.  Value buffer allocation failed.");}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct csr matrix object.");}

    //need to update the copy assignment operators
    csr_matrix_base(const self_type& src) : csr_matrix_base() {CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy construct csr matrix base object.");}
    template <typename Container, typename = other_copy_constructable_type<Container, self_type> >
    csr_matrix_base(const Container& src) : csr_matrix_base(){CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy construct csr matrix base object.");}

    csr_matrix_base(self_type&& src) noexcept : m_vals{src.m_vals}, m_topo(std::move(src.m_topo)){src.m_vals = nullptr;}
    template <typename Container, typename = other_move_constructable_type<Container, self_type>> 
    csr_matrix_base(Container&& src) : csr_matrix_base() {CALL_AND_HANDLE(move_assign_impl(std::forward<Container>(src)), "Failed to construct csr matrix base object from expression.");}

    //construct from three std::vector containing the vals, colind and rowptr arrays and the number of columns.  This assumes that the
    //input csr matrix is a square matrix
    csr_matrix_base(const std::vector<value_type>& _vals, const std::vector<index_type>& _colinds, const std::vector<index_type>& _rowptr, size_type _ncols = 0) : csr_matrix_base()
    {
        CALL_AND_HANDLE(init(_vals, _colinds, _rowptr, _ncols), "Failed to construct csr_matrix_base object.  Error when constructing from vectors.");
    }

    csr_matrix_base(const std::vector<std::tuple<index_type, index_type, value_type> >& coo, size_type _nrows = 0, size_type _ncols = 0) : csr_matrix_base()
    {
        CALL_AND_HANDLE(init(coo, _nrows, _ncols), "Failed to construct csr_matrix_base object.  Error when constructing from coordinate list.");
    }

    ~csr_matrix_base()
    {
        try{if(m_vals != nullptr){allocator::deallocate(m_vals);}}catch(...){}
    }

    self_type& operator=(const self_type& src){if(&src != this){CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy assign csr matrix base type.");} return *this;}
    template <typename Container> inline other_copy_assignable_type<Container, self_type> operator=(const Container& src){CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy assign csr matrix base type.");  return *this;}

    self_type& operator=(self_type&& src)
    {
        if(&src != this)
        {
            if(m_vals != nullptr){deallocate();}
            m_vals = src.m_vals;    src.m_vals = nullptr;
            m_topo = std::move(src.m_topo);
        }
        return *this;
    }
    template <typename Container> inline other_move_assignable_type<Container, self_type> operator=(Container&& src){CALL_AND_RETHROW(return move_assign_impl(std::forward<Container>(src)));}

    //we need to implement the move assignment operator here.
 

    inline self_type& swap(self_type& src) noexcept 
    {
        if(this != &src){pointer temp = src.m_vals;  src.m_vals = m_vals;  m_vals = temp;     m_topo.swap(src.m_topo);}
        return *this;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    //             Functions for allocating and deallocating csr matrix buffers              //
    ///////////////////////////////////////////////////////////////////////////////////////////
    inline void deallocate()
    {
        if(m_vals != nullptr){CALL_AND_HANDLE(allocator::deallocate(m_vals), "Failed to deallocate csr matrix object.  Error when deallocating vals buffer.");}  m_vals = nullptr;    
        CALL_AND_HANDLE(m_topo.deallocate(), "Failed to deallocate csr matrix object. Failed to deallocate topology object.");
    }

    inline void allocate(size_type _nnz, size_type _nrows)
    {
        ASSERT(m_vals == nullptr, "Unable to allocate csr matrix object.  It has already been allocated.");
        CALL_AND_HANDLE(m_topo.allocate(_nnz, _nrows), "Failed to allocate csr matrix object.  Failed to allocate topology object.");
        CALL_AND_HANDLE(m_vals = allocator::allocate(_nnz),"Failed to allocate csr matrix object.  Value buffer allocation failed.");
    }

    inline void reallocate(size_type _nnz, size_type _nrows)
    {
        CALL_AND_HANDLE(deallocate(), "Failed to reallocate csr matrix object.  Error when deallocating previously allocated csr matrix object.");
        CALL_AND_HANDLE(allocate(_nnz, _nrows), "Failed to reallocate csr matrix object.  Error when allocate new buffers.");
    }

    inline void clear(){CALL_AND_HANDLE(deallocate(), "Failed to clear csr matrix object.");}

    //reallocate will decrease the size of the array if it is not the correct size
    void resize(size_type _nnz)
    {
        bool resized;   CALL_AND_HANDLE(resized = m_topo.resize(_nnz), "Failed to resize the csr matrix object.  Failed when resizing the topology array.");
        if(resized){CALL_AND_RETHROW(resize_buffer(_nnz));}
    }
    template <typename backend> void resize(const csr_topology_type<backend>& _topology)
    {
        if(_topology.m_nnz > m_topo.m_max_nnz){CALL_AND_HANDLE(resize_buffer(_topology.m_nnz), "Failed to resize csr matrix object from topology object.  Failed to resize the values buffer.");}
        CALL_AND_HANDLE(m_topo = _topology, "Failed to resize csr matrix object from topology object.  Topology assignment failed.");
    }
    void resize(size_type _nrows, size_type _ncols){CALL_AND_HANDLE(m_topo.resize(_nrows, _ncols), "Failed to resize the csr matrix object.  Failed when resizing the topology array.");}
    void resize(size_type _nnz, size_type _nrows, size_type _ncols){CALL_AND_RETHROW(resize(_nnz));  CALL_AND_RETHROW(resize(_nrows, _ncols));}
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    //Shrink all of the buffers so that their total capacity is the same as there size.  This//
    //                       keeps all of the values in the buffers.                         //
    ///////////////////////////////////////////////////////////////////////////////////////////
    inline void shrink_to_fit()
    {
        if(m_topo.m_nnz < m_topo.m_max_nnz)
        {
            value_type* temp2;    CALL_AND_HANDLE(temp2 = allocator::allocate(m_topo.m_nnz), "Failed to resize the csr matrix object.  Failed to allocate the new valus buffer when an increase in size is required.");     
            CALL_AND_HANDLE(memtransfer<backend_type>::copy(m_vals, m_topo.m_nnz, temp2), "Failed to shrink the csr matrix object.  Failed when copying the current vals buffer into the new vals buffer.");
            CALL_AND_HANDLE(allocator::deallocate(m_vals), "Failed to shrink the csr matrix object.  Failed to deallocate the previous vals buffer object.");
            m_vals = temp2;    temp2 = nullptr;
        }
        CALL_AND_HANDLE(m_topo.shrink_to_fit(), "Failed to shrink the csr matrix object.  Error when shrinking the topology buffers.");
    }


    ///////////////////////////////////////////////////////////////////////////////////////////
    //        Functions for setting the values in the buffers of the csr matrix object       //
    ///////////////////////////////////////////////////////////////////////////////////////////
    void set_vals(const std::vector<value_type>& _vals)
    {
        ASSERT(_vals.size() == m_topo.m_nnz, "Failed to set csr matrix vals from vector.  The two buffers are not the same size.");
        CALL_AND_HANDLE(memtransfer<blas_backend>::copy(&_vals[0], m_topo.m_nnz, m_vals), "Failed to set csr matrix vals from vector.  Failed when copying the current vals buffer into the new vals buffer.");
    }
    void set_colind(const std::vector<index_type>& _colind){CALL_AND_HANDLE(m_topo.set_colind(_colind), "Failed to set csr matrix colind from buffer.  The two buffers are not the same size.");}
    void set_rowptr(const std::vector<index_type>& _rowptr){CALL_AND_HANDLE(m_topo.set_rowptr(_rowptr), "Failed to set csr matrix rowptr from buffer.  The two buffers are not the same size.");}


    ///////////////////////////////////////////////////////////////////////////////////////////
    //              functions for accessing the size of the csr matrix object.               //
    ///////////////////////////////////////////////////////////////////////////////////////////
    inline const size_type& capacity() const{return m_topo.m_max_nnz;}
    inline const size_type& max_nnz() const{ return m_topo.m_max_nnz;}
    inline const size_type& max_rows() const{ return m_topo.m_max_rows;}

    inline const size_type& nnz() const{return m_topo.m_nnz;}
    inline const size_type& nelems() const {return m_topo.m_nnz;}
    inline const size_type& size() const{ return m_topo.m_nnz;}

    inline const size_type& size(size_type i) const{    ASSERT(i < rank, "Unable to access csr_matrix size element.  Index out of bounds.");  return m_topo.m_shape[i];}
    inline const size_type& dims(size_type i) const{    ASSERT(i < rank, "Unable to access csr_matrix size element.  Index out of bounds.");  return m_topo.m_shape[i];}
    inline const size_type& shape(size_type i) const{   ASSERT(i < rank, "Unable to access csr_matrix size element.  Index out of bounds.");  return m_topo.m_shape[i];}

    inline const size_type& nrows() const{return m_topo.m_shape[0];}
    inline size_type& ncols(){return m_topo.m_shape[1];}
    inline const size_type& ncols() const{return m_topo.m_shape[1];}

    inline const shape_type& dims() const{return m_topo.m_shape;}
    inline const topology_type& shape() const{return m_topo;}
    inline const topology_type& topology() const{return m_topo;}
    inline bool same_shape(const topology_type& _shape) const{return _shape == m_topo;}

#ifdef CEREAL_LIBRARY_FOUND
    ///////////////////////////////////////////////////////////////////////////////////////////
    //          functions for serialising and deserialising the csr matrix object.           //
    ///////////////////////////////////////////////////////////////////////////////////////////
    using buffer_reader_type = internal::buffer_reader_wrapper<value_type, backend_type>;
    using buffer_writer_type = internal::buffer_writer_wrapper<value_type, backend_type>;

    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::make_nvp("buffer", buffer_writer_type{m_vals, m_topo.m_max_nnz})), "Failed to serialise csr matrix object.  Error when serialising the data buffer.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("topology", m_topo)), "Failed to serialise csr matrix object.  Error when serialising the topology object.");
    }
    
    template <typename archive>
    void load(archive& ar)
    {
        size_type _max_nnz;
        CALL_AND_HANDLE(ar(cereal::make_nvp("buffer", buffer_reader_type{&m_vals, &_max_nnz})), "Failed to deserialise csr matrix object.  Error when deserialising the data buffer.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("topology", m_topo)), "Failed to deserialise csr matrix object.  Error when deserialising the topology object.");
        ASSERT(_max_nnz == m_topo.m_max_nnz, "Failed do deserialise the values buffer and buffer object do not have the same maximum number of non-zeros.");
    }
#endif

    ///////////////////////////////////////////////////////////////////////////////////////////
    //  functions for performing scalar multiplication and division on a csr matrix object.  //
    ///////////////////////////////////////////////////////////////////////////////////////////
    template <typename Vt> inline value_update_type<Vt, self_type> operator*=(const Vt& v){CALL_AND_HANDLE(backend_type::scal(m_topo.m_nnz, value_type(v), m_vals, 1), "Failed to perform operator*= on csr matrix object.  scal call failed.");      return *this;}
    template <typename Vt> inline value_update_type<Vt, self_type> operator/=(const Vt& v){CALL_AND_HANDLE(backend_type::scal(m_topo.m_nnz, value_type(1.0/v), m_vals, 1), "Failed to perform operator/= on csr matrix object.  scal call failed.");  return *this;}

    ///////////////////////////////////////////////////////////////////////////////////////////
    //     functions providing access to the raw buffers storing the csr matrix object.      //
    ///////////////////////////////////////////////////////////////////////////////////////////
    inline pointer buffer(){return m_vals;}
    inline const_pointer buffer()const{return m_vals;}
    inline pointer data(){return m_vals;}
    inline const_pointer data()const{return m_vals;}

    inline index_pointer rowptr(){return m_topo.m_rowptr;}
    inline const_index_pointer rowptr() const{return m_topo.m_rowptr;}

    inline index_pointer colind(){return m_topo.m_colind;}
    inline const_index_pointer colind() const{return m_topo.m_colind;}

    ///////////////////////////////////////////////////////////////////////////////////////////
    // functions for initialising a csr matrix object from values, column indices and rowptr //
    //                                       arrays.                                         //
    ///////////////////////////////////////////////////////////////////////////////////////////
    void init(const std::vector<value_type>& _vals, const std::vector<index_type>& _colinds, const std::vector<index_type>& _rowptr, size_type _ncols = 0)
    {
        if(m_topo.is_allocated()){deallocate();}

        ASSERT(_vals.size() == _colinds.size(), "Failed to initialise csr_matrix_base object.  The input vals and colind vectors do not have the same size.");
        ASSERT(_rowptr[0] == 0, "Failed to initialise csr_matrix_base object.  The first element of the rowptr array is not zero.");
        ASSERT(_vals.size() == static_cast<size_type>(_rowptr[_rowptr.size()-1]), "Failed to initialise csr_matrix_base object.  The final element of the rowptr array is not equal to the number of non-zeros.");

        size_type max_per_row = 0;
        //check that the rowptr array is nondecreasing
        for(size_type i = 0; i<_rowptr.size()-1; ++i)
        {
            ASSERT(_rowptr[i] <= _rowptr[i+1], "Failed to initialise csr_matrix_base object.  An element of the rowptr array is smaller than its proceeding value.");
            size_type ninrow = _rowptr[i+1] - _rowptr[i];
            if(ninrow > max_per_row){max_per_row = ninrow;}
        }

        //first we assort the vals and colinds so that for a given row the specified colinds are sorted in ascending value (to allow for more efficient use of the csr_matrix.)
        //to do this we will allocate a small vector that helps us process each row.
        std::vector<index_type> row_indexer(max_per_row);
        std::vector<index_type> row_colinds(max_per_row);
        std::vector<value_type> row_vals(max_per_row);

        m_topo.m_shape[1] = _ncols;
        //now we validate that there are no duplicate column indices in a row
        for(size_type i=0; i<_rowptr.size()-1; ++i)
        {
            size_type ninrow = _rowptr[i+1] - _rowptr[i];
            for(size_type j=0; j<ninrow; ++j){row_indexer[j] = j;   row_colinds[j] = _colinds[j+_rowptr[i]];  }

            //sort the row_colinds vector and search for any duplicates
            CALL_AND_HANDLE(std::sort(row_colinds.begin(), row_colinds.begin()+ninrow), "Failed to initialise csr_matrix_base object.  Failed to sort column indices when searching for duplicates.");

            index_type prev_val = row_colinds[0]; row_colinds[0] = _colinds[_rowptr[i]];
            for(size_type j=1; j<ninrow; ++j)
            {
                ASSERT(row_colinds[j] != prev_val, "Failed to initialise csr_matrix_base object.  The colinds array contains duplicate column indices on the same row.");
                prev_val = row_colinds[j];
            }
            for(size_type j=0; j<ninrow; ++j)
            {
                if(_ncols != 0){ASSERT(static_cast<size_type>(row_colinds[j]) < _ncols, "Failed to initialise csr_matrix_base object.  The colinds array contains an element that is out of bounds.");}
                else{m_topo.m_shape[1] = static_cast<size_type>(row_colinds[j])+1 > m_topo.m_shape[1] ? row_colinds[j]+1 : m_topo.m_shape[1];}
            }
        }
        //now that we know that we have a valid format we can go ahead and create the csr matrix objects buffers
        CALL_AND_HANDLE(resize(_vals.size(), _rowptr.size(), m_topo.m_shape[1]), "Failed to initialise csr_matrix_base object.  Failed to resize buffers to fit input.");

        m_topo.m_max_nnz = _vals.size();        m_topo.m_nnz =m_topo.m_max_nnz;
        m_topo.m_max_rows = _rowptr.size()-1;   m_topo.m_shape[0] = m_topo.m_max_rows;    
        
        //copy the rowptr buffer
        CALL_AND_HANDLE(memtransfer<blas_backend>::copy(&_rowptr[0], m_topo.m_max_rows+1, m_topo.m_rowptr), "Failed to initialise csr_matrix_base object.  Failed to copy row pointer buffer.");

        for(size_type i=0; i<_rowptr.size()-1; ++i)
        {
            size_type ninrow = _rowptr[i+1] - _rowptr[i];
            for(size_type j=0; j<ninrow; ++j){row_indexer[j] = j;   row_colinds[j] = _colinds[j+_rowptr[i]];  }

            //now that we know there are no duplicates we can now sort the row_indexer array and use this to reorder our vals and colinds arrays
            CALL_AND_HANDLE(std::sort(row_indexer.begin(), row_indexer.begin()+ninrow, [&row_colinds](size_type lhs, size_type rhs){return row_colinds[lhs] < row_colinds[rhs];}), "Failed to initialise csr_matrix_base object.   Failed to sort the column indices.");

            //now we set the row_colinds array to be correctly sorted
            for(size_type j=0; j<ninrow; ++j)
            {
                row_colinds[j] = _colinds[_rowptr[i] + row_indexer[j]];
                row_vals[j] = _vals[_rowptr[i] + row_indexer[j]];
            }

            CALL_AND_HANDLE(memtransfer<blas_backend>::copy(&row_vals[0], ninrow, m_vals+_rowptr[i]), "Failed to initialise csr_matrix_base object.  Failed to copy src value buffer.");
            CALL_AND_HANDLE(memtransfer<blas_backend>::copy(&row_colinds[0], ninrow, m_topo.m_colind+_rowptr[i]), "Failed to initialise csr_matrix_base object.  Failed to copy src column index buffer.");
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    //     functions for initialising a csr matrix from a vector storing the coo format      //
    ///////////////////////////////////////////////////////////////////////////////////////////
    void init(const std::vector<std::tuple<index_type, index_type, value_type> >& _coo, size_type _nrows, size_type _ncols = 0)
    {
        if(m_topo.is_allocated()){deallocate();}

        //make a copy of the coo vector that we can actually sort
        std::vector<std::tuple<index_type, index_type, value_type> > coo(_coo);

        CALL_AND_HANDLE(std::sort(coo.begin(), coo.end(), [](const std::tuple<index_type, index_type, value_type>& lhs, const std::tuple<index_type, index_type, value_type>& rhs){if(std::get<0>(lhs) != std::get<0>(rhs)){return (std::get<0>(lhs) < std::get<0>(rhs));}else{return (std::get<1>(lhs) < std::get<1>(rhs));}}), "Failed to initialise csr_matrix_base object.  Failed to sort the coordinate form array array.");

        std::array<index_type, 2> coords{std::get<0>(coo[0]), std::get<1>(coo[0])};
        //now check to make sure the coo array forms a valid matrix
        for(size_type i=1; i<coo.size(); ++i)
        {
            std::array<index_type, 2> coordsi{std::get<0>(coo[i]), std::get<1>(coo[i])};
            ASSERT(coords[0] >= 0 && coords[1] >= 0, "Failed to initialise csr_matrix_base_obect.  There was a negative coordinate index in the coordinate array.");
            ASSERT( coords != coordsi, "Failed to initialise csr_matrix_base object.  There is a repeated coordinate in the coordinate array.");
            coords = coordsi;
        }

        std::array<size_type, 2> _shape;
        _shape[0] = _nrows;        _shape[1] = _ncols;
        for(size_type i=0; i<coo.size(); ++i)
        {
            std::array<index_type, 2> coordsi{std::get<0>(coo[i]), std::get<1>(coo[i])};
            if(_ncols != 0){ASSERT(coordsi[1] < static_cast<index_type>(_ncols), "Failed to initialise csr_matrix_base object.  There is a column index which is out of bounds.");}
            else{_shape[1] = static_cast<size_type>(coordsi[1]+1 > static_cast<index_type>(_shape[1]) ? coordsi[1]+1 : _shape[1]);}
            if(_nrows != 0){ASSERT(coordsi[0] < static_cast<index_type>(_nrows), "Failed to initialise csr_matrix_base object.  There is a row index which is out of bounds.");}
            else{_shape[0] = static_cast<size_type>(coordsi[0]+1 > static_cast<index_type>(_shape[0]) ? coordsi[0]+1 : _shape[0]);}
        }

        //now that we have validated the inputs we can actually allocate the required buffers
        size_type _nnz = coo.size();
        CALL_AND_HANDLE(resize(_nnz, _shape[0], _shape[1]), "Failed to initialise csr_matrix_base object.  Failed to resize the underlying buffers.");

        //now we calculate the rowptr array and store it in a vector
        std::vector<index_type> _rowptr(_shape[0]+1, 0);
        for(size_type i=0; i<coo.size(); ++i){++_rowptr[std::get<0>(coo[i])+1];}
        for(size_type i=0; i<_shape[0]; ++i){_rowptr[i+1] += _rowptr[i];}
       
        CALL_AND_HANDLE(memtransfer<blas_backend>::copy(&_rowptr[0], _rowptr.size(), m_topo.m_rowptr), "Failed to initialise csr_matrix_base object.  Failed when copying the current rowptr buffer into the new rowptr buffer.");
        CALL_AND_HANDLE(backend_type::transfer_coo_tuple_to_csr(coo, m_vals, m_topo.m_colind), "Failed to initialise csr_matrix_base object.  Failed when copying the vals and colinds from the coo tuple array.");
    }

private:
    template <typename Container>
    inline typename std::enable_if<is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container& src)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for csr matrix object.  The specified csr matrix is not mutable.");
        bool assign_reallocated;    CALL_AND_HANDLE(assign_reallocated = m_topo.copy_assign_impl(src.m_topo), "Failed to copy assign csr_matrix_base object.  Failed when copy assigning the topology array.");
        if(assign_reallocated){CALL_AND_HANDLE(resize_buffer(src.nnz()), "Failed to copy assign csr_matrix_base object.  Failed to resize buffers so that they could fit the src matrix.");}
        using srcbck = typename traits<Container>::backend_type;
        CALL_AND_HANDLE(memtransfer<srcbck>::copy(src.buffer(), m_topo.m_nnz, m_vals), "Failed to shrink the csr matrix object.  Failed when copying the src value buffer into m_vals.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container& src)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for csr matrix object.  The specified csr matrix is not mutable.");
        static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate copy_assign_impl.");
        bool assign_reallocated;    CALL_AND_HANDLE(assign_reallocated = m_topo.copy_assign_impl(src.m_topo), "Failed to copy assign csr_matrix_base object.  Failed when copy assigning the topology array.");
        if(assign_reallocated){CALL_AND_HANDLE(resize_buffer(src.nnz()), "Failed to copy assign csr_matrix_base object.  Failed to resize buffers so that they could fit the src matrix.");}
        CALL_AND_HANDLE(backend_type::copy_real_to_complex(src.buffer(), m_topo.m_nnz, m_vals), "Failed to shrink the csr matrix object.  Failed when copying the src value buffer into m_vals.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for csr matrix object.  The specified csr matrix is not mutable.");
        bool requires_resize;   CALL_AND_HANDLE(requires_resize = !(expr.shape() == m_topo), "Copy assignment failed. Failed to determine whether the csr matrix requires resizing before evaluation of the input expression.");
        if(requires_resize){CALL_AND_HANDLE(resize(expr.shape()), "Copy assignment failed.  Failed to resize the csr matrix object.");}
        CALL_AND_HANDLE(expr(*this), "Copy assignment failed.  Failed to evaluate the expression into the csr matrix object.");
        return *this;
    }

    template <typename Container>
    inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type move_assign_impl(Container&& expr)
    {
        static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for csr matrix object.  The specified csr matrix is not mutable.");
        bool requires_resize;   CALL_AND_HANDLE(requires_resize = !(expr.shape() == m_topo), "Copy assignment failed. Failed to determine whether the csr matrix requires resizing before evaluation of the input expression.");
        if(requires_resize){CALL_AND_HANDLE(resize(expr.shape()), "Copy assignment failed.  Failed to resize the csr matrix object.");}
        CALL_AND_HANDLE(expr(*this), "Copy assignment failed.  Failed to evaluate the expression into the csr matrix object.");
        return *this;
    }

    void resize_buffer(size_type _nnz)
    {
        if(m_vals != nullptr){CALL_AND_HANDLE(allocator::deallocate(m_vals), "Failed to resize the csr matrix object.  Failed to deallocate the previous vals buffer object.");}
        CALL_AND_HANDLE(m_vals = allocator::allocate(_nnz), "Failed to resize the csr matrix object.  Failed to allocate the new valus buffer when an increase in size is required.");     
    }
};  //class csr_matrix_base



template <typename T> 
class csr_matrix<T, blas_backend> : public csr_matrix_base<csr_matrix<T, blas_backend> >
{
public:
    using self_type = csr_matrix<T, blas_backend>;
    using base_type = csr_matrix_base<self_type>;
    using coo_type = typename base_type::coo_type;
    using size_type = typename base_type::size_type;
    using real_type = typename base_type::real_type;
    using index_type = typename base_type::index_type;

    template <typename U> 
    friend std::ostream& operator<<(std::ostream& out, const csr_matrix<U, blas_backend>& mat);
public:
    template <typename ... Args> csr_matrix(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct csr matrix object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));   return *this;}


    inline size_type nnz_in_row(size_type i) const
    {
        ASSERT(i < this->nrows(), "Unable to get number of terms in row.  Index out of bounds.");
        auto rowptr = this->rowptr();
        return (rowptr[i+1] - rowptr[i]);
    }

    inline bool contains_diagonal(size_type i) const
    {
        ASSERT(i < this->nrows(), "Unable to get number of terms in row.  Index out of bounds.");
        auto rowptr = this->rowptr();
        auto colind = this->colind();
        for(size_type j=static_cast<size_type>(rowptr[i]); j<static_cast<size_type>(rowptr[i+1]); ++j)
        {
            if(static_cast<size_t>(colind[j]) == i){return true;}
        }
        return false;
    }

    //a function for pruning zeros from the csr matrix.  This iterates over the tree and if a value has magnitude less than the tolerance we remove it.
    //This doesn't change the size of any buffers at all
    inline void prune(real_type tol = 1e-12)
    {
        if(tol > 0)
        {
            size_type counter = 0;
            size_t rpi = 0;

            auto buffer = this->buffer();
            auto rowptr = this->rowptr();
            auto colind = this->colind();
            for(size_type i=0; i<this->nrows(); ++i)
            {
                size_t rpi1 = static_cast<size_type>(rowptr[i+1]);
                for(size_type j=rpi; j<rpi1; ++j)
                {
                    //if the absolute value of the current value type is greater than the pruning tolerance we are going to reinsert it
                    //at position counter and incement counter. If a term isn't greater than the pruning tolerance then we are not incrementing
                    //counter and so at a later stage it will be overwritten.
                    if(tol <= linalg::abs(buffer[j]) )
                    {   
                        buffer[counter] = buffer[j];
                        colind[counter] = colind[j];
                        ++counter;
                    }
                }
                rowptr[i+1] = counter;
                rpi = rpi1;
            }
            this->resize(counter);
        }
    }

    inline void transpose(csr_matrix<T, blas_backend>& o, T alpha=T(1)) const
    {
        transpose(*this, o, alpha);
    }

    static inline void transpose(const csr_matrix<T, blas_backend>& imat, csr_matrix<T, blas_backend>& o, T alpha=T(1))
    {
        CALL_AND_HANDLE(o.resize(imat.nnz(), imat.ncols(), imat.nrows()), "Failed to allocate storage for new array.");
        auto ibuffer = imat.buffer();
        auto irowptr = imat.rowptr();
        auto icolind = imat.colind();

        auto obuffer = o.buffer();
        auto orowptr = o.rowptr();
        auto ocolind = o.colind();

        //pad the rowptr array with zeros
        std::fill(orowptr, orowptr + (imat.ncols()+1), index_type(0));
        //iterate over each term in the buffer and use its column index to set the rowptr of the output result
        for(size_type i = 0; i < imat.nnz(); ++i)
        {
            //get the column index in the original array and increment the rowptr object of the transposed array corresponding to this index
            ++orowptr[icolind[i]+1];
        }
        //now increment the rowptr objects so that it is the cumulative sum rather than number of elements in row.  
        //Following this step the output rowptr is completed, but we will be editing it in the next step
        for(size_type i = 0; i < imat.ncols(); ++i){orowptr[i+1] += orowptr[i];}

        size_type rpi = 0;
        //now set up the output data and column indices.  We do this by iterating over each row of the original array.
        for(size_type i=0; i < imat.nrows(); ++i)
        {
            size_type rpi1 = static_cast<size_type>(irowptr[i+1]);
            for(size_type j=rpi; j<rpi1; ++j)
            {
                //for this column index.  Get the current output rowptr value.  Incrementing the results after 
                //we have extracted the value.  Now set the output value and column index
                index_type index = orowptr[icolind[j]]++;
                obuffer[index] = alpha*ibuffer[j];
                ocolind[index] = i;
            }
            rpi = rpi1;
        }

        //now each output rowptr has been incremented by the number of elements in the row so we need to shift everything 
        //along to get the correct structure
        for(size_t i = 0; i < imat.ncols(); ++i)
        {
            size_t j = imat.ncols()-i;
            orowptr[j] = orowptr[j-1];
        }
        orowptr[0] = 0;
    }


    inline matrix<T, blas_backend> todense() const
    {
        matrix<T> ret(this->shape(0), this->shape(1), T(0));
        auto buffer = this->buffer();
        auto rowptr = this->rowptr();
        auto colind = this->colind();

        size_type rpi = 0;
        for(size_t i = 0; i < this->nrows(); ++i)
        {
            size_type rpi1 = static_cast<size_type>(rowptr[i+1]);
            for(size_type j=rpi; j<rpi1; ++j)
            {
                index_type index = colind[j];
                ret(i, index) = buffer[j];
            }
            rpi = rpi1;
        }
        return ret;
    }
};  //csr_matrix<T, blas_backend>


#ifdef __NVCC__
template <typename T> 
class csr_matrix<T, cuda_backend> : public csr_matrix_base<csr_matrix<T, cuda_backend> >
{
public:
    using self_type = csr_matrix<T, cuda_backend>;
    using base_type = csr_matrix_base<self_type>;
    using real_type = typename base_type::real_type;

    using pointer = typename base_type::pointer;    using const_pointer = typename base_type::const_pointer;
    using index_pointer = typename base_type::index_pointer;    using const_index_pointer = typename base_type::const_index_pointer;
    using coo_type = typename base_type::coo_type;
public:
    template <typename ... Args> csr_matrix(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct csr matrix object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));   return *this;}

    inline void transpose(csr_matrix<T, cuda_backend>& o, T alpha=T(1)) const
    {
        RAISE_EXCEPTION("CUDA CSR MATRIX TRANSPOSE NOT IMPLEMENTED.");
    }

    inline matrix<T, blas_backend> todense() const
    {
        csr_matrix<T, blas_backend> ret(*this);
        return ret.todense();
    }
};  //csr_matrix<T, blas_backend>
#endif


template <typename T>
std::ostream& operator<<(std::ostream& out, const csr_matrix<T, blas_backend>& mat)
{
    using size_type = typename csr_matrix<T, blas_backend>::size_type;
    using const_index_pointer = typename csr_matrix<T, blas_backend>::const_index_pointer;
    const_index_pointer rowptr = mat.rowptr();
    const_index_pointer colind = mat.colind();
    for(size_type i=0; i<mat.nrows(); ++i)
    {
        for(size_type j=static_cast<size_type>(rowptr[i]); j<static_cast<size_type>(rowptr[i+1]); ++j){out << i << " " << colind[j] << " " << mat.m_vals[j] << std::endl;}
    }
    return out;
}

}   //namespace linalg

#endif  //LINALG_CSR_MATRIX_HPP//


