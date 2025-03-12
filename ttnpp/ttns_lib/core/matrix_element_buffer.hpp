#ifndef TTNS_MATRIX_ELEMENT_BUFFER_HPP
#define TTNS_MATRIX_ELEMENT_BUFFER_HPP


#include <linalg/linalg.hpp>

namespace ttns
{

namespace helper
{

struct contraction_capacity
{
protected:
    template <typename T, typename U, typename BE1, typename BE2>
    static size_t _contraction_capacity(const ttn_node_data<T, BE1>& a, const ttn_node_data<U, BE2>& b)
    {
        size_t capacity = 1;
        for(size_t nm = 0; nm < a.nmodes(); ++nm)
        {
            capacity *= std::max(a.max_dim(nm), b.max_dim(nm));
        }
        return capacity;
    }

public:
    template <typename Atype, typename Btype>
    static inline size_t evaluate(const Atype& A, const Btype & B)
    {
        size_t capacity = 0;
        ASSERT(A.nset() == B.nset(), "Cannot compute contraction capacity of two objects with different set sizes.");
        for(size_t i = 0; i < A.nset(); ++i)
        {
            size_t mcap = _contraction_capacity(A(i), B(i));
            if(mcap > capacity)
            {
                capacity = mcap;
            }
        }
        return capacity;

    }
};

}

//TODO: Implement matrix element evaluation for sop_operator.  This will require a modification of the underlying datastructures for storing elements slightly
template <typename T, typename backend=linalg::blas_backend>
struct matrix_element_buffer
{
    using mat_type = linalg::matrix<T, backend>;
    using triad_type = std::vector<mat_type>;

    matrix_element_buffer(){}
    matrix_element_buffer(const matrix_element_buffer& o) = default;
    matrix_element_buffer(matrix_element_buffer&& o) = default;

    matrix_element_buffer& operator=(const matrix_element_buffer& o) = default;
    matrix_element_buffer& operator=(matrix_element_buffer&& o) = default;

    mutable triad_type HA;
    mutable triad_type temp;
    mutable triad_type temp2;
    mutable size_t cap = 0;
    mutable size_t buf = 0;

    void reallocate(size_t maxcapacity, size_t nbuffers)
    {
        cap = maxcapacity;
        buf = nbuffers;
        CALL_AND_HANDLE(HA.resize(nbuffers), "Failed to resize the opA array.");
        CALL_AND_HANDLE(temp.resize(nbuffers), "Failed to resize the temporary matrix.");
        CALL_AND_HANDLE(temp2.resize(nbuffers), "Failed to resize the temporary matrix.");
        for(size_t i=0; i<HA.size(); ++i)
        {
            CALL_AND_HANDLE(HA[i].reallocate(maxcapacity), "Failed to resize the opA array.");
            CALL_AND_HANDLE(temp[i].reallocate(maxcapacity), "Failed to resize the temporary matrix.");
            CALL_AND_HANDLE(temp2[i].reallocate(maxcapacity), "Failed to reszie temporary matrix.");
        }
    }

    void resize(size_t s1, size_t s2)
    {
        for(size_t i=0; i<HA.size(); ++i)
        {
            CALL_AND_HANDLE(HA[i].resize(s1, s2), "Failed to resize the opA array.");
            CALL_AND_HANDLE(temp[i].resize(s1, s2), "Failed to resize the temporary matrix.");
            CALL_AND_HANDLE(temp2[i].resize(s1, s2), "Failed to reszie temporary matrix.");
        }
    }

    void clear()
    {
        for(size_t i=0; i<HA.size(); ++i)
        {
            CALL_AND_HANDLE(HA[i].clear(), "Failed to clear a temporary working array tree.");
            CALL_AND_HANDLE(temp[i].clear(), "Failed to clear a temporary working array tree.");
            CALL_AND_HANDLE(temp2[i].clear(), "Failed to clear a temporary working array tree.");
        }

        CALL_AND_HANDLE(temp2.clear(), "Failed to clear a temporary working array tree.");
        CALL_AND_HANDLE(HA.clear(), "Failed to clear a temporary working array tree.");
        CALL_AND_HANDLE(temp.clear(), "Failed to clear a temporary working array tree.");
    }

    template <typename state_type>
    static size_t get_maximum_size(const state_type& A)
    {
        size_t maxsize = 0;
        for(const auto& a : A)
        {
            size_t size = a.maxsize();            if(size > maxsize){maxsize = size;}
        }
        return maxsize;
    }

    template <typename state_type>
    static size_t get_maximum_capacity(const state_type& A)
    {
        size_t maxcapacity = 0;
        for(const auto& a : A)
        {
            size_t capacity = a.maxcapacity();    if(capacity > maxcapacity){maxcapacity = capacity;}
        }
        return maxcapacity;
    }

    template <typename state_type_A, typename state_type_B>
    static size_t get_maximum_capacity(const state_type_A& A, const state_type_B& B)
    {
        size_t maxcapacity = 0;
        for(size_t ind = 0; ind < A.size(); ++ind)
        {
            size_t capacity = helper::contraction_capacity::evaluate(A[ind], B[ind]);
            for(size_t i = 0; i < A[ind].nmodes(); ++i)
            {
                capacity *= std::max(A[ind].max_dim(i), B[ind].max_dim(i));
            }                                     
            if(capacity > maxcapacity){maxcapacity = capacity;}
        }                                         
        return maxcapacity;                       
    }

    template <typename node_type_A, typename node_type_B>
    static size_t get_size(const node_type_A& a, const node_type_B& b)
    {
        return a.maxsize() > b.maxsize() ? a.maxsize() : b.maxsize();
    }

    template <typename node_type_A, typename node_type_B>
    static size_t get_capacity(const node_type_A& a, const node_type_B& b, bool use_capacity = false)
    {
        size_t capacity = helper::contraction_capacity::evaluate(a, b);
        for(size_t i = 0; i < a.nmodes(); ++i)
        {
            capacity *= std::max(a.max_dim(i), b.max_dim(i));
        }                                     
        size_t size = get_size(a, b);
        if(use_capacity)
        {
            return capacity > size ? capacity : size;
        }
        return size;
    }
};

}   //namespace ttns

#endif  //TTNS_MATRIX_ELEMENT_HPP//

