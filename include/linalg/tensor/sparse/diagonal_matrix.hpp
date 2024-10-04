#ifndef LINALG_DIAGONAL_MATRIX_BASE_CRTP_HPP
#define LINALG_DIAGONAL_MATRIX_BASE_CRTP_HPP

#include "special_matrix_base.hpp"

namespace linalg
{
template <typename impl>
class diagonal_matrix_base : public special_matrix_base<diagonal_matrix_base<impl>>
{
public:
    static constexpr size_t rank = 2;
    using self_type = diagonal_matrix_base<impl>;
    using base_type = special_matrix_base<self_type>;
    using size_type = typename base_type::size_type;
    using shape_type = typename base_type::shape_type;
    using value_type = typename base_type::value_type;    using pointer = typename base_type::pointer;    using const_pointer = typename base_type::const_pointer;
    friend base_type;
protected:
    using base_type::m_vals;
    using base_type::m_shape;

public:
    template <typename ... Args>
    diagonal_matrix_base(Args&& ... args) try : base_type(std::forward<Args>(args)...){}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct diagonal matrix object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));   return *this;}

    constexpr size_type incx() const{return 1;}
    constexpr size_type diagonal_stride() const {return 1;}
protected:
    static size_type nnz_from_shape(const shape_type& shape){return shape[0] < shape[1] ? shape[0] : shape[1];}
    static size_type nrows_from_nnz(const size_type& nnz){return nnz;}
public:
    inline value_type* D(){return base_type::m_vals;}
    inline const value_type * D()const{return base_type::m_vals;}

};  //class diagonal_matrix_base


template <typename T> 
class diagonal_matrix<T, blas_backend> : public diagonal_matrix_base<diagonal_matrix<T, blas_backend> >
{
public:
    using self_type = diagonal_matrix<T, blas_backend>;
    using base_type = diagonal_matrix_base<self_type>;
    using size_type = typename blas_backend::size_type;

    template <typename U> friend std::ostream& operator<<(std::ostream& out, const diagonal_matrix<U, blas_backend>& mat);
    template <typename U> friend std::istream& operator>>(std::istream& out, diagonal_matrix<U, blas_backend>& mat);
public:
    template <typename ... Args> diagonal_matrix(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct diagonal_matrix object.");}
    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));   return *this;}

    T& operator[](size_type i){return base_type::m_vals[i];}
    const T& operator[](size_type i) const{return base_type::m_vals[i];}
    T& at(size_type i){ASSERT(internal::compare_bounds(i, base_type::m_nnz), "Failed to access element of diagonal matrix.  Index out of bounds.");  return base_type::m_vals[i];}
    const T& at(size_type i) const{ASSERT(internal::compare_bounds(i, base_type::m_nnz), "Failed to access element of diagonal matrix.  Index out of bounds.");  return base_type::m_vals[i];}

    T& operator()(size_type i, size_type /* j */){return base_type::m_vals[i];}
    const T& operator()(size_type i, size_type /* j */) const{return base_type::m_vals[i];}

    T& at(size_type i, size_type j)
    {
        ASSERT(internal::compare_bounds(i, base_type::m_shape[0]) && internal::compare_bounds(j, base_type::m_shape[1]), "Failed to access element of diagonal matrix.  Index out of bounds.");
        ASSERT(i == j, "Failed to access element of diagonal matrix.  Requested element is not on the diagonal.");
        return base_type::m_vals[i];
    }

    const T& at(size_type i, size_type j) const
    {
        ASSERT(internal::compare_bounds(i, base_type::m_shape[0]) && internal::compare_bounds(j, base_type::m_shape[1]), "Failed to access element of diagonal matrix.  Index out of bounds.");
        ASSERT(i == j, "Failed to access element of diagonal matrix.  Requested element is not on the diagonal.");
        return base_type::m_vals[i];
    }

    matrix<T, blas_backend> todense() const
    {
        matrix<T, blas_backend> mat(base_type::m_shape[0], base_type::m_shape[1]);
        for(size_t i = 0; i < base_type::m_nnz; ++i)
        {
            mat(i, i) = base_type::m_vals[i];
        }
        return mat;
    }
};  //diagonal_matrix<T, blas_backend>


#ifdef __NVCC__
template <typename T> 
class diagonal_matrix<T, cuda_backend> : public diagonal_matrix_base<diagonal_matrix<T, cuda_backend> >
{
public:
    using self_type = diagonal_matrix<T, cuda_backend>;
    using base_type = diagonal_matrix_base<self_type>;
    using pointer = typename base_type::pointer;    using const_pointer = typename base_type::const_pointer;
public:
    template <typename ... Args> diagonal_matrix(Args&& ... args) try : base_type(std::forward<Args>(args)...) {}
    catch(const std::exception& ex){std::cerr << ex.what() << std::endl;    RAISE_EXCEPTION("Failed to construct diagonal_matrix object.");}

    template <typename ... Args> self_type& operator=(Args&& ... args){CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));   return *this;}

    __host__ __device__ pointer buffer(){return base_type::m_vals;}
    __host__ __device__ const_pointer buffer()const{return base_type::m_vals;}
    __host__ __device__ pointer data(){return base_type::m_vals;}
    __host__ __device__ const_pointer data()const{return base_type::m_vals;}

    matrix<T> todense() const
    {
        diagonal_matrix<T> mat(*this);
        return mat.todense();
    }
};  //diagonal_matrix<T, cuda_backend>
#endif


template <typename T>
std::ostream& operator<<(std::ostream& out, const diagonal_matrix<T, blas_backend>& mat)
{
    using size_type = typename diagonal_matrix<T, blas_backend>::size_type;
    out << "diagonal: " << mat.m_shape[0] << " " << mat.m_shape[1] << std::endl;
    for(size_type i=0; i<mat.nnz(); ++i){out << i << " " << i << " " << mat.m_vals[i] << std::endl;}
    return out;
}

}   //namespace linalg

#endif  //LINALG_TENSOR_BASE_CRTP_HPP//


