#ifndef LINALG_TENSOR_DETAILS_HPP
#define LINALG_TENSOR_DETAILS_HPP

#include "../../linalg_forward_decl.hpp"

#ifdef __NVCC__
#include "../../backends/cuda_kernels.hpp"
#endif

namespace linalg
{

/**
 *  @cond INTERNAL
 *  Forward declaration of the Tensor class
 */ 

//////////////////////////////////////////////////////////////////////////////////////////////////
//                            GENERIC DETAILS OBJECTS FOR THE TENSORS                           //
//////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ArrType, typename backend>
class tensor_details<ArrType, 1, false, backend >
{
public:
    using array_type = typename traits<ArrType>::base_type;
    using size_type = typename backend::size_type;
    using value_type = typename traits<ArrType>::value_type;

    inline size_type length() const {return static_cast<const array_type*>(this)->shape(0);}
    inline size_type incx() const {return 1;}
};

//The general matrix type.
template <typename ArrType, typename backend>
class tensor_details<ArrType, 2, false, backend >
{
public:
    using array_type = typename traits<ArrType>::base_type;
    using value_type = typename traits<ArrType>::value_type;
    using size_type = typename backend::size_type;
    inline size_type nrows() const {return static_cast<const array_type*>(this)->shape(0);}
    inline size_type ncols() const {return static_cast<const array_type*>(this)->shape(1);}

    size_type incx() const {return 1;}
    size_type diagonal_stride() const {return static_cast<const array_type*>(this)->shape(1)+1;}
};

template <typename ArrType, typename backend>
class tensor_details<ArrType, 3, false, backend >
{
public:
    using array_type = typename traits<ArrType>::base_type;
    using value_type = typename traits<ArrType>::value_type;
    using size_type = typename backend::size_type;
    inline size_type nslices() const{return static_cast<const array_type*>(this)->shape(0);}
    inline size_type nrows() const {return static_cast<const array_type*>(this)->shape(1);}
    inline size_type ncols() const {return static_cast<const array_type*>(this)->shape(2);}

    size_type incx() const {return 1;}
};



//////////////////////////////////////////////////////////////////////////////////////////////////
//                          DETAILS OBJECTS FOR THE BLAS BACKEND TENSORS                        //
//////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ArrType>
class tensor_details<ArrType, 1, true, blas_backend >
{
public:
    using array_type = typename traits<ArrType>::base_type;
    using size_type = typename blas_backend::size_type;
    using value_type = typename traits<ArrType>::value_type;

    inline size_type length() const {return static_cast<const array_type*>(this)->shape(0);}
    inline size_type incx() const {return 1;}

    template <typename Func, typename ... Args>
    void fill(Func&& f, Args&&... args)
    {
        array_type& a = static_cast<array_type&>(*this);
        size_type m = a.shape(0);
        auto abuffer = a.buffer();
        for(size_type i=0; i<m; ++i){CALL_AND_HANDLE(abuffer[i] = std::forward<Func>(f)(i, std::forward<Args>(args)...), "Failed to fill 1d tensor array from functor.");}
    }
};

//The general matrix type.
template <typename ArrType>
class tensor_details<ArrType, 2, true, blas_backend >
{
public:
    using array_type = typename traits<ArrType>::base_type;
    using value_type = typename traits<ArrType>::value_type;
    using size_type = typename blas_backend::size_type;
    inline size_type nrows() const {return static_cast<const array_type*>(this)->shape(0);}
    inline size_type ncols() const {return static_cast<const array_type*>(this)->shape(1);}
    size_type incx() const {return 1;}
    size_type diagonal_stride() const {return static_cast<const array_type*>(this)->shape(1)+1;}
    
    template <typename Func, typename ... Args>
    void fill(Func&& f, Args&&... args)
    {
        array_type& a = static_cast<array_type&>(*this);
        
        size_type m = a.shape(0);
        size_type n = a.shape(1);
        auto abuffer = a.buffer();
        for(size_type i=0; i<m; ++i)
        {
            size_type ind = i*n;
            for(size_type j=0; j<n; ++j)
            {
                CALL_AND_HANDLE(abuffer[ind+j] = std::forward<Func>(f)(i,j, std::forward<Args>(args)...), "Failed to fill 2d tensor array from functor.");
            }
        }
    }
};

template <typename ArrType>
class tensor_details<ArrType, 3, true, blas_backend >
{
public:
    using array_type = typename traits<ArrType>::base_type;
    using value_type = typename traits<ArrType>::value_type;
    using size_type = typename blas_backend::size_type;
    inline size_type nslices() const{return static_cast<const array_type*>(this)->shape(0);}
    inline size_type nrows() const {return static_cast<const array_type*>(this)->shape(1);}
    inline size_type ncols() const {return static_cast<const array_type*>(this)->shape(2);}
    size_type incx() const {return 1;}

    template <typename Func, typename ... Args>
    void fill(Func&& f, Args&&... args)
    {
        array_type& a = static_cast<array_type&>(*this);
        size_type m = a.shape(0);
        size_type n = a.shape(1);
        size_type o = a.shape(2);
        auto abuffer = a.buffer();
        for(size_type i=0; i<m; ++i)
        {
            size_type ind = i*n;
            for(size_type j=0; j<n; ++j)
            {
                size_type ind2 = (ind+j)*o;
                for(size_type k=0; k<o; ++k)
                {
                    CALL_AND_HANDLE(abuffer[ind2+k] = std::forward<Func>(f)(i,j,k, std::forward<Args>(args)...),  "Failed to fill 3d tensor array from functor.");
                }
            }
        }
    }
};

#ifdef __NVCC__
//////////////////////////////////////////////////////////////////////////////////////////////////
//                          DETAILS OBJECTS FOR THE CUDA BACKEND TENSORS                        //
//////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ArrType>
class tensor_details<ArrType, 1, true, cuda_backend >
{
public:
    using array_type = typename traits<ArrType>::base_type;
    using value_type = typename traits<ArrType>::value_type;
    using size_type = typename cuda_backend::size_type;

    inline size_type length() const {return static_cast<const array_type*>(this)->shape(0);}
    inline size_type incx() const {return 1;}

    template <typename Func, typename ... Args>
    void fill(Func&& f, Args&&... args)
    {
        array_type& a = static_cast<array_type&>(*this);
        size_type m = a.shape(0);
        cuda_backend::func_fill_1(a.buffer(), m, std::forward<Func>(f), std::forward<Args>(args)...);
    }
};


//The general matrix type.
template <typename ArrType>
class tensor_details<ArrType, 2, true, cuda_backend >
{
public:
    using array_type = typename traits<ArrType>::base_type;
    using value_type = typename traits<ArrType>::value_type;
    using size_type = typename cuda_backend::size_type;

    inline size_type nrows() const {return static_cast<const array_type*>(this)->shape(0);}
    inline size_type ncols() const {return static_cast<const array_type*>(this)->shape(1);}
    size_type incx() const {return 1;}
    size_type diagonal_stride() const {return static_cast<const array_type*>(this)->shape(1)+1;}
    

    template <typename Func, typename ... Args>
    void fill(Func&& f, Args&&... args)
    {
        array_type& a = static_cast<array_type&>(*this);
        size_type m = a.shape(0);
        size_type n = a.shape(1);
        cuda_backend::func_fill_2(a.buffer(), m, n, std::forward<Func>(f), std::forward<Args>(args)...);
    }
};

template <typename ArrType>
class tensor_details<ArrType, 3, true, cuda_backend >
{
public:
    using array_type = typename traits<ArrType>::base_type;
    using value_type = typename traits<ArrType>::value_type;
    using size_type = typename cuda_backend::size_type;
    inline size_type nslices() const{return static_cast<const array_type*>(this)->shape(0);}
    inline size_type nrows() const {return static_cast<const array_type*>(this)->shape(1);}
    inline size_type ncols() const {return static_cast<const array_type*>(this)->shape(2);}
    size_type incx() const {return 1;}


    template <typename Func, typename ... Args>
    void fill(Func&& f, Args&&... args)
    {
        array_type& a = static_cast<array_type&>(*this);
        size_type m = a.shape(0);
        size_type n = a.shape(1);
        size_type o = a.shape(2);
        cuda_backend::func_fill_3(a.buffer(), m, n, o, std::forward<Func>(f), std::forward<Args>(args)...);
    }
};
#endif //__NVCC__

///@endcond
} //namespace linalg //

#endif  //LINALG_TENSOR_DETAILS_HPP//

