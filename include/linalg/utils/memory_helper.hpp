#ifndef LINALG_MEMORY_HELPER_HPP
#define LINALG_MEMORY_HELPER_HPP

#include <common/exception_handling.hpp>

namespace linalg
{
namespace memory
{

template <typename T, typename backend> class allocator;
template <typename src_bck, typename dest_bck> class transfer;
template <typename T, typename backend> class filler;

template <typename T>
class allocator<T, blas_backend>
{
    using size_type = blas_backend::size_type;
public:
    static inline T* allocate(size_type n)
    {
        T* res;   CALL_AND_RETHROW(res = new T[n]);   return res;
    }
    static inline void deallocate(T*& v)
    {
        if(v != nullptr){delete[] v;} v=nullptr;
    }
};

namespace internal
{

template <typename T>
static inline void copy_buffers(const T* const src, blas_backend::size_type n, T* dest)
{
    if(dest == src){return;}

    std::ptrdiff_t overlap = std::abs(src-dest) < static_cast<std::ptrdiff_t>(n) ? src-dest : 0;

#ifdef LINALG_ALLOW_BUFFER_OVERLAP
    using size_type = blas_backend::size_type;
    //if the two buffers don't overlap then we can just use copy_n
    if(overlap == 0){CALL_AND_HANDLE(std::copy_n(src, n, dest), "Failed to copy blas buffers.   Error when calling copy_n.");}
    //if src < dest then copying we need to copy src to dest backwards so that we don't overwrite the elements of src that overlap with the elements of dest
    else if(overlap < 0){for(size_type i=0; i<n; ++i){dest[n-(i+1)] = src[n-(i+1)];}}
    //if src > dest then copying we need to copy src to dest forwards so that we don't overwrite the elements of src that overlap with the elements of dest
    else{for(size_type i=0; i<n; ++i){dest[i] = src[i];}}
#else
    ASSERT(overlap == 0, "Failed to copy buffers. The underlying buffers overlap which has not been allowed.");
    if(overlap == 0){CALL_AND_HANDLE(std::copy_n(src, n, dest), "Failed to copy blas buffers.   Error when calling copy_n.");}
#endif
} 
}   //namespace internal


//blas to blas memory transfer
template <> 
class transfer<blas_backend, blas_backend> 
{
    using size_type = blas_backend::size_type;
public:
    template <typename T> 
    static inline void copy(const T* const src, size_type n, T* dest){CALL_AND_HANDLE(internal::copy_buffers(src, n, dest), "Failed to copy buffers.");}

    template <typename T, size_t D> 
    static inline void copy_noncontiguous(const T* const src, const std::array<size_type, D>& size, const std::array<size_type, D>& s_strides, T* dest, const std::array<size_type, D>& d_strides)
    {
        size_type dim = 1;
        for(size_type d=0; d < D; ++d)
        {
            dim*=size[d];
        }
        for(size_type i = 0; i < dim; ++i)
        {
            size_type d2 = 0;
            size_type itemp = i;
            for(size_type d=0; d < D; ++d)
            {
                size_type iv = (itemp/d_strides[i]);
                d2 += iv*s_strides[i];
                itemp -= iv*d_strides[i];
            }
            dest[i] = src[d2];
        }
    }
};

template <typename T> 
class filler<T, blas_backend>
{
    using size_type = blas_backend::size_type;
public:
    static inline void fill(T* dest, size_type n, const T& val){CALL_AND_HANDLE(std::fill_n(dest, n, val), "Failed to fill buffer.  Call to std::fill_n failed.");}
    
};  //class filler<blas_backend>


#ifdef PYTTN_BUILD_CUDA
template <typename T>
class allocator<T, cuda_backend>
{
    using size_type = cuda_backend::size_type;
public:
    static inline T* allocate(size_type n)
    {
        T* res;
        CALL_AND_HANDLE(cuda_safe_call(cudaMalloc(&res, n*sizeof(T))), "Failed to allocate memory buffer.");
        return res;
    }

    static inline void deallocate(T*& v)
    {   
        if(v != nullptr){CALL_AND_HANDLE(cuda_safe_call(cudaFree(v)), "Failed to deallocate memory buffer.");}
        v = nullptr;
    }
};

namespace internal
{
class cuda_transfer
{
public:
    using size_type = cuda_backend::size_type;
    template <typename T, typename size_type>
    static inline void copy(const T* const src, size_type n, T* const dest, cudaMemcpyKind type)
    {
        CALL_AND_HANDLE(cuda_safe_call(cudaMemcpy(dest, src, n*sizeof(T), type)), "cudaMemcpy call failed.");
    }

    template <typename T, typename size_type>
    static inline void copy(const T* const src, size_type n, T* const dest, cudaMemcpyKind type, cudaStream_t stream)
    {
        CALL_AND_HANDLE(cuda_safe_call(cudaMemcpy(dest, src, n*sizeof(T), type, stream)), "cudaMemcpy call failed.");
    }

    template <typename T, size_t D> 
    static inline void copy_noncontiguous(const T* const src, const std::array<size_type, D>& size, const std::array<size_type, D>& s_strides, T* dest, const std::array<size_type, D>& d_strides, cudaMemcpyKind type)
    {
        RAISE_EXCEPTION("Noncontiguous copy not supported by cuda transfer.");
    }

    template <typename T, size_t D> 
    static inline void copy_noncontiguous(const T* const src, const std::array<size_type, D>& size, const std::array<size_type, D>& s_strides, T* dest, const std::array<size_type, D>& d_strides, cudaMemcpyKind type, cudaStream_t stream)
    {
        RAISE_EXCEPTION("Noncontiguous copy not supported by cuda transfer.");
    }
};

}   //namespace internal


//blas to cuda memory transfer
template <> 
class transfer<blas_backend, cuda_backend> 
{
    using size_type = cuda_backend::size_type;
public:
    template <typename T> static inline void copy(const T* const src, size_type n, T* const dest){CALL_AND_HANDLE(internal::cuda_transfer::copy(src, n, dest, cudaMemcpyHostToDevice), "Failed to perform host to device memory transfer.");}
    template <typename T> static inline void copy(const T* const src, size_type n, T* const dest, cudaStream_t stream){CALL_AND_HANDLE(internal::cuda_transfer::copy(src, n, dest, cudaMemcpyHostToDevice, stream), "Failed to perform asynchronous host to device memory transfer.");}

    template <typename T, size_t D>
    static inline void copy_noncontiguous(const T* const src, const std::array<size_t, D>& strides, size_type n, T* const dest)
    {
        CALL_AND_RETHROW(internal::cuda_transfer::copy_noncontiguous(src, strides, n, dest, cudaMemcpyHostToDevice));
    }

    template <typename T, size_t D> 
    static inline void copy_noncontiguous(const T* const src, const std::array<size_type, D>& size, const std::array<size_type, D>& s_strides, T* dest, const std::array<size_type, D>& d_strides)
    {
        CALL_AND_RETHROW(internal::cuda_transfer::copy_noncontiguous(src, size, s_strides, dest, d_strides, cudaMemcpyHostToDevice));
    }

    template <typename T, size_t D> 
    static inline void copy_noncontiguous(const T* const src, const std::array<size_type, D>& size, const std::array<size_type, D>& s_strides, T* dest, const std::array<size_type, D>& d_strides, cudaStream_t stream)
    {
        CALL_AND_RETHROW(internal::cuda_transfer::copy_noncontiguous(src, size, s_strides, dest, d_strides, cudaMemcpyHostToDevice, stream));
    }
};

//cuda to blas memory transfer
template <> 
class transfer<cuda_backend, blas_backend> 
{
    using size_type = cuda_backend::size_type;
public:
    template <typename T> static inline void copy(const T* const src, size_type n, T* const dest){CALL_AND_HANDLE(internal::cuda_transfer::copy(src, n, dest, cudaMemcpyDeviceToHost), "Failed to perform device to host memory transfer.");}
    template <typename T> static inline void copy(const T* const src, size_type n, T* const dest, cudaStream_t stream){CALL_AND_HANDLE(internal::cuda_transfer::copy(src, n, dest, cudaMemcpyDeviceToHost, stream), "Failed to perform asynchronous device to host memory transfer.");}
};

//cuda to cuda memory transfer
template <> 
class transfer<cuda_backend, cuda_backend> 
{
    using size_type = cuda_backend::size_type;
public:
    template <typename T> static inline void copy(const T* const src, size_type n, T* const dest){CALL_AND_HANDLE(internal::cuda_transfer::copy(src, n, dest, cudaMemcpyDeviceToDevice), "Failed to perform device to device memory transfer.");}
    template <typename T> static inline void copy(const T* const src, size_type n, T* const dest, cudaStream_t stream){CALL_AND_HANDLE(internal::cuda_transfer::copy(src, n, dest, cudaMemcpyDeviceToDevice, stream), "Failed to perform asynchronous device to device memory transfer.");}

    template <typename T, size_t D> 
    static inline void copy_noncontiguous(const T* const src, const std::array<size_type, D>& size, const std::array<size_type, D>& s_strides, T* dest, const std::array<size_type, D>& d_strides)
    {
        CALL_AND_RETHROW(internal::cuda_transfer::copy_noncontiguous(src, size, s_strides, dest, d_strides, cudaMemcpyDeviceToDevice));
    }

    template <typename T, size_t D> 
    static inline void copy_noncontiguous(const T* const src, const std::array<size_type, D>& size, const std::array<size_type, D>& s_strides, T* dest, const std::array<size_type, D>& d_strides, cudaStream_t stream)
    {
        CALL_AND_RETHROW(internal::cuda_transfer::copy_noncontiguous(src, size, s_strides, dest, d_strides, cudaMemcpyDeviceToDevice, stream));
    }
};

template <typename T> 
class filler<T, cuda_backend>
{
    using size_type = cuda_backend::size_type;
public:
    static inline void fill(T* dest, size_type n, const T& val){cuda_backend::fill_n(dest, n, val);}
};  //class filler<cuda_backend>

#endif

}   //namespace memory
}   //namespace linalg

#endif  //LINALG_MEMORY_HELPER_HPP//

