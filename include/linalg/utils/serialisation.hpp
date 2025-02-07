#ifndef LINALG_UTILS_SERIALISATION_HPP
#define LINALG_UTILS_SERIALISATION_HPP

#ifdef CEREAL_LIBRARY_FOUND

#include <common/exception_handling.hpp>
#include "../linalg_forward_decl.hpp"

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/complex.hpp>
#include <cereal/details/helpers.hpp>

#ifdef PYTTN_BUILD_CUDA

namespace cereal
{
template <typename archive> void save(archive& ar, const complex<float>& val){ar(cereal::make_nvp("real", val.real()), cereal::make_nvp("imag", val.imag()));}
template <typename archive> void load(archive& ar, complex<float>& val){float v;  ar(cereal::make_nvp("real", v));  val.real(v);    ar(cereal::make_nvp("imag", v)); val.imag(v);}

template <typename archive> void save(archive& ar, const complex<double>& val){ar(cereal::make_nvp("real", val.real()), cereal::make_nvp("imag", val.imag()));}
template <typename archive> void load(archive& ar, complex<double>& val){double v;  ar(cereal::make_nvp("real", v));  val.real(v);    ar(cereal::make_nvp("imag", v)); val.imag(v);}
}   //namespace cereal

#endif

namespace linalg
{

namespace internal
{
template <typename T, typename backend> struct buffer_reader_wrapper;
template <typename T, typename backend> struct buffer_writer_wrapper;

template <typename T>
struct buffer_writer_wrapper<T, blas_backend>
{
    using size_type = blas_backend::size_type;
    T* buf;
    size_type cap;

    ~buffer_writer_wrapper(){buf = nullptr;}
    
    template <typename Archive>
    void save(Archive& archive) const
    {
        archive(cereal::make_size_tag(cap));
        for(size_type i=0; i<cap; ++i){archive(buf[i]);}
    }

    template <typename Archive>
    void load(Archive& /* archive */) {RAISE_EXCEPTION("IF THIS COMES UP SOMETHING HAS GONE HORRIBLY WRONG.");}
};

template <typename T>
struct buffer_reader_wrapper<T, blas_backend>
{
    using size_type = blas_backend::size_type;
    T** buf;
    size_type* cap;

    ~buffer_reader_wrapper(){buf = nullptr; cap = nullptr;}

    template <typename U> buffer_reader_wrapper& operator=(const U& other) = delete;
    template <typename U> buffer_reader_wrapper& operator=(U&& other) = delete;

    template <typename Archive>
    void save(Archive& /* archive */) const{RAISE_EXCEPTION("IF THIS COMES UP SOMETHING HAS GONE HORRIBLY WRONG.");}

    template <typename Archive>
    void load(Archive& archive)
    {
        using allocator = memory::allocator<T, blas_backend>;
        size_type s;
        archive(cereal::make_size_tag(s));
        if(*buf == nullptr){CALL_AND_HANDLE(*buf = allocator::allocate(s), "Failed to deserialize cpu buffer.  Error when allocating new buffer to store result in.");}
        else
        {
            if(s != *cap)
            {
                CALL_AND_HANDLE(allocator::deallocate(*buf), "Failed to deserialize cpu buffer.  Error when deallocating previously allocated buffer to overwrite.");
                CALL_AND_HANDLE(*buf = allocator::allocate(s), "Failed to deserialize cpu buffer.  Error when allocating new buffer to store result in.");
            }
        }
        for(size_t i=0; i<s; ++i){archive((*buf)[i]);}   *cap = s;   
    }
};

#ifdef PYTTN_BUILD_CUDA
template <typename T>
struct buffer_writer_wrapper<T, cuda_backend>
{
    using size_type = cuda_backend::size_type;
    T* buf;
    size_type cap;

    ~buffer_writer_wrapper(){buf = nullptr;}

    template <typename Archive>
    void save(Archive& archive) const
    {
        using cpu_allocator = memory::allocator<T, blas_backend>;
        using memtransfer = memory::transfer<cuda_backend, blas_backend>;

        T* cpu_buf = nullptr;
        CALL_AND_HANDLE(cpu_buf = cpu_allocator::allocate(cap), "Failed to serialize cuda buffer.  Error when allocating temporary cpu buffer object.");
        CALL_AND_HANDLE(memtransfer::copy(buf, cap, cpu_buf), "Failed to serialize cuda buffer.  Failed to copy temporary gpu buffer to the temporary cpu buffer.");

        //now do the serialization 
        archive(cereal::make_size_tag(cap));
        for(size_type i=0; i<cap; ++i){archive(cpu_buf[i]);}

        //and clean up the temporary cpu buffer
        CALL_AND_HANDLE(cpu_allocator::deallocate(cpu_buf), "Failed to serialize cuda buffer.  Failed to clean up temporary cpu buffer object.");        cpu_buf = nullptr;
    }

    template <typename Archive>
    void load(Archive& /* archive */) {RAISE_EXCEPTION("IF THIS COMES UP SOMETHING HAS GONE HORRIBLY WRONG.");}
};

template <typename T>
struct buffer_reader_wrapper<T, cuda_backend>
{
    using size_type = cuda_backend::size_type;
    T** buf;
    size_type* cap;

    ~buffer_reader_wrapper(){buf = nullptr; cap = nullptr;}

    template <typename U> buffer_reader_wrapper& operator=(const U& other) = delete;
    template <typename U> buffer_reader_wrapper& operator=(U&& other) = delete;

    template <typename Archive>
    void save(Archive& /* archive */) const{RAISE_EXCEPTION("IF THIS COMES UP SOMETHING HAS GONE HORRIBLY WRONG.");}

    template <typename Archive>
    void load(Archive& archive)
    {
        using cpu_allocator = memory::allocator<T, blas_backend>;
        using gpu_allocator = memory::allocator<T, cuda_backend>;
        using memtransfer = memory::transfer<blas_backend, cuda_backend>;

        size_type s;
        archive(cereal::make_size_tag(s));
        
        //resize the gpu buffer given the new size_type 
        if(*buf == nullptr){CALL_AND_HANDLE(*buf = gpu_allocator::allocate(s), "Failed to deserialize buffer.  Error when allocating new buffer to store result in.");}
        else
        {
            if(s != *cap)
            {
                CALL_AND_HANDLE(gpu_allocator::deallocate(*buf), "Failed to deserialize buffer.  Error when deallocating previously allocated buffer to overwrite.");
                CALL_AND_HANDLE(*buf = gpu_allocator::allocate(s), "Failed to deserialize buffer.  Error when allocating new buffer to store result in.");
            }
        }
        *cap = s;
    
        //now allocate the cpu buffer and read in the result
        T* cpu_buf = nullptr;
        CALL_AND_HANDLE(cpu_buf = cpu_allocator::allocate(s), "Failed to deserialize cuda buffer.  Failed to allocate temporary cpu buffer object.");
        for(size_t i=0; i<s; ++i){archive(cpu_buf[i]);}

        //now transfer the cpu_buf values to the gpu_buf
        CALL_AND_HANDLE(memtransfer::copy(cpu_buf, s, *buf), "Failed to deserialize cuda buffer.  Failed to copy temporary cpu buffer to gpu.");

        //and clean up the temporary cpu buffer
        CALL_AND_HANDLE(cpu_allocator::deallocate(cpu_buf), "Failed to deserialize cuda buffer.  Failed to clean up temporary cpu buffer object.");        cpu_buf = nullptr;
    }
};
#endif

}   //namespace internal
}   //namespace linalg


#endif  //CEREAL_LIBRARY_FOUND

#endif  //LINALG_UTILS_SERIALISATION_HPP//

