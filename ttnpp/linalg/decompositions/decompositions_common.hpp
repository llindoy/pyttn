#ifndef LINALG_DECOMPOSITIONS_COMMON_HPP
#define LINALG_DECOMPOSITIONS_COMMON_HPP


#include <common/exception_handling.hpp>
#include "../linalg_forward_decl.hpp"
#include "../backends/lapack_wrapper.hpp"

namespace linalg
{
namespace internal
{
template <typename array_type, typename value_type, typename backend>
struct valid_decomposition_vector_type 
    : public std::conditional
    <
        is_dense_tensor<array_type>::value,
        typename std::conditional
        <
            std::is_same<typename traits<array_type>::value_type, value_type>::value &&
            std::is_same<typename traits<array_type>::backend_type, backend>::value &&
            traits<array_type>::rank == 1 &&
            traits<array_type>::is_mutable,
            std::true_type,
            std::false_type
        >::type,
        typename std::conditional
        <
            is_diagonal_matrix_type<array_type>::value && 
            traits<array_type>::is_mutable,
            std::true_type,
            std::false_type
        >::type
    >::type
{};

template <typename array_type, typename value_type, typename backend>
struct valid_decomposition_matrix 
    : public std::conditional
    <
        is_dense_tensor<array_type>::value,
        typename std::conditional
        <
            std::is_same<typename traits<array_type>::value_type, value_type>::value &&
            std::is_same<typename traits<array_type>::backend_type, backend>::value &&
            traits<array_type>::rank == 2 &&
            traits<array_type>::is_mutable,
            std::true_type,
            std::false_type
        >::type,
        std::false_type
    >::type
{};

static inline blas_backend::size_type worksize_as_integer(float a)
{
    ASSERT(a > 0.0f, "Failed to convert worksize to integer type.  The worksize is negative");
    return static_cast<blas_backend::size_type>(a+0.5f);
}
static inline blas_backend::size_type worksize_as_integer(double a)
{
    ASSERT(a > 0.0, "Failed to convert worksize to integer type.  The worksize is negative");
    return static_cast<blas_backend::size_type>(a+0.5);
}
static inline blas_backend::size_type worksize_as_integer(complex<float> a)
{
    ASSERT(a.real() > 0.0f, "Failed to convert worksize to integer type.  The worksize is negative");
    return static_cast<blas_backend::size_type>(a.real()+0.5f);
}
static inline blas_backend::size_type worksize_as_integer(complex<double> a)
{
    ASSERT(a.real() > 0.0, "Failed to convert worksize to integer type.  The worksize is negative");
    return static_cast<blas_backend::size_type>(a.real()+0.5);
}


template <typename T>
void interleave_eigenvalues(typename blas_backend::size_type N, const T* wr, typename blas_backend::size_type incwr, const T* wi, typename blas_backend::size_type incwi, complex<T>* w, typename blas_backend::size_type incw)
{
    static_assert(is_number<T>::value && !is_complex<T>::value, "Failed to instantiate interleave eigenvalues routine.");
    for(typename blas_backend::size_type i=0; i<N; ++i){w[i*incw] = complex<T>(wr[i*incwr], wi[i*incwi]);}
}

template <typename T>
void set_real_eigenvects(typename blas_backend::size_type N, const T* vecs, complex<T>* working)
{
    static_assert(is_number<T>::value && !is_complex<T>::value, "Failed to instantiate interleave eigenvalues routine.");
    for(typename blas_backend::size_type i=0; i<N; ++i){working[i] = complex<T>(vecs[i], 0.0);}
}

template <typename T>
void unpack_complex_eigenvector(typename blas_backend::size_type N, const T* re, const T* im, complex<T>* w1, complex<T>* w2)
{
    static_assert(is_number<T>::value && !is_complex<T>::value, "Failed to instantiate interleave eigenvalues routine.");
    for(typename blas_backend::size_type i=0; i<N; ++i){w1[i] = complex<T>(re[i], im[i]);  w2[i] = complex<T>(re[i], -im[i]);}
}

template <typename T>
void unpack_eigenvectors(typename blas_backend::size_type N, T* iw, const T* vecs, complex<T>* out)
{
    using size_type = typename blas_backend::size_type;
    static_assert(is_number<T>::value && !is_complex<T>::value, "Failed to instantiate interleave eigenvectors routine.");

    for(size_type i=0; i<N; ++i)
    {
        if(iw[i] == T(0.0)){set_real_eigenvects(N, vecs+i*N, out+i*N);}
        else{unpack_complex_eigenvector(N, vecs+i*N, vecs+(i+1)*N, out+i*N, out+(i+1)*N); ++i;}
    }
}


template <typename vals_type, typename T, typename B>
struct validate_vals_type{static constexpr bool value = valid_decomposition_vector_type<vals_type, T, B>::value;};

template <typename vals_type, typename T, typename B>
struct validate_real_vals_type{static constexpr bool value = valid_decomposition_vector_type<vals_type, typename get_real_type<T>::type, B>::value;};

template <typename vals_type, typename T, typename B>
struct validate_complex_vals_type{static constexpr bool value = (!is_complex<T>::value && valid_decomposition_vector_type<vals_type, complex<T>, B>::value);};

template <typename vecs_type, typename T, typename B>
struct validate_vecs_type{static constexpr bool value = valid_decomposition_matrix<vecs_type, T, B>::value;};

template <typename vecs_type, typename T, typename B>
struct validate_complex_vecs_type{static constexpr bool value = (!is_complex<T>::value && valid_decomposition_matrix<vecs_type, complex<T>, B>::value);};

template <typename vecs_typer, typename vecs_typel, typename T, typename B>
struct validate_vecs_rl_type{static constexpr bool value = valid_decomposition_matrix<vecs_typer, T, B>::value && valid_decomposition_matrix<vecs_typel, T, B>::value;};

template <typename vecs_typer, typename vecs_typel, typename T, typename B>
struct validate_complex_vecs_rl_type{static constexpr bool value = (!is_complex<T>::value && valid_decomposition_matrix<vecs_typer, complex<T>, B>::value && valid_decomposition_matrix<vecs_typel, complex<T>, B>::value);};

template <typename vals_type, typename vecs_type, typename T, typename B>
struct validate_vals_vecs_type{static constexpr bool value = valid_decomposition_vector_type<vals_type, T, B>::value && valid_decomposition_matrix<vecs_type, T, B>::value;};

template <typename vals_type, typename vecs_typer, typename vecs_typel, typename T, typename B>
struct validate_vals_vecs_rl_type{static constexpr bool value = valid_decomposition_vector_type<vals_type, T, B>::value && valid_decomposition_matrix<vecs_typer, T, B>::value && valid_decomposition_matrix<vecs_typel, T, B>::value;};


template <typename vals_type, typename vecs_type, typename T, typename B>
struct validate_complex_vals_vecs_type{static constexpr bool value = !is_complex<T>::value && valid_decomposition_vector_type<vals_type, complex<T>, B>::value && valid_decomposition_matrix<vecs_type, complex<T>, B>::value;};

template <typename vals_type, typename vecs_typer, typename vecs_typel, typename T, typename B>
struct validate_complex_vals_vecs_rl_type{static constexpr bool value = !is_complex<T>::value && valid_decomposition_vector_type<vals_type, complex<T>, B>::value && valid_decomposition_matrix<vecs_typer, complex<T>, B>::value && valid_decomposition_matrix<vecs_typel, complex<T>, B>::value;};

template <typename vals_type, typename vecs_type, typename T, typename B>
struct validate_real_vals_vecs_type{static constexpr bool value = valid_decomposition_vector_type<vals_type, typename get_real_type<T>::type, B>::value && valid_decomposition_matrix<vecs_type, T, B>::value;};

template <typename vals_type, typename vecs_typer, typename vecs_typel, typename T, typename B>
struct validate_real_vals_vecs_rl_type{static constexpr bool value = valid_decomposition_vector_type<vals_type, typename get_real_type<T>::type, B>::value && valid_decomposition_matrix<vecs_typer, T, B>::value && valid_decomposition_matrix<vecs_typel, T, B>::value;};


template <typename m1, typename m2> 
struct is_valid_decomp_matrix_type{static constexpr bool value = (same_topology_type<m1, m2>::value && compatible_traits<m1, m2>::value);};

template <typename m1, typename m2, typename return_type>
using valid_decomp_matrix_type = typename std::enable_if<is_valid_decomp_matrix_type<m1, m2>::value, return_type>::type;

template <typename m1, typename m2, typename m3, typename return_type>
using valid_decomp_matrix_type_2 = typename std::enable_if<is_valid_decomp_matrix_type<m1, m2>::value &&is_valid_decomp_matrix_type<m1, m3>::value , return_type>::type;

template <typename m1, typename m2, typename m3, typename m4, typename return_type>
using valid_decomp_matrix_type_3 = typename std::enable_if<is_valid_decomp_matrix_type<m1, m2>::value && is_valid_decomp_matrix_type<m1, m3>::value && is_valid_decomp_matrix_type<m1, m4>::value, return_type>::type;

template <typename m1, typename m2, typename return_type, template <typename ...> class condition, typename ... args>
using valid_mutable_decomp_func = typename std::enable_if<is_valid_decomp_matrix_type<m1, m2>::value && traits<m2>::is_mutable && condition<args..., remove_cvref_t<typename traits<m1>::value_type>, typename traits<m1>::backend_type>::value, return_type>::type;

template <typename m1, typename m2, typename return_type, template <typename ...> class condition, typename ... args>
using valid_decomp_func = typename std::enable_if<is_valid_decomp_matrix_type<m1, m2>::value && condition<args..., remove_cvref_t<typename traits<m1>::value_type>, typename traits<m1>::backend_type>::value, return_type>::type;

template <typename m1, typename m2, typename m3, typename return_type, template <typename ...> class condition, typename ... args>
using valid_mutable_decomp_func_2 = typename std::enable_if<is_valid_decomp_matrix_type<m1, m2>::value && is_valid_decomp_matrix_type<m1, m3>::value && traits<m2>::is_mutable && condition<args..., remove_cvref_t<typename traits<m1>::value_type>, typename traits<m1>::backend_type>::value, return_type>::type;

template <typename m1, typename m2, typename m3, typename return_type, template <typename ...> class condition, typename ... args>
using valid_decomp_func_2 = typename std::enable_if<is_valid_decomp_matrix_type<m1, m2>::value && is_valid_decomp_matrix_type<m1, m3>::value && condition<args..., remove_cvref_t<typename traits<m1>::value_type>, typename traits<m1>::backend_type>::value, return_type>::type;
}   //namespace internal
}   //namespace linalg

#endif //LINALG_DECOMPOSITIONS_COMMON_HPP

