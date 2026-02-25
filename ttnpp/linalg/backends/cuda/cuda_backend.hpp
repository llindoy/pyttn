/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2025 NPL Management Limited
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

#ifndef PYTTN_LINALG_BACKENDS_CUDA_BACKEND_HPP_
#define PYTTN_LINALG_BACKENDS_CUDA_BACKEND_HPP_

#include "../../linalg_forward_decl.hpp"
#include "../../utils/linalg_utils.hpp"
#include "../../../common/exception_handling.hpp"
#include "../backend.hpp"
#include "cuda_environment.hpp"

#include <array>
#include <iostream>
#include <vector>
#include <tuple>
#include <utility>
#include <type_traits>


namespace linalg
{
    enum class cuda_transform_type : uint8_t {n, t, h, c};
    enum class eig_mode : uint8_t {no_vectors, vectors};
    enum class fill_mode : uint8_t {upper, lower};

    class cuda_backend : public backend_base
    {
    public:
        using size_type = typename cuda_environment::size_type;
        using index_type = typename cuda_environment::index_type;
        using int_type = index_type;
        using transform_type = cuda_transform_type;
        static constexpr transform_type op_n = cuda_transform_type::n;
        static constexpr transform_type op_t = cuda_transform_type::t;
        static constexpr transform_type op_h = cuda_transform_type::h;
        static constexpr transform_type op_c = cuda_transform_type::c;
    
    protected:
        template <typename T>
        using ones_type = std::pair<T*, size_type>;
        using ones_state = std::tuple<ones_type<float>, ones_type<double>, ones_type<std::complex<float>>, ones_type<std::complex<double>>>;

        static ones_state& ones();
        template <typename T>   static void clean_up_ones();
        static void clean_up_ones();
        template <typename T>   static void initialise_empty_ones_buffer();
        static void initialise_empty_ones_buffers();
        template <typename T>  static void allocate_ones(size_type n);

    public:
        /*
         * Functions for handling environment properties of the backend
         */
        static cuda_environment &environment();
        
        static bool is_initialised();



        // the initialisation routines for the cuda_backend are not thread safe.
        static void initialise(cuda_environment &&env);
        static void initialise(size_type device_id = 0, size_type nstreams = 1);

        static void destroy();

        static std::ostream &device_properties(std::ostream &out);
        static void* current_stream();
        static void synchronise();

    public:
        template <typename F, typename... Args>
        static inline void async_for(size_type start, size_type end, F &&f, Args &&...args)
        {
            try
            {
                ASSERT(environment().is_initialised(), "Failed to perform async_for calculation on cuda_backend.  The backend has not been initialised.");
                ASSERT(start <= end, "Unable to perform async_for.  The final index must be less than the starting index");
                for (size_type i = start; i < end; ++i)
                {
                    f(i, std::forward<Args>(args)...);
                    environment().increment_stream_id();
                }
                environment().reset_stream_id();
                // now we sync all of the streams
                synchronise();
            }
            catch (const common::invalid_value &ex)
            {
                logging::error(ex.what());
                RAISE_NUMERIC("evaluating for loop using asynchronous execution.");
            }
            catch (const std::exception &ex)
            {
                logging::error(ex.what());
                RAISE_EXCEPTION("Failed to evaluate for loop using asynchronous execution.");
            }
        }

    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    static void dgmv_kernel_selector(bool conjA, bool conjB, size_type m, size_type n, T1 alpha, const T2 *A, int_type inca, const T3 *X, int_type incx, T4 beta, T5 *Y, int_type incy);
private:
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    static void dgmm_kernel_parameter_selector(bool sparse_left, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, T1 alpha, const T2 *A, size_type inca, const T3 *B, size_type ldb, T4 beta, T5 *C, size_type ldc);

public:
    template <size_t TILE_DIM, size_t BLOCK_ROWS, typename T1, typename T2, typename T3, typename T4, typename T5>
    static void dgmm_kernel_launcher(bool sparse_left, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, T1 alpha, const T2 *A, size_type inca, const T3 *B, size_type ldb, T4 beta, T5 *C, size_type ldc);


    protected:
        static void set_cublas_stream();
    public:
        template <typename T>
        static bool is_equal(const T *a, const T *b, size_type n);

        template <typename T>
        static typename std::enable_if<is_valid_value_type<T>::value, void>::type vector_scalar_product(int_type N, T A, const T *X, int_type INCX, T *Y, int_type INCY);

        template <typename T>
        static typename std::enable_if<is_valid_value_type<T>::value, T>::type trace(int_type N, const T *X, int_type INCX);

        // valid-1/2/3 via cublas routines
        template <typename T>
        static typename std::enable_if<is_valid_value_type<T>::value, void>::type axpy(int_type N, T A, const T *X, int_type INCX, T *Y, int_type INCY);

        template <typename T>
        static typename std::enable_if<is_valid_value_type<T>::value, void>::type scal(int_type N, T A, T *X, int_type INCX);

        template <typename T>
        static typename std::enable_if<is_valid_value_type<T>::value, T>::type dot(bool conj, int_type N, const T *X, int_type INCX, const T *Y, int_type INCY);

        template <typename T>
        static typename std::enable_if<is_valid_value_type<T>::value, void>::type
        gemm(transform_type TRANSA, transform_type TRANSB, int_type M, int_type N, int_type K, T ALPHA, const T *A, int_type LDA, const T *B, int_type LDB, T BETA, T *C, int_type LDC);

        template <typename T>
        static typename std::enable_if<is_valid_value_type<T>::value, void>::type
        gemv(transform_type trans, int_type m, int_type n, T alpha, const T *A, int_type lda, const T *x, int_type incx, T beta, T *y, int_type incy);

        // batched_gemm
        template <typename T>
        static typename std::enable_if<is_valid_value_type<T>::value, void>::type
        batched_gemm(transform_type opA, transform_type opB, int_type m, int_type n, int_type k, T alpha, const T *A, int_type lda, long long int strideA, const T *B, int_type ldb, long long int strideB, T beta, T *C, int_type ldc, long long int strideC, int_type batchCount);

        // outer contraction of rank 3 tensors
        template <typename T>
        static void outer_contract(transform_type opA, transform_type opB, int_type m, int_type n, int_type k, T alpha, const T *A,
                                          int_type lda, long long int strideA, const T *B, int_type ldb, long long int strideB, T beta, T *C, int_type ldc,
                                          long long int strideC, int_type batchCount, T *res);

    public:
        // sparse matrix vector operations
        template <typename T1, typename T2, typename T3, typename T4, typename T5>
        static void dgmv(bool conjA, bool conjB, int_type m, int_type n, T1 alpha, const T2 *A, int_type inca, const T3 *X, int_type incx, T4 beta, T5 *Y, int_type incy);

        // sparse matrix matrix operations
        template <typename T1, typename T2, typename T3, typename T4, typename T5>
        static void dgmm(bool sparse_left, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, T1 alpha, const T2 *A, size_type inca, const T3 *B, size_type ldb, T4 beta, T5 *C, size_type ldc);

    public:
        template <typename T>
        static void csrmv(transform_type opA, bool /*conjB*/, int_type m, int_type n, size_type nnz, T alpha, const T *A, const int *rowptr, const int *colind, const T *X, int_type incx, T beta, T *Y, int_type incy);
        template <typename T>
        static void csrmv(transform_type opA, bool conjB, int_type m, int_type n, size_type nnz, std::complex<T> alpha, const std::complex<T> *A, const int *rowptr, const int *colind, const std::complex<T> *X, int_type incx, std::complex<T> beta, std::complex<T> *Y, int_type incy);

    public:
        template <typename T>
        static void csrmm(bool opres, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, size_type nnz, T alpha, const T *A, const int *rowptr, const int *colind, const T *B, size_type ldb, T beta, T *C, size_type ldc);

    public:
        template <typename T>
        static void complex_conjugate(size_type /*size*/, const T *const /*X*/, T *const /*Y*/);
        template <typename T>
        static void complex_conjugate(size_type size, const std::complex<T> *const X, std::complex<T> *const Y);

        template <typename T, typename expr>
        static void evaluate_expression_tree(T *res, size_type n, const expr &e);

        template <typename T, typename expr>
        static void evaluate_add_expression_tree(T *res, size_type n, const expr &e);

        template <typename T, typename expr>
        static void evaluate_sub_expression_tree(T *res, size_type n, const expr &e);

        template <typename T, typename expr>
        static void evaluate_expression_tree_strided(T *res, size_type n, size_type stride, const expr &e);

        template <typename T, typename expr>
        static void evaluate_add_expression_tree_strided(T *res, size_type n, size_type stride, const expr &e);

        template <typename T, typename expr>
        static void evaluate_sub_expression_tree_strided(T *res, size_type n, size_type stride, const expr &e);

    public:
        template <typename T>
        static void transpose(bool conj, int_type m, int_type n, const T &alpha, const T *in, const T &beta, T *out);

        // we might want to modify this in the future so that it uses a specialised kernel for performing the batched transpose operation.  This is likely to lead to improved performance
        // for problems in which the overhead of the kernel launches dominates the time to run the operation.
        template <typename T>
        static void batched_transpose(bool conj, size_type m, size_type n, const T &alpha, const T *in, const T &beta, T *out, size_type batchCount);

        // function for copying between two buffers
        template <typename T>
        static void copy(const T *src, size_type n, T *dest);

        template <typename T>
        static void rank_3_strided_copy(const T *src, size_type n1, size_type n2, size_type n3, T *dest, size_type n4);

        template <typename T>
        static void rank_3_strided_append(const T *src, size_type n1, size_type n2, size_type n3, size_type iadd, T *dest, size_type n4);

        template <typename T>
        static void assign(const T *src, size_type n, T *dest, T beta = T(0));

        template <typename T>
        static void addition_assign(const T *src, size_type n, T *dest);

        template <typename T>
        static void subtraction_assign(const T *src, size_type n, T *dest);

        template <typename T>
        static void copy_real_to_complex(const T *src, size_type n, std::complex<T> *dest);

        template <typename T>
        static void addition_assign_real_to_complex(const T *src, size_type n, std::complex<T> *dest);

        template <typename T>
        static void subtraction_assign_real_to_complex(const T *src, size_type n, std::complex<T> *dest);

        template <typename T>
        static void copy_matrix_subblock(size_type m1, size_type n1, const T *src, size_type lda, T *dest, size_type ldb);

        // function for filling a buffer with a value
        template <typename T>
        static void fill_n(T *dest, size_type n, const T &val);

        template <typename T, typename Func, typename... Args>
        static void func_fill_1(T *res, size_type m, Func &&f, Args &&...args);

        template <typename T, typename Func, typename... Args>
        static void func_fill_2(T *res, size_type m, size_type n, Func &&f, Args &&...args);

        template <typename T, typename Func, typename... Args>
        static void func_fill_3(T *res, size_type m, size_type n, size_type o, Func &&f, Args &&...args);

        template <typename T>
        static void fill_matrix_block(const T *src, size_type m, size_type n, T *dest, size_type m2, size_type n2);

        template <typename T, size_t D>
        static void set_tensor_block(const T* src, const std::array<size_type, D>& src_dims, T* dest, const std::array<size_type, D>& dest_dims, const std::array<size_type, D>& skip);

        template <typename T>
        static void transfer_coo_tuple_to_csr(const std::vector<std::tuple<index_type, index_type, T>> &coo, T *vals, index_type *colinds);

    public:
        template <typename T>
        static void heev(eig_mode jobz, fill_mode uplo, int_type n, T *A, int_type lda, typename get_real_type<T>::type *W, T *work, int_type lwork, int *devinfo);

        template <typename T>
        static void heev_buffersize(eig_mode jobz, fill_mode uplo, int_type n, T *A, int_type lda, typename get_real_type<T>::type *W, int *lwork);

    public:
        template <typename T>
        static void getrf(int_type m, int_type n, T *A, int_type lda, T *work, int *ipiv, int *devinfo);

        template <typename T>
        static void getrf_buffersize(int_type m, int_type n, T *A, int_type lda, int *lwork);

    public:
        template <typename T>
        static void gesvd(const char jobu, const char jobv, const int_type m, const int_type n, T *A, const int_type lda, typename get_real_type<T>::type *S, T *U, const int_type ldu, T *VT, const int_type ldvt, T *work, const int_type lwork, typename get_real_type<T>::type *rwork, int *devinfo);

        template <typename T>
        static void gesvd_buffersize(int_type m, int_type n, int &lwork);

        template <typename T>
        static void gesvdj_buffersize(eig_mode jobz, const int_type econ, const int_type m, const int_type n, T *A, const int_type lda, typename get_real_type<T>::type *S, T *U, const int_type ldu, T *VT, const int_type ldvt, int &lwork, void* params);
        template <typename T>
        static void gesvdj(eig_mode jobz, const int_type econ, const int_type m, const int_type n, T *A, const int_type lda, typename get_real_type<T>::type *S, T *U, const int_type ldu, T *VT, const int_type ldvt, T *work, const int_type lwork, int *devinfo, void* params);

    public:
        template <typename T, typename arr1, typename arr2, typename arr3, typename arr4>
        static void tensor_transpose(const T *in, const arr1 &dimsA, const arr2 &_strideA, T *out, const arr3 &dimsB, const arr4 &_strideB, const std::vector<size_type> &inds);

        template <typename T>
        T determinant_reduction(T* red, size_t N);
    }; // cuda_backend

    template <>
    struct traits<cuda_backend>
    {
        using size_type = typename cuda_environment::size_type;
        using index_type = typename cuda_environment::index_type;
        using int_type = index_type;
        static inline std::string label() { return std::string("cuda"); }

    };

} // namespace linalg

#endif // PYTTN_LINALG_BACKENDS_CUDA_BACKEND_HPP_//
