#ifndef TTNS_OPERATORS_MATRIX_OPERATORS_HPP
#define TTNS_OPERATORS_MATRIX_OPERATORS_HPP

#include <linalg/linalg.hpp>
#include "primitive_operator.hpp"

namespace ttns
{
namespace ops
{

template <typename T, typename backend = linalg::blas_backend> 
class dense_matrix_operator : public primitive<T, backend>
{
public:
    using base_type = primitive<T, backend>;

    //use the parent class type aliases
    using typename base_type::matrix_type;
    using typename base_type::size_type;
    using typename base_type::matrix_ref;
    using typename base_type::const_matrix_ref;
    using typename base_type::real_type;
    using typename base_type::vector_type;
    using typename base_type::vector_ref;
    using typename base_type::const_vector_ref;
    using typename base_type::matview;
    using typename base_type::resview;
    using typename base_type::tensview;
    using typename base_type::restensview;

public:
    dense_matrix_operator()  : base_type() {}
    template <typename ... Args>
    dense_matrix_operator(Args&& ... args) 
    try : base_type(), m_operator(std::forward<Args>(args)...)
    {
        ASSERT
        (
            m_operator.shape(0) == m_operator.shape(1), 
            "Failed to construct dense operator. The operator to be bound must be a square matrix."
        );
        base_type::m_size = m_operator.shape(0);
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct dense matrix operator object.");
    }

    dense_matrix_operator(const dense_matrix_operator& o) = default;
    dense_matrix_operator(dense_matrix_operator&& o) = default;

    dense_matrix_operator& operator=(const dense_matrix_operator& o) = default;
    dense_matrix_operator& operator=(dense_matrix_operator&& o) = default;

    bool is_resizable() const final{return false;}
    void resize(size_type /*n*/){ASSERT(false, "This shouldn't be called.");}

    std::shared_ptr<base_type> clone() const {return std::make_shared<dense_matrix_operator>(m_operator);}
    std::shared_ptr<base_type> transpose() const {return std::make_shared<dense_matrix_operator>(linalg::trans(m_operator));}
    std::shared_ptr<dense_matrix_operator> transpose_matrix() const {return std::make_shared<dense_matrix_operator>(linalg::trans(m_operator));}

    linalg::matrix<T> todense() const 
    {
        linalg::matrix<T> ret(m_operator);
        return ret;
    }

    void apply(const resview& A, resview& HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}  
    void apply(const resview& A, resview& HA, real_type /* t */, real_type /* dt */) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}  
    void apply(const matview& A, resview& HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));} 
    void apply(const matview& A, resview& HA, real_type /* t */, real_type /* dt */) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
  
    void apply(const_matrix_ref A, matrix_ref HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_vector_ref A, vector_ref HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}

    //functions for applying the operator to rank 3 tensor views
    void apply(const restensview& A, restensview& HA) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const restensview& A, restensview& HA, real_type /* t */, real_type /* dt */) final  {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const tensview& A, restensview& HA) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const tensview& A, restensview& HA, real_type /* t */, real_type /* dt */) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}

protected:
    template <typename A1, typename A2>
    void apply_rank_2(const A1& A, A2& HA)
    {
        CALL_AND_HANDLE(HA = m_operator*A, "Failed to apply dense matrix operator.  Failed to rank 2 tensor.");
    }  
    template <typename A1, typename A2>
    void apply_rank_3(const A1& A, A2& HA)
    {
        CALL_AND_HANDLE(HA = linalg::contract(m_operator, 1, A, 1), "Failed to apply dense matrix operator to rank 3 tensor.");
    }

public:
    void update(real_type /*t*/, real_type /*dt*/) final{}  
    const matrix_type& mat()const{return m_operator;}

    std::string to_string() const final
    {
        std::stringstream oss;
        oss << "dense matrix operator: " << std::endl;
        oss << m_operator << std::endl;
        return oss.str();
    }

protected:
    matrix_type m_operator;

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE
        (
            ar(cereal::base_class<primitive<T, backend> >(this)), 
            "Failed to serialise dense_matrix operator object.  Error when serialising the base object."
        );

        CALL_AND_HANDLE
        (
            ar(cereal::make_nvp("matrix", m_operator)), 
            "Failed to serialise dense_matrix operator object.  Error when serialising the matrix."
        );
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE
        (
            ar(cereal::base_class<primitive<T, backend> >(this)), 
            "Failed to serialise dense_matrix operator object.  Error when serialising the base object."
        );
        CALL_AND_HANDLE
        (
            ar(cereal::make_nvp("matrix", m_operator)), 
            "Failed to serialise dense_matrix operator object.  Error when serialising the matrix."
        );
    }
#endif
};

//A class for wrapping the multiplication by a sparse matrix.  This stores the matrix in csr form to make it easy to perform all of the required operations.
template <typename T, typename backend = linalg::blas_backend>
class sparse_matrix_operator;

//blas specialisation of the sparse matrix operator.
template <typename T> 
class sparse_matrix_operator<T, linalg::blas_backend> : public primitive<T, linalg::blas_backend>
{
public:
    using backend = linalg::blas_backend;
    using base_type = primitive<T, backend>;

    //use the parent class type aliases
    using typename base_type::matrix_type;
    using typename base_type::size_type;
    using typename base_type::matrix_ref;
    using typename base_type::const_matrix_ref;
    using typename base_type::real_type;
    using typename base_type::vector_type;
    using typename base_type::vector_ref;
    using typename base_type::const_vector_ref;
    using typename base_type::matview;
    using typename base_type::resview;
    using typename base_type::tensview;
    using typename base_type::restensview;

public:
    sparse_matrix_operator() : base_type() {}
    sparse_matrix_operator(const sparse_matrix_operator& o) = default;
    sparse_matrix_operator(sparse_matrix_operator&& o) = default;
    template <typename ... Args>
    sparse_matrix_operator(Args&& ... args) 
    try : base_type(), m_operator(std::forward<Args>(args)...)
    {
        ASSERT(m_operator.dims(0) == m_operator.dims(1), "The operator to be bound must be a square matrix.");
        base_type::m_size = m_operator.dims(0);
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct sparse matrix operator object.");
    }

    bool is_resizable() const final{return false;}
    void resize(size_type /*n*/){ASSERT(false, "This shouldn't be called.");}

    linalg::matrix<T> todense() const 
    {
        return m_operator.todense();
    }

    std::shared_ptr<base_type> clone() const{return std::make_shared<sparse_matrix_operator>(m_operator);}
    std::shared_ptr<base_type> transpose() const
    {
        linalg::csr_matrix<T, backend> mat;
        m_operator.transpose(mat);
        return std::make_shared<sparse_matrix_operator>(mat);
    }

    void apply(const resview& A, resview& HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}  
    void apply(const resview& A, resview& HA, real_type /* t */, real_type /* dt */) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}  
    void apply(const matview& A, resview& HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));} 
    void apply(const matview& A, resview& HA, real_type /* t */, real_type /* dt */) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
  
    void apply(const_matrix_ref A, matrix_ref HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_vector_ref A, vector_ref HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}

    //functions for applying the operator to rank 3 tensor views
    void apply(const restensview& A, restensview& HA) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const restensview& A, restensview& HA, real_type /* t */, real_type /* dt */) final  {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const tensview& A, restensview& HA) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const tensview& A, restensview& HA, real_type /* t */, real_type /* dt */) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}

protected:
    template <typename A1, typename A2>
    void apply_rank_2(const A1& A, A2& HA) 
    {
        ASSERT(this->size() == A.shape(0), "Failed to apply rank 2 contraction.");
        CALL_AND_HANDLE(HA = m_operator*A, "Failed to apply sparse matrix operator.  Failed to rank 2 tensor.");
    }  
    template <typename At1, typename At2>
    void apply_rank_3(const At1& A, At2& HA)
    {
        ASSERT(this->size() == A.shape(1), "Failed to apply rank 3 contraction.");
        for(size_t i = 0; i < A.shape(0); ++i)
        {
            CALL_AND_HANDLE(HA[i] = m_operator*A[i], "Failed to apply sparse matrix operator.");
        }
    }

public:

    void update(real_type /*t*/, real_type /*dt*/) final{}  
    const linalg::csr_matrix<T, backend>& mat()const{return m_operator;}

    std::string to_string() const final
    {
        std::stringstream oss;
        oss << "sparse matrix operator: " << std::endl;
        oss << m_operator << std::endl;
        return oss.str();
    }

protected:
    linalg::csr_matrix<T, backend> m_operator;
#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE
        (
            ar(cereal::base_class<primitive<T, backend> >(this)), 
            "Failed to serialise sparse_matrix operator object.  Error when serialising the base object."
        );
        CALL_AND_HANDLE
        (
            ar(cereal::make_nvp("matrix", m_operator)), 
            "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix."
        );
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE
        (
            ar(cereal::base_class<primitive<T, backend> >(this)), 
            "Failed to serialise sparse_matrix operator object.  Error when serialising the base object."
        );

        CALL_AND_HANDLE
        (
            ar(cereal::make_nvp("matrix", m_operator)), 
            "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix."
        );
    }
#endif
};

#ifdef __NVCC__

template <typename T> 
class sparse_matrix_operator<T, linalg::cuda_backend> : public primitive<T, linalg::cuda_backend>
{
public:
    using backend = linalg::cuda_backend;
    using base_type = primitive<T, backend>;

    //use the parent class type aliases
    using typename base_type::matrix_type;
    using typename base_type::size_type;
    using typename base_type::matrix_ref;
    using typename base_type::const_matrix_ref;
    using typename base_type::real_type;
    using typename base_type::vector_type;
    using typename base_type::vector_ref;
    using typename base_type::const_vector_ref;
    using typename base_type::matview;
    using typename base_type::resview;
    using typename base_type::tensview;
    using typename base_type::restensview;

protected:
    linalg::csr_matrix<T, backend> m_operator;
    linalg::matrix<T, backend> m_temp;
    linalg::vector<T, backend> m_externalBuffer;

    cusparseDnMatDescr_t m_rd;
    cusparseSpMatDescr_t m_Hd;
    bool m_sparse_init;
    bool m_temp_init;

public:
    sparse_matrix_operator() : base_type(), m_sparse_init(false), m_temp_init(false) {}
    sparse_matrix_operator(const sparse_matrix_operator& o) : base_type(static_cast<const base_type&>(o))
    {
        try
        {
            CALL_AND_HANDLE(m_operator = o.m_operator, "Failed to copy construct csr matrix.");
            CALL_AND_HANDLE(m_temp = o.m_temp, "Failed to copy construct temp matrix.");
            CALL_AND_HANDLE(m_externalBuffer = o.m_externalBuffer, "Failed to copy construct external buffer.");
            CALL_AND_HANDLE(create_sparse_descriptor(m_Hd, m_operator), "Failed to allocate sparse descriptor.");
            m_sparse_init = true;
            m_temp_init = false;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to copy construct sparse matrix operator.");
        }
    }
    sparse_matrix_operator(sparse_matrix_operator&& o) : 
        base_type(std::forward<base_type>(o)), m_operator(std::move(o.m_operator)), 
        m_temp(std::move(o.m_temp)), m_externalBuffer(std::move(o.m_externalBuffer))
    {
        try
        {
            CALL_AND_HANDLE(create_sparse_descriptor(m_Hd, m_operator), "Failed to allocate sparse descriptor.");
            m_sparse_init = true;
            m_temp_init = false;
            CALL_AND_HANDLE(o.deallocate_descriptors(), "Failed to deallocate descriptors.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to move construct sparse matrix operator.");
        }
    }

    template <typename ... Args>
    sparse_matrix_operator(Args&& ... args) try : base_type(), m_operator(std::forward<Args>(args)...)
    {
        ASSERT(m_operator.dims(0) == m_operator.dims(1), "The operator to be bound must be a square matrix.");
        CALL_AND_HANDLE(create_sparse_descriptor(m_Hd, m_operator), "Failed to allocate sparse descriptor.");
        m_sparse_init = true;
        m_temp_init = false;
        base_type::m_size = m_operator.dims(0);
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct sparse matrix operator object.");  
    }

    ~sparse_matrix_operator()
    {
        try
        {
            deallocate_descriptors();
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            std::cerr << "Exception raised when calling destructor of sparse matrix operator." << std::endl;
            exit(1);
        }
    }

    bool is_resizable() const final{return false;}

    void resize(size_type /*n*/){ASSERT(false, "This shouldn't be called.");}

    std::shared_ptr<base_type> clone() const{return std::make_shared<sparse_matrix_operator>(m_operator);}
    std::shared_ptr<base_type> transpose() const {RAISE_EXCEPTION("Transpose of cuda sparse matrices currently not supported.");}

    const linalg::csr_matrix<T, backend>& mat()const{return m_operator;}

    void apply(const resview& A, resview& HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}  
    void apply(const resview& A, resview& HA, real_type /* t */, real_type /* dt */) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}  
    void apply(const matview& A, resview& HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));} 
    void apply(const matview& A, resview& HA, real_type /* t */, real_type /* dt */) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
  
    void apply(const_matrix_ref A, matrix_ref HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_vector_ref A, vector_ref HA) final{RAISE_EXCEPTION("Sparse matrix vector product is not implemented for CUDA.");}
    void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final{RAISE_EXCEPTION("Sparse matrix vector product is not implemented for CUDA.");}

    //functions for applying the operator to rank 3 tensor views
    void apply(const restensview& A, restensview& HA) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const restensview& A, restensview& HA, real_type /* t */, real_type /* dt */) final  {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const tensview& A, restensview& HA) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const tensview& A, restensview& HA, real_type /* t */, real_type /* dt */) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}

protected:
    template <typename At1, typename At2>
    void apply_rank_3(const At1& A, At2& HA)
    {
        RAISE_EXCEPTION("CSR TENSOR MULTIPLICATION NOT SUPPORTED FOR CUDA MATRICES.");
    }
    template <typename Atype, typename Rtype>
    void apply_rank_2(const Atype& a, Rtype& HA)
    {
        ASSERT
        (
            backend::environment().is_initialised(), 
            "Failed to apply sparse matrix operator.  The cuda backend has not been initialised."
        );
        ASSERT
        (   
            m_sparse_init, 
            "Failed to apply sparse matrix operator.  The sparse matrix descriptor has not been initialised."
        );

        if(m_temp.shape() != A.shape())
        {
            CALL_AND_HANDLE(m_temp.resize(A.shape(1), A.shape(0)), "Failed to resize HA array.");
            if(m_temp_init)
            {
                CALL_AND_HANDLE
                (
                    deallocate_dense_descriptor(m_rd), 
                    "Failed to destroy previously constructed cusparse descriptor."
                );
            }
            CALL_AND_HANDLE(create_dense_descriptor(m_rd, m_temp), "Failed to create new cusparse descriptor.");
            m_temp_init = true;
        }

        CALL_AND_HANDLE(HA = trans(A), "Evaluate the transpose of A so we can compute the csr matrix product");
        //now we set up the matrix description for the input A matrix
        cusparseDnMatDescr_t Ad;
        CALL_AND_HANDLE(create_dense_descriptor(Ad, HA), "Failed to create new cusparse dense descriptor.");
        
        //now we query the buffer size
        size_t bufferSize;
        T one(1.0); T zero(0.0);
        CALL_AND_HANDLE
        (
            cusparse_safe_call
            (
                cusparseSpMM_bufferSize
                (   
                    backend::environment().cusparse_handle(), 
                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    &one, 
                    m_Hd, 
                    Ad, 
                    &zero,
                    m_rd, 
                    linalg::cuda_type<T>::type_enum(), 
                    CUSPARSE_MM_ALG_DEFAULT, 
                    &bufferSize
                )
            ), 
            "Failed to determine buffer size for cusparseSpMM call"
        );

        CALL_AND_HANDLE(m_externalBuffer.resize(bufferSize), "Failed to resize buffer size.");

        CALL_AND_HANDLE
        (
            cusparse_safe_call
            (
                cusparseSpMM
                (
                    backend::environment().cusparse_handle(), 
                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    &one, 
                    m_Hd, 
                    Ad, 
                    &zero, 
                    m_rd, 
                    linalg::cuda_type<T>::type_enum(), 
                    CUSPARSE_MM_ALG_DEFAULT, 
                    m_externalBuffer.buffer()
                )
            ), 
            "Failed to evaluate cusparseSpMM call"
        );

        //now we transpose m_rd into HA
        CALL_AND_HANDLE(HA = trans(m_temp), "Failed to evaluate transpose of result.");

        CALL_AND_HANDLE
        (
            deallocate_dense_descriptor(Ad), 
            "Failed to destroy previously constructed cusparse descriptor."
        );
    }


    void deallocate_dense_descriptor(cusparseDnMatDescr_t& desc)
    {
        CALL_AND_HANDLE(cusparse_safe_call(cusparseDestroyDnMat(desc)), "Failed to deallocate dense matrix descriptor.");
    }
    
    void deallocate_descriptors()
    {
        if(m_temp_init)
        {
            CALL_AND_HANDLE(cusparse_safe_call(cusparseDestroyDnMat(m_rd)), "Failed to deallocate the dense matrix descriptor.");
            m_temp_init = false;
        }
        if(m_sparse_init)
        {
            CALL_AND_HANDLE(cusparse_safe_call(cusparseDestroySpMat(m_Hd)), "Failed to deallocate the m_Hd sparse matrix descriptor.");
            m_sparse_init = false;
        }
    }

    template <typename MatType>
    void create_dense_descriptor(cusparseDnMatDescr_t& desc, MatType& m)
    {
        //this descriptor treats m as m^T
        CALL_AND_HANDLE(cusparse_safe_call(cusparseCreateDnMat(&desc, static_cast<int64_t>(m.shape(1)), static_cast<int64_t>(m.shape(0)), static_cast<int64_t>(m.shape(1)), m.buffer(), linalg::cuda_type<T>::type_enum(), CUSPARSE_ORDER_COL)), "Failed to create dense matrix descriptor.");
    }

    void create_sparse_descriptor(cusparseSpMatDescr_t& desc, csr_matrix<T, backend>& m)
    {
        CALL_AND_HANDLE(cusparse_safe_call(cusparseCreateCsr(&desc, static_cast<int64_t>(m.shape(1)), static_cast<int64_t>(m.shape(0)), static_cast<int64_t>(m.nnz()), m.rowptr(), m.colind(), m.buffer(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, linalg::cuda_type<T>::type_enum())), "Failed to create sparse matrix descriptor.");
    }

    std::string to_string() const final
    {
        std::stringstream oss;
        oss << "sparse matrix operator: " << std::endl;
        oss << m_operator << std::endl;
        return oss.str();
    }
#ifdef CEREAL_LIBRARY_FOUND


    linalg::matrix<T> todense() const 
    {
        return m_operator.todense();
    }
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise sparse_matrix operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operator)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE(ar(cereal::base_class<primitive<T, backend> >(this)), "Failed to serialise sparse_matrix operator object.  Error when serialising the base object.");
        CALL_AND_HANDLE(ar(cereal::make_nvp("matrix", m_operator)), "Failed to serialise sparse_matrix operator object.  Error when serialising the matrix.");
    }
#endif
};
#endif


template <typename T, typename backend = linalg::blas_backend> 
class diagonal_matrix_operator : public primitive<T, backend>
{
public:
    using base_type = primitive<T, backend>;

    //use the parent type aliases
    using typename base_type::matrix_type;
    using typename base_type::size_type;
    using typename base_type::matrix_ref;
    using typename base_type::const_matrix_ref;
    using typename base_type::real_type;
    using typename base_type::vector_type;
    using typename base_type::vector_ref;
    using typename base_type::const_vector_ref;
    using typename base_type::matview;
    using typename base_type::resview;
    using typename base_type::tensview;
    using typename base_type::restensview;

public:
    diagonal_matrix_operator() : base_type() {}
    template <typename ... Args>
    diagonal_matrix_operator(Args&& ... args) 
    try : base_type(), m_operator(std::forward<Args>(args)...)
    {
        ASSERT
        (
            m_operator.shape(0) == m_operator.shape(1), 
            "The operator to be bound must be a square matrix."
        );
        base_type::m_size = m_operator.shape(0);
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct diagonal matrix operator object.");
    }

    diagonal_matrix_operator(const diagonal_matrix_operator& o) = default;
    diagonal_matrix_operator(diagonal_matrix_operator&& o) = default;

    diagonal_matrix_operator& operator=(const diagonal_matrix_operator& o) = default;
    diagonal_matrix_operator& operator=(diagonal_matrix_operator&& o) = default;

    bool is_resizable() const final{return false;}
    void resize(size_type /* n */){ASSERT(false, "This shouldn't be called.");}

    std::shared_ptr<base_type> clone() const{return std::make_shared<diagonal_matrix_operator>(m_operator);}
    std::shared_ptr<base_type> transpose() const {return std::make_shared<diagonal_matrix_operator>(m_operator);}

    linalg::matrix<T> todense() const 
    {
        return m_operator.todense();
    }

    void apply(const resview& A, resview& HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}  
    void apply(const resview& A, resview& HA, real_type /* t */, real_type /* dt */) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}  
    void apply(const matview& A, resview& HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));} 
    void apply(const matview& A, resview& HA, real_type /* t */, real_type /* dt */) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
  

    void apply(const_matrix_ref A, matrix_ref HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_matrix_ref A, matrix_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_vector_ref A, vector_ref HA) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}
    void apply(const_vector_ref A, vector_ref HA, real_type /*t*/, real_type /*dt*/) final{CALL_AND_RETHROW(apply_rank_2(A, HA));}

    //functions for applying the operator to rank 3 tensor views
    void apply(const restensview& A, restensview& HA) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const restensview& A, restensview& HA, real_type /* t */, real_type /* dt */) final  {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const tensview& A, restensview& HA) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}
    void apply(const tensview& A, restensview& HA, real_type /* t */, real_type /* dt */) final {CALL_AND_RETHROW(apply_rank_3(A, HA));}

protected:
    template <typename A1, typename A2>
    void apply_rank_2(const A1& A, A2& HA) 
    {
        CALL_AND_HANDLE(HA = m_operator*A, "Failed to apply dense matrix operator.  Failed to rank 2 tensor.");
    }  
    template <typename At1, typename At2>
    void apply_rank_3(const At1& A, At2& HA)
    {
        for(size_t i = 0; i < A.shape(0); ++i)
        {
            CALL_AND_RETHROW(HA[i] = m_operator*A[i]);
        }
    }

public:

    void update(real_type /*t*/, real_type /*dt*/) final{}  
    const linalg::diagonal_matrix<T, backend>& mat()const{return m_operator;}

    std::string to_string() const final
    {
        std::stringstream oss;
        oss << "diagonal matrix operator: " << std::endl;
        oss << m_operator << std::endl;
        return oss.str();
    }

protected:
    linalg::diagonal_matrix<T, backend> m_operator;

#ifdef CEREAL_LIBRARY_FOUND
public:
    template <typename archive>
    void save(archive& ar) const
    {
        CALL_AND_HANDLE
        (
            ar(cereal::base_class<primitive<T, backend> >(this)), 
            "Failed to serialise diagonal_matrix operator object.  Error when serialising the base object."
        );

        CALL_AND_HANDLE
        (
            ar(cereal::make_nvp("matrix", m_operator)), 
            "Failed to serialise diagonal_matrix operator object.  Error when serialising the matrix."
        );
    }

    template <typename archive>
    void load(archive& ar)
    {
        CALL_AND_HANDLE
        (
            ar(cereal::base_class<primitive<T, backend> >(this)), 
            "Failed to serialise diagonal_matrix operator object.  Error when serialising the base object."
        );

        CALL_AND_HANDLE
        (
            ar(cereal::make_nvp("matrix", m_operator)), 
            "Failed to serialise diagonal_matrix operator object.  Error when serialising the matrix."
        );
    }
#endif
};

}   //namespace ops
}   //namespace ttns


#ifdef CEREAL_LIBRARY_FOUND
TTNS_REGISTER_SERIALIZATION(ttns::ops::dense_matrix_operator, ttns::ops::primitive)
TTNS_REGISTER_SERIALIZATION(ttns::ops::sparse_matrix_operator, ttns::ops::primitive)
TTNS_REGISTER_SERIALIZATION(ttns::ops::diagonal_matrix_operator, ttns::ops::primitive)
#endif


#endif  //TTNS_OPERATORS_MATRIX_OPERATOR_HPP

