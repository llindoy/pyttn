#ifndef TTNS_OPERATOR_GEN_ONE_BODY_OPERATORS_HPP
#define TTNS_OPERATOR_GEN_ONE_BODY_OPERATORS_HPP

#include <linalg/linalg.hpp>
#include <utils/occupation_number_basis_indexing.hpp>

namespace ttns
{

template <typename T> 
class single_site_operator
{
public:
    single_site_operator() {}
    virtual ~single_site_operator(){}

    virtual bool is_diagonal() const{return false;}
    virtual bool is_sparse() const{return false;}
    virtual bool changes_parity() const{return false;}
    virtual bool jw_sign_change() const{return false;}

    virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::diagonal_matrix<T>& mat) const = 0;
    virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::csr_matrix<T>& mat) const = 0;
    virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::matrix<T>& mat) const = 0;

};


template <typename T> 
class diagonal_mode_operator : public single_site_operator<T>
{
public:
    diagonal_mode_operator() {}

    template <typename ... Args>
    diagonal_mode_operator(Args&& ... args) : m_op(std::forward<Args>(args)...){}

    diagonal_mode_operator(const diagonal_mode_operator& o) = default;

    linalg::diagonal_matrix<T>& op(){return m_op;}
    const linalg::diagonal_matrix<T>& op()const {return m_op;}

    virtual bool is_diagonal() const{return true;}
    virtual bool is_sparse() const{return true;}

    virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::diagonal_matrix<T>& mat) const
    {
        try
        {
            ASSERT(op->dim(index) == m_op.shape(0) && op->dim(index) == m_op.shape(1), "Unable to create diagonal matrix one-body operator object.");
            ASSERT(index < op->nmodes(), "Index out of bounds.");
            mat.resize(op->nstates(), op->nstates());
            mat.fill_zeros();
    
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                size_t n = op->get_occupation(i, index);
                mat(i, i) = m_op(n, n);
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct sigma_z as dense.");
        }
    }

    virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::csr_matrix<T>& mat) const
    {
        try
        {
            ASSERT(op->dim(index) == m_op.shape(0) && op->dim(index) == m_op.shape(1), "Unable to create diagonal matrix one-body operator object.");
            ASSERT(index < op->nmodes(), "Index out of bounds.");
            mat.resize(op->nstates(), op->nstates());
    
            mat.resize(op->nstates(), op->nstates(), op->nstates());
            auto rowptr = mat.rowptr();    rowptr[0] = 0;
            auto colind = mat.colind();
            auto buffer = mat.buffer();

            size_t counter = 0;
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                size_t n = op->get_occupation(i, index);
                mat(i, i) = m_op(n, n);
                colind[counter] = i;
                rowptr[i+1] = counter;
            }
            
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct sigma_z as csr.");
        }
    }

    virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::matrix<T>& mat) const
    {
        try
        {
            ASSERT(op->dim(index) == m_op.shape(0) && op->dim(index) == m_op.shape(1), "Unable to create diagonal matrix one-body operator object.");
            ASSERT(index < op->nmodes(), "Index out of bounds.");
            mat.resize(op->nstates(), op->nstates());
            mat.fill_zeros();
    
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                size_t n = op->get_occupation(i, index);
                mat(i, i) = m_op(n, n);
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct sigma_z as dense.");
        }
    }

protected:
    linalg::diagonal_matrix<T> m_op;
};


}//namespace ttns

#endif

