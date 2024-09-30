#ifndef TTNS_OPERATOR_GEN_SPIN_OPERATORS_HPP
#define TTNS_OPERATOR_GEN_SPIN_OPERATORS_HPP

#include <linalg/linalg.hpp>
#include "single_site_operator.hpp"

//TODO: Need to alter the functions for forming the operators so that if the operator is explicitly complex valued attempting to 
//initialise it with a real variable leads to a runtime error not a compile time error.
namespace ttns
{
namespace spin
{

template <typename T> 
class S_p : public single_site_operator<T>
{
protected:
    static T val(size_t S, size_t i, size_t j)
    {
        T Sp = (S-1.0)/2.0;
        T m1 = Sp - T(i);
        T m2 = Sp - T(j);
        return std::sqrt(Sp*(Sp+1.0) - m1*m2);
    }
public:
    S_p() {}
    virtual bool is_sparse() const{return true;}

    virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis>& /* op */, size_t /* index */, linalg::diagonal_matrix<T>& /* mat */) const
    {
        RAISE_EXCEPTION("Cannot form S_+ as a diagonal operator.  It contains off-diagonal terms in the occupation number basis.");
    }

    virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::csr_matrix<T>& mat) const
    {
        try
        {
            size_t S = op->dim(index);
            ASSERT(index < op->nmodes(), "Index out of bounds.");
            mat.resize(op->nstates(), op->nstates());
    
            size_t nnz = 0;
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                if(op->contains_raised_state(i, index))
                {   
                    ++nnz;
                }
            }

            mat.resize(nnz, op->nstates(), op->nstates());
            auto rowptr = mat.rowptr();    rowptr[0] = 0;
            auto colind = mat.colind();
            auto buffer = mat.buffer();

            size_t counter = 0;
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                if(op->contains_raised_state(i, index))
                {
                    size_t n = op->get_occupation(i, index);
                    size_t m = op->get_raised_index(i, index);
                    buffer[counter] = val(S, n, n+1);
                    colind[counter] = m;
                    ++counter;
                }
                rowptr[i+1] = counter;
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct S_+ as csr.");
        }
    }

    virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::matrix<T>& mat) const
    {
        size_t S = op->dim(index);
        ASSERT(index < op->nmodes(), "Index out of bounds.");
        mat.resize(op->nstates(), op->nstates());
        mat.fill_zeros();
    
        try
        {
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                if(op->contains_raised_state(i, index))
                {
                    size_t n = op->get_occupation(i, index);
                    size_t m = op->get_raised_index(i, index);
                    mat(i, m) = val(S, n, n+1);
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct S_+ as dense.");
        }
    }

    virtual std::pair<T, std::string> transpose() const
    {
        std::pair<T, std::string> ret =  std::make_pair(T(1), std::string("s-"));
        return ret;
    }
};

template <typename T> 
class S_m : public single_site_operator<T>
{
protected:
    static T val(size_t S, size_t i, size_t j)
    {
        T Sp = (S-1.0)/2.0;
        T m1 = Sp - T(i);
        T m2 = Sp - T(j);
        return std::sqrt(Sp*(Sp+1.0) - m1*m2);
    }

public:
    S_m() {}

    virtual bool is_sparse() const{return true;}

    virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis>& /* op */, size_t /* index */, linalg::diagonal_matrix<T>& /* mat */) const
    {
        RAISE_EXCEPTION("Cannot form S_+ as a diagonal operator.  It contains off-diagonal terms in the occupation number basis.");
    }

    virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::csr_matrix<T>& mat) const
    {
        try
        {
            size_t S = op->dim(index);
            ASSERT(index < op->nmodes(), "Index out of bounds.");
            mat.resize(op->nstates(), op->nstates());
    
            size_t nnz = 0;
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                if(op->contains_lowered_state(i, index))
                {   
                    ++nnz;
                }
            }

            mat.resize(nnz, op->nstates(), op->nstates());
            auto rowptr = mat.rowptr();    rowptr[0] = 0;
            auto colind = mat.colind();
            auto buffer = mat.buffer();

            size_t counter = 0;
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                if(op->contains_lowered_state(i, index))
                {
                    size_t n = op->get_occupation(i, index);
                    size_t m = op->get_lowered_index(i, index);
                    buffer[counter] = val(S, n, n-1);
                    colind[counter] = m;
                    ++counter;
                }
                rowptr[i+1] = counter;
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct S_- as csr.");
        }
    }

    virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::matrix<T>& mat) const
    {
        try
        {
            size_t S = op->dim(index);
            ASSERT(index < op->nmodes(), "Index out of bounds.");
            mat.resize(op->nstates(), op->nstates());
            mat.fill_zeros();
    
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                if(op->contains_lowered_state(i, index))
                {
                    size_t n = op->get_occupation(i, index);
                    size_t m = op->get_lowered_index(i, index);
                    mat(i, m) = val(S, n, n-1);
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct S_- as dense.");
        }
    }
    virtual std::pair<T, std::string> transpose() const
    {
        std::pair<T, std::string> ret =  std::make_pair(T(1), std::string("s+"));
        return ret;
    }
};

template <typename T> 
class S_x : public single_site_operator<T>
{
protected:
    static T val(size_t S, size_t i, size_t j)
    {
        T Sp = (S-1.0)/2.0;
        T m1 = Sp - T(i);
        T m2 = Sp - T(j);
        return 0.5*std::sqrt(Sp*(Sp+1.0) - m1*m2);
    }
public:
    S_x() {}

    virtual bool is_sparse() const{return true;}

    virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis>& /* op */, size_t /* index */, linalg::diagonal_matrix<T>& /* mat */) const
    {
        RAISE_EXCEPTION("Cannot form S_+ as a diagonal operator.  It contains off-diagonal terms in the occupation number basis.");
    }

    virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::csr_matrix<T>& mat) const
    {
        try
        {
            size_t S = op->dim(index);
            ASSERT(index < op->nmodes(), "Index out of bounds.");
            mat.resize(op->nstates(), op->nstates());
    
            size_t nnz = 0;
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                if(op->contains_lowered_state(i, index))
                {   
                    ++nnz;
                }
                if(op->contains_raised_state(i, index))
                {   
                    ++nnz;
                }
            }

            mat.resize(nnz, op->nstates(), op->nstates());
            auto rowptr = mat.rowptr();    rowptr[0] = 0;
            auto colind = mat.colind();
            auto buffer = mat.buffer();

            size_t counter = 0;
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                if(op->contains_lowered_state(i, index))
                {
                    size_t n = op->get_occupation(i, index);
                    size_t m = op->get_lowered_index(i, index);
                    buffer[counter] = val(S, n, n-1);
                    colind[counter] = m;
                    ++counter;
                }
                if(op->contains_raised_state(i, index))
                {
                    size_t n = op->get_occupation(i, index);
                    size_t m = op->get_raised_index(i, index);
                    buffer[counter] = val(S, n, n+1);
                    colind[counter] = m;
                    ++counter;
                }
                rowptr[i+1] = counter;
            }

        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct S_x as csr.");
        }
    }

    virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::matrix<T>& mat) const
    {
        try
        {
            size_t S = op->dim(index);
            ASSERT(index < op->nmodes(), "Index out of bounds.");
            mat.resize(op->nstates(), op->nstates());
            mat.fill_zeros();

            for(size_t i = 0; i < op->nstates(); ++i)
            {
                if(op->contains_lowered_state(i, index))
                {
                    size_t n = op->get_occupation(i, index);
                    size_t m = op->get_lowered_index(i, index);
                    mat(i, m) = val(S, n, n-1);
                }
                if(op->contains_raised_state(i, index))
                {
                    size_t n = op->get_occupation(i, index);
                    size_t m = op->get_raised_index(i, index);
                    mat(i, m) = val(S, n, n+1);
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct S_x as dense.");
        }
    }
    virtual std::pair<T, std::string> transpose() const
    {
        std::pair<T, std::string> ret =  std::make_pair(T(1), std::string("sx"));
        return ret;
    }
};

template <typename T, bool is_complex = linalg::is_complex<T>::value>
class S_y;

template <typename T>
class S_y<T, false> : public single_site_operator<T>
{
public:

    S_y() {}

    virtual bool is_sparse() const{return true;}

    virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis>& /* op */, size_t /* index */, linalg::diagonal_matrix<T>& /* mat */) const
    {
        RAISE_EXCEPTION("Cannot form S_y as a diagonal operator.  It contains off-diagonal terms in the occupation number basis.");
    }

    virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis>& /* op */, size_t /* index */, linalg::csr_matrix<T>& /* mat */) const
    {
        RAISE_EXCEPTION("Cannot form S_y as a real valued operator.");
    }

    virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis>& /* op */, size_t /* index */, linalg::matrix<T>& /* mat */) const
    {
        RAISE_EXCEPTION("Cannot form S_y as a real valued operator.");
    }

    virtual std::pair<T, std::string> transpose() const
    {
        std::pair<T, std::string> ret =  std::make_pair(T(-1), std::string("sy"));
        return ret;
    }
};

template <typename RT> 
class S_y<linalg::complex<RT>, true> : public single_site_operator<linalg::complex<RT>>
{
public:
    using T = linalg::complex<RT>;
protected:
    static T val_l(size_t S, size_t i, size_t j)
    {
        T Sp = (S-1.0)/2.0;
        T m1 = Sp - T(i);
        T m2 = Sp - T(j);
        return T(0, 0.5)*std::sqrt(Sp*(Sp+1.0) - m1*m2);
    }


    static T val_r(size_t S, size_t i, size_t j)
    {
        T Sp = (S-1.0)/2.0;
        T m1 = Sp - T(i);
        T m2 = Sp - T(j);
        return T(0, -0.5)*std::sqrt(Sp*(Sp+1.0) - m1*m2);
    }

public:
    S_y() {}

    virtual bool is_sparse() const{return true;}

    virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis>& /* op */, size_t /* index */, linalg::diagonal_matrix<T>& /* mat */) const
    {
        RAISE_EXCEPTION("Cannot form S_+ as a diagonal operator.  It contains off-diagonal terms in the occupation number basis.");
    }

    virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::csr_matrix<T>& mat) const
    {
        try
        {
            size_t S = op->dim(index);
            ASSERT(index < op->nmodes(), "Index out of bounds.");
            mat.resize(op->nstates(), op->nstates());
    
            size_t nnz = 0;
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                if(op->contains_lowered_state(i, index))
                {   
                    ++nnz;
                }
                if(op->contains_raised_state(i, index))
                {   
                    ++nnz;
                }
            }

            mat.resize(nnz, op->nstates(), op->nstates());
            
            auto rowptr = mat.rowptr();    rowptr[0] = 0;
            auto colind = mat.colind();
            auto buffer = mat.buffer();

            size_t counter = 0;
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                if(op->contains_lowered_state(i, index))
                {
                    size_t n = op->get_occupation(i, index);
                    size_t m = op->get_lowered_index(i, index);
                    buffer[counter] = val_l(S, n, n-1);
                    colind[counter] = m;
                    ++counter;
                }
                if(op->contains_raised_state(i, index))
                {
                    size_t n = op->get_occupation(i, index);
                    size_t m = op->get_raised_index(i, index);
                    buffer[counter] = val_r(S, n, n+1);
                    colind[counter] = m;
                    ++counter;
                }
                rowptr[i+1] = counter;
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct S_y as csr.");
        }
    }

    virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::matrix<T>& mat) const
    {
        try
        {
            size_t S = op->dim(index);
            ASSERT(index < op->nmodes(), "Index out of bounds.");
            mat.resize(op->nstates(), op->nstates());
            mat.fill_zeros();
    
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                if(op->contains_lowered_state(i, index))
                {
                    size_t n = op->get_occupation(i, index);
                    size_t m = op->get_lowered_index(i, index);
                    mat(i, m) = val_l(S, n, n-1);
                }
                if(op->contains_raised_state(i, index))
                {
                    size_t n = op->get_occupation(i, index);
                    size_t m = op->get_raised_index(i, index);
                    mat(i, m) = val_r(S, n, n+1);
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct S_y as dense.");
        }
    }
    virtual std::pair<T, std::string> transpose() const
    {
        std::pair<T, std::string> ret =  std::make_pair(T(-1), std::string("sy"));
        return ret;
    }
};

template <typename T> 
class S_z : public single_site_operator<T>
{

protected:
    static T m(size_t S, size_t i){return (S-1.0)/2.0 - i;}

public:
    S_z() {}

    virtual bool is_sparse() const{return true;}
    virtual bool is_diagonal() const{return true;}

    virtual void as_diagonal(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::diagonal_matrix<T>& mat) const
    {
        try
        {
            size_t S = op->dim(index);
            ASSERT(index < op->nmodes(), "Index out of bounds.");
            mat.resize(op->nstates(), op->nstates());
    
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                size_t n = op->get_occupation(i, index);
                mat(i, i) = m(S, n);
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct S_z as dense.");
        }
    }

    virtual void as_csr(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::csr_matrix<T>& mat) const
    {
        try
        {
            size_t S = op->dim(index);
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
                buffer[counter] = m(S, n);
                colind[counter] = i;
                ++counter;
                rowptr[i+1] = counter;
            }
            
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct S_z as csr.");
        }
    }

    virtual void as_dense(const std::shared_ptr<utils::occupation_number_basis>& op, size_t index, linalg::matrix<T>& mat) const
    {
        try
        {
            size_t S = op->dim(index);
            ASSERT(index < op->nmodes(), "Index out of bounds.");
            mat.resize(op->nstates(), op->nstates());
            mat.fill_zeros();
    
            for(size_t i = 0; i < op->nstates(); ++i)
            {
                size_t n = op->get_occupation(i, index);

                mat(i, i) = m(S, n);
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct S_z as dense.");
        }
    }
    virtual std::pair<T, std::string> transpose() const
    {
        std::pair<T, std::string> ret =  std::make_pair(T(1), std::string("sz"));
        return ret;
    }
};

}//namespace pauli

}//namespace ttns
#endif

