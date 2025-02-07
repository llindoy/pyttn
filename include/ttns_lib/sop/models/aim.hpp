#ifndef TTNS_SOP_AIM_HAMILTONIAN_HPP
#define TTNS_SOP_AIM_HAMILTONIAN_HPP

#include "model.hpp"
#include "../SOP.hpp"

namespace ttns
{

//a class for handling the generation of the second quantised AIM hamiltonian object. 
template <typename value_type> 
class AIM : public model<value_type>
{
public:
    using real_type = typename linalg::get_real_type<value_type>::type;
    using model<value_type>::hamiltonian;
    using model<value_type>::system_info;

public:
    AIM(){}
    AIM(size_t N, size_t Nimp) : m_T(N+Nimp, N+Nimp), m_U(Nimp, Nimp, Nimp, Nimp), m_impurity_indices(Nimp){}
    AIM(const linalg::matrix<value_type>& _T, const linalg::tensor<value_type, 4>& _U) : m_T(_T), m_U(_U), m_impurity_indices(_U.shape(0))
    {
        ASSERT(_T.shape(0) == _T.shape(1), "Kinetic energy matrix is not square.");
        ASSERT(_U.shape(0) <= _T.shape(0), "Kinetic energy term must have at least as many terms as the potential energy term.");
        ASSERT(_U.shape(0) == _U.shape(1) && _U.shape(0) == _U.shape(2) && _U.shape(0) == _U.shape(3), "Potential energy tensors does not have compatible dimensions.");
        for(size_t i = 0; i < m_impurity_indices.size(); ++i)
        {
            m_impurity_indices[i] = i;
        }
    }
    AIM(const linalg::matrix<value_type>& _T, const linalg::tensor<value_type, 4>& _U, const std::vector<size_t>& impurity_indices) : m_T(_T), m_U(_U), m_impurity_indices(impurity_indices)
    {
        ASSERT(_T.shape(0) == _T.shape(1), "Kinetic energy matrix is not square.");
        ASSERT(_U.shape(0) <= _T.shape(0), "Kinetic energy term must have at least as many terms as the potential energy term.");
        ASSERT(_U.shape(0) == impurity_indices.size(), "Kinetic energy term must have at least as many terms as the potential energy term.");
        ASSERT(_U.shape(0) == _U.shape(1) && _U.shape(0) == _U.shape(2) && _U.shape(0) == _U.shape(3), "Potential energy tensors does not have compatible dimensions.");
    }
    virtual ~AIM(){}

    //functions for building the different sop representations of the Hamiltonian
    virtual void hamiltonian(sSOP<value_type>& H, real_type tol = 1e-14) final
    {
        H.clear();
        size_t N = m_T.shape(0);
        H.reserve(N*N+N*N*N*N);
        build_sop_repr(H, tol);
    }

    virtual void hamiltonian(SOP<value_type>& H, real_type tol = 1e-14) final
    {
        H.clear();
        size_t N = m_T.shape(0);
        H.resize(N);
        build_sop_repr(H, tol);
    }


    virtual void system_info(system_modes& sysinf)
    {
        size_t N = m_T.size(0);
        sysinf.resize(N);
        for(size_t i = 0; i < N; ++i)
        {
            sysinf[i] = fermion_mode();
        }
    }

    //functions for acting the impurity indices
    const std::vector<size_t>& impurity_indices() const{return m_impurity_indices;}
    std::vector<size_t>& impurity_indices(){return m_impurity_indices;}

    const size_t& impurity_index(size_t i) const
    {
          ASSERT(i < m_impurity_indices.size(), "Index out of bounds."); 
          return m_impurity_indices[i];
    }

    size_t& impurity_index(size_t i)
    {
          ASSERT(i < m_impurity_indices.size(), "Index out of bounds."); 
          return m_impurity_indices[i];
    }

    //functions for accessing the kinetic energy matrix
    const linalg::matrix<value_type>& T() const{return m_T;}
    linalg::matrix<value_type>& T(){return m_T;}

    const value_type& T(size_t i, size_t j) const
    {
          ASSERT(i < m_T.size(0) && j < m_T.size(1), "Index out of bounds."); 
          return m_T(i, j);
    }

    value_type& T(size_t i, size_t j)
    {
        ASSERT(i < m_T.size(0) && j < m_T.size(1), "Index out of bounds."); 
        return m_T(i, j);
    }

    //functions for accessing the potential energy tensor
    const linalg::tensor<value_type, 4>& U() const{return m_U;}
    linalg::tensor<value_type, 4>& U(){return m_U;}

    const value_type& U(size_t i, size_t j, size_t k, size_t l) const
    {
        ASSERT(i < m_U.size(0) && j < m_U.size(1) && k < m_U.size(2) && l < m_U.size(3), "Index out of bounds."); 
        return m_U(i, j, k, l);
    }

    value_type& U(size_t i, size_t j, size_t k, size_t l)
    {
        ASSERT(i < m_U.size(0) && j < m_U.size(1) && k < m_U.size(2) && l < m_U.size(3), "Index out of bounds."); 
        return m_U(i, j, k, l);
    }

protected:
    template <typename Hop>
    void build_sop_repr(Hop& H, real_type tol)
    {
        if(m_impurity_indices.size() == 0)
        {
            m_impurity_indices.resize(m_U.shape(0));
            for(size_t i = 0; i < m_U.shape(0); ++i)
            {
                m_impurity_indices[i] = i;
            }
        }

        size_t Nt = m_T.size(0);
        size_t Nimp = m_impurity_indices.size();
        ASSERT(m_T.shape(1) == Nt, "Kinetic energy matrix is not square.");
        ASSERT(m_U.shape(0) == Nimp && m_U.shape(1) == Nimp && m_U.shape(2) == Nimp && m_U.shape(3) == Nimp, "Potential energy tensors does not have compatible dimensions.");

        for(size_t i = 0; i < Nimp; ++i)
        {
            ASSERT(m_impurity_indices[i] < Nt, "Impurity index out of bounds.");
        }

        //bind the kinetic energy terms
        for(size_t p = 0; p < Nt; ++p)
        {
            for(size_t q = 0; q < Nt; ++q)
            {
                if(linalg::abs(m_T(p, q)) > tol)
                {
                    H += m_T(p, q)*fermion_operator("cdag", p)*fermion_operator("c", q);
                }
            }
        }

        //bind the potential energy terms
        for(size_t p = 0; p < Nimp; ++p)
        {
            size_t pi = m_impurity_indices[p];
            for(size_t q = 0; q < Nimp; ++q)
            {
                size_t qi = m_impurity_indices[q];
                for(size_t r=0; r < Nimp; ++r)
                {
                    size_t ri = m_impurity_indices[r];
                    for(size_t s=0; s < Nimp; ++s)
                    {
                        size_t si = m_impurity_indices[s];
                        if(linalg::abs(m_U(p, q, r, s)) > tol)
                        {
                            H += m_U(p, q, r, s)*fermion_operator("cdag", pi)*fermion_operator("cdag", qi)*fermion_operator("c", si)*fermion_operator("c", ri);
                        }
                    }
                }
            }
        }
    }
protected:
    linalg::matrix<value_type> m_T;
    linalg::tensor<value_type, 4> m_U;
    std::vector<size_t> m_impurity_indices;
};
}

#endif

