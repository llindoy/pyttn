#ifndef TTNS_SOP_ELECTRONIC_STRUCTURE_HAMILTONIAN_HPP
#define TTNS_SOP_ELECTRONIC_STRUCTURE_HAMILTONIAN_HPP

#include "model.hpp"
#include "../SOP.hpp"

namespace ttns
{

//a class for handling the generation of the second quantised electronic structure hamiltonian object. 
template <typename value_type> 
class electronic_structure : public model<value_type>
{
public:
    using real_type = typename linalg::get_real_type<value_type>::type;
public:
    electronic_structure(){}
    electronic_structure(size_t N) : m_T(N, N), m_U(N, N, N, N) {}
    electronic_structure(const linalg::matrix<value_type>& _T, const linalg::tensor<value_type, 4>& _U) : m_T(_T), m_U(_U) 
    {
        ASSERT(_T.shape(0) == _T.shape(1), "Kinetic energy matrix is not square.");
        ASSERT(_U.shape(0) == _T.shape(0), "Kinetic energy and potential energy tensors do not consider the same number of orbitals.");
        ASSERT(_U.shape(0) == _U.shape(1) && _U.shape(0) == _U.shape(2) && _U.shape(0) == _U.shape(3), "Potential energy tensors does not have compatible dimensions.");
    }
    virtual ~electronic_structure(){}

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
        size_t N = m_T.shape(0);
        sysinf.resize(N);
        for(size_t i = 0; i < N; ++i)
        {
            sysinf[i] = fermion_mode();
        }
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
        size_t N = m_T.size(0);
        ASSERT(m_T.shape(1) == N, "Kinetic energy matrix is not square.");
        ASSERT(m_U.shape(0) == N && m_U.shape(1) == N && m_U.shape(2) == N && m_U.shape(3) == N, "Potential energy tensors does not have compatible dimensions.");

        //bind the kinetic energy terms
        for(size_t p = 0; p < N; ++p)
        {
            for(size_t q = 0; q < N; ++q)
            {
                if(std::abs(m_T(p, q)) > tol)
                {
                    H += m_T(p, q)*fermion_operator("cdag", p)*fermion_operator("c", q);
                }
            }
        }

        //bind the potential energy terms
        for(size_t p = 0; p < N; ++p)
        {
            for(size_t q = 0; q < N; ++q)
            {
                for(size_t r=0; r < N; ++r)
                {
                    for(size_t s=0; s < N; ++s)
                    {
                        if(std::abs(m_U(p, q, r, s)) > tol)
                        {
                            H += m_U(p, q, r, s)*fermion_operator("cdag", p)*fermion_operator("cdag", q)*fermion_operator("c", s)*fermion_operator("c", r);
                        }
                    }
                }
            }
        }
    }
protected:
    linalg::matrix<value_type> m_T;
    linalg::tensor<value_type, 4> m_U;
};
}

#endif

