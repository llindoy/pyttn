#ifndef TTNS_SOP_TFIM_HAMILTONIAN_HPP
#define TTNS_SOP_TFIM_HAMILTONIAN_HPP

#include "model.hpp"
#include "../SOP.hpp"

namespace ttns
{

//a class for handling the generation of the second quantised TFIM hamiltonian object. 
template <typename value_type> 
class TFIM : public model<value_type>
{
public:
    using real_type = typename linalg::get_real_type<value_type>::type;
public:
    TFIM(){}
    TFIM(size_t N, real_type _t, real_type _J, bool open_boundary_condition = false) : m_N(N), m_t(_t), m_J(_J), m_open_boundary_conditions(open_boundary_condition){}
    virtual ~TFIM(){}

    //functions for building the different sop representations of the Hamiltonian
    virtual void hamiltonian(sSOP<value_type>& H, real_type tol = 1e-14) final
    {
        H.clear();
        H.reserve(2*m_N - (m_open_boundary_conditions ? 0 : 1));
        build_sop_repr(H, tol);
    }

    virtual void hamiltonian(SOP<value_type>& H, real_type tol = 1e-14) final
    {
        H.clear();
        H.resize(m_N);
        build_sop_repr(H, tol);
    }

    //functions for accessing the size of the model
    size_t N() const{return m_N;}
    size_t& N(){return m_N;}

    //functions for accessing the onsite energy term
    const real_type& t() const{return m_t;}
    real_type& t(){return m_t;}

    //functions for accessing the interaction term
    const real_type& J() const{return m_J;}
    real_type& J(){return m_J;}

    //functions for accessing whether or not to use open boundary conditions
    const bool& open_boundary_conditions() const{return m_open_boundary_conditions;}
    bool& open_boundary_conditions(){return m_open_boundary_conditions;}


    virtual void system_info(system_modes& sysinf) final
    {
        sysinf.resize(m_N);
        for(size_t i = 0; i < m_N; ++i)
        {
            sysinf[i] = spin_mode(2);
        }
    }
protected:
    template <typename Hop>
    void build_sop_repr(Hop& H, real_type tol)
    {
        if(std::abs(m_t) > tol)
        {
            //add on the onsite terms
            for(size_t i = 0; i < m_N; ++i)
            {
                H += m_t * sOP("sx", i);
            }       
        }

        if(std::abs(m_J) > tol)
        {
            //add on the onsite terms
            for(size_t i = 1; i < m_N; ++i)
            {
                H += m_J * sOP("sz", i-1)* sOP("sz", i);
            }       
            if(m_open_boundary_conditions)
            {
                H += m_J*sOP("sz", 0) * sOP("sz", m_N-1);
            }
        }
    }
protected:
    size_t m_N;
    real_type m_t, m_J;
    bool m_open_boundary_conditions;
};
}

#endif

