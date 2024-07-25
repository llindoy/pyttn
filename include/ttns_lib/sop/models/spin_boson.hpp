#ifndef TTNS_SOP_SPIN_BOSON_HAMILTONIAN_HPP
#define TTNS_SOP_SPIN_BOSON_HAMILTONIAN_HPP

#include "model.hpp"
#include "../SOP.hpp"

namespace ttns
{

template <typename value_type>
class spin_boson_base : public model<value_type>
{
public:
    using real_type = typename linalg::get_real_type<value_type>::type;

    spin_boson_base() : m_eps(0), m_delta(0), m_spin_index(0) {}
    spin_boson_base(size_t spin_index, real_type eps, real_type delta, size_t N) : m_eps(eps), m_delta(delta), m_spin_index(spin_index), m_mode_dims(N) {}

    virtual ~spin_boson_base(){}
  

    //functions for accessing the spin index
    size_t spin_index() const{return m_spin_index;}
    size_t& spin_index(){return m_spin_index;}

    //functions for accessing the bias term
    const real_type& eps() const{return m_eps;}
    real_type& eps(){return m_eps;}

    //functions for accessing the tunneling matirx element
    const real_type& delta() const{return m_delta;}
    real_type& delta(){return m_delta;}

    //functions for accessing the boson mode dimensions
    const std::vector<size_t>& mode_dims() const{return m_mode_dims;}
    std::vector<size_t>& mode_dims(){return m_mode_dims;}
    const size_t& mode_dim(size_t i) const{ASSERT(i < m_mode_dims.size(), "Failed to access mode dimensions."); return m_mode_dims[i];}
    size_t& mode_dim(size_t i){ASSERT(i < m_mode_dims.size(), "Failed to access mode dimensions."); return m_mode_dims[i];}


    //functions for building the different sop representations of the Hamiltonian
    virtual void system_info(system_modes& sysinf) final
    {
        size_t N = m_mode_dims.size();
        sysinf.resize(N+1);
        size_t ci = 0;
        for(size_t i = 0; i < N+1; ++i)
        {
            if(i != m_spin_index)
            {
                sysinf[i] = boson_mode(m_mode_dims[ci]);
                ++ci;
            }
            else
            {
                sysinf[i] = spin_mode(2);
            }
        }
    }
protected:
    template <typename Hop>
    void build_system_op(Hop& H, real_type tol)
    {
        //add on the spin terms
        if(std::abs(m_eps) > tol)
        {   
            H += m_eps * sOP("sz", this->m_spin_index);
        }
        if(std::abs(m_delta) > tol)
        {   
            H += m_delta * sOP("sx", m_spin_index);
        }
    }


    real_type m_eps;
    real_type m_delta;
    size_t m_spin_index;
    std::vector<size_t> m_mode_dims;
};

//a class for handling the generation of the spin_boson hamiltonian with completely generic bath coupling topology
template <typename value_type> 
class spin_boson_generic : public spin_boson_base<value_type>
{
public:
    using base_type = spin_boson_base<value_type>;
    using real_type = typename linalg::get_real_type<value_type>::type;
public:
    spin_boson_generic() : base_type(){}
    spin_boson_generic(real_type eps, real_type delta, const linalg::matrix<value_type>& _T) : base_type(0, eps, delta, _T.shape(0)), m_T(_T){}
    spin_boson_generic(size_t spin_index, real_type eps, real_type delta, const linalg::matrix<value_type>& _T) : base_type(spin_index, eps, delta, _T.shape(0)), m_T(_T){}
    virtual ~spin_boson_generic(){}

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

    //functions for building the different sop representations of the Hamiltonian
    virtual void hamiltonian(sSOP<value_type>& H, real_type tol = 1e-14) final
    {
        H.clear();
        //H.reserve(3*N+1);
        build_sop_repr(H, tol);
    }

    virtual void hamiltonian(SOP<value_type>& H, real_type tol = 1e-14) final
    {
        H.clear();
        H.resize(this->m_mode_dims.size()+1);
        build_sop_repr(H, tol);
    }

protected:
    template <typename Hop>
    void build_sop_repr(Hop& H, real_type tol)
    {
        ASSERT(m_T.shape(0) == m_T.shape(1), "Invalid kinetic energy matrix.");

        this->build_system_op(H, tol);

        std::string li, lj;
        //add on the terms containing bath operators.
        for(size_t i = 0; i < m_T.shape(0); ++i)
        {
            if(i != this->m_spin_index){li = std::string("a");}
            else{li = std::string("sz");}
            for(size_t j = 0; j < m_T.shape(1); ++j)
            {
                if(j != this->m_spin_index){lj = std::string("adag");}
                else{lj = std::string("sz");}

                if(!(i == this->m_spin_index && j == this->m_spin_index))
                {
                    if(std::abs(m_T(i, j)) > tol)
                    {
                        H += m_T(i, j) * sOP(li, i) * sOP(lj, j);
                    }
                }
            }
        }
    }
protected:
    //the kinetic energy hopping matrix.  This 
    linalg::matrix<value_type> m_T;
};


//a class for handling the generation of the spin_boson hamiltonian with star bath coupling topology
template <typename value_type> 
class spin_boson_star : public spin_boson_base<value_type>
{
public:
    using base_type = spin_boson_base<value_type>;
    using real_type = typename linalg::get_real_type<value_type>::type;
public:
    spin_boson_star(){}
    spin_boson_star(real_type eps, real_type delta, const std::vector<real_type>& _w, const std::vector<value_type>& _g) : base_type(0, eps, delta, _w.size()), m_w(_w), m_g(_g)
    {
        ASSERT(_w.size()  == _g.size(), "Invalid boson parameters.")
    }
    spin_boson_star(size_t spin_index, real_type eps, real_type delta, const std::vector<real_type>& _w, const std::vector<value_type>& _g) : base_type(spin_index, eps, delta, _w.size()), m_w(_w), m_g(_g)
    {
        ASSERT(_w.size()  == _g.size(), "Invalid boson parameters.")
    }
    virtual ~spin_boson_star(){}

    //functions for accessing the boson frequencies
    const std::vector<real_type>& w() const{return m_w;}
    std::vector<real_type>& w(){return m_w;}

    const real_type& w(size_t i) const
    {
          ASSERT(i < m_w.size(), "Index out of bounds."); 
          return m_w[i];
    }
    real_type& w(size_t i)
    {
          ASSERT(i < m_w.size(), "Index out of bounds."); 
          return m_w[i];
    }

    //functions for accessing the boson couplings
    const std::vector<value_type>& g() const{return m_g;}
    std::vector<value_type>& g(){return m_g;}

    const value_type& g(size_t i) const
    {
          ASSERT(i < m_g.size(), "Index out of bounds."); 
          return m_g[i];
    }
    value_type& g(size_t i)
    {
          ASSERT(i < m_g.size(), "Index out of bounds."); 
          return m_g[i];
    }

    //functions for building the different sop representations of the Hamiltonian
    virtual void hamiltonian(sSOP<value_type>& H, real_type tol = 1e-14) final
    {
        H.clear();
        //H.resize(N+1);
        build_sop_repr(H, tol);
    }

    virtual void hamiltonian(SOP<value_type>& H, real_type tol = 1e-14) final
    {
        H.clear();
        H.resize(this->m_mode_dims.size()+1);
        build_sop_repr(H, tol);
    }


protected:
    template <typename Hop>
    void build_sop_repr(Hop& H, real_type tol)
    {
        ASSERT(m_w.size() == m_g.size(), "Invalid bath parameters.");

        this->build_system_op(H, tol);

        //add on the terms containing bath operators.
        size_t counter = 0;
        for(size_t i = 0; i < m_g.size(); ++i)
        {
            if(i == this->m_spin_index){++counter;}
            H += (std::sqrt(2.0)*m_g[i])*sOP("sz", this->m_spin_index) * sOP("q", counter);   //write the Hamiltonian in terms of q = frac{1}{\sqrt(2)}(adag + a)
            H += m_w[i]*sOP("n", counter);
            ++counter;
        }
    }

protected:
    std::vector<real_type> m_w;
    std::vector<value_type> m_g;
};


//a class for handling the generation of the spin_boson hamiltonian with chain bath coupling topology
template <typename value_type> 
class spin_boson_chain : public spin_boson_base<value_type>
{
public:
    using real_type = typename linalg::get_real_type<value_type>::type;
    using base_type = spin_boson_base<value_type>;
public:
    spin_boson_chain(){}
    spin_boson_chain(real_type eps, real_type delta, const std::vector<real_type>& _e, const std::vector<value_type>& _t) : base_type(0, eps, delta, _e.size()), m_e(_e), m_t(_t)
    {
        ASSERT(_e.size()  == _t.size(), "Invalid boson parameters.")
    }
    virtual ~spin_boson_chain(){}

    //functions for accessing the boson frequencies
    const std::vector<real_type>& e() const{return m_e;}
    std::vector<real_type>& e(){return m_e;}

    const real_type& e(size_t i) const
    {
          ASSERT(i < m_e.size(0), "Index out of bounds."); 
          return m_e(i);
    }
    real_type& e(size_t i)
    {
          ASSERT(i < m_e.size(0), "Index out of bounds."); 
          return m_e(i);
    }

    //functions for accessing the boson couplings
    const std::vector<value_type>& t() const{return m_t;}
    std::vector<value_type>& t(){return m_t;}

    const value_type& t(size_t i) const
    {
          ASSERT(i < m_t.size(0), "Index out of bounds."); 
          return m_t(i);
    }
    value_type& t(size_t i)
    {
          ASSERT(i < m_t.size(0), "Index out of bounds."); 
          return m_t(i);
    }

    //functions for building the different sop representations of the Hamiltonian
    virtual void hamiltonian(sSOP<value_type>& H, real_type tol = 1e-14) final
    {
        H.clear();
        //H.resize(N+1);
        build_sop_repr(H, tol);
    }

    virtual void hamiltonian(SOP<value_type>& H, real_type tol = 1e-14) final
    {
        H.clear();
        H.resize(this->m_mode_dims.size()+1);
        build_sop_repr(H, tol);
    }
protected:
    template <typename Hop>
    void build_sop_repr(Hop& H, real_type tol)
    {
        ASSERT(m_t.size() == m_e.size(), "Invalid kinetic energy matrix.");
        ASSERT(this->m_spin_index == 0, "spin_boson_chain expects the system Hamiltonian to be at index 0.");

        //add on the spin terms
        this->build_system_op(H, tol);

        H += m_t[0]*sOP("sz", 0)*(sOP("adag", 1)+sOP("a", 1));
        H += m_e[0]*sOP("n", 1);
        //add on the terms containing bath operators.
        size_t counter = 0;
        for(size_t i = 1; i < m_e.size(); ++i)
        {
            H += m_t[i] * (sOP("adag", i)*sOP("a", i+1)+sOP("a", i)*sOP("adag", i+1));
            H += m_e[i]*sOP("n", i+1);
            ++counter;
        }
    }

protected:
    std::vector<real_type> m_e;
    std::vector<value_type> m_t;
};

}

#endif

