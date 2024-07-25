#ifndef UTILS_AAA_HPP_
#define UTILS_AAA_HPP_

#include <type_traits>
#include <linalg/linalg.hpp>

#include <linalg/decompositions/generalised_eigensolvers/generalised_eigensolver.hpp>
#include <linalg/decompositions/singular_value_decomposition/singular_value_decomposition.hpp>

#include "quadrature/adaptive_integrate.hpp"

namespace utils
{
template <typename T> 
class AAA_algorithm;

template <typename T> 
linalg::complex<T> Ct(T t, const linalg::vector<linalg::complex<T>>& p, const linalg::vector<linalg::complex<T>>& r)
{
    linalg::complex<T> ret(0, 0);
    linalg::complex<T> ii (0, 1);
    for(size_t i = 0; i < p.size(); ++i)
    {
        if(linalg::imag(p(i)) > 0)
        {
            ret += ii*r(i)*std::exp(ii*p(i)*t);
        }
    }
    return ret;
}

template <typename T> 
class baryocentric_rational_function
{
public:
    using complex_type = linalg::complex<T>;
    baryocentric_rational_function() {}
    baryocentric_rational_function(const baryocentric_rational_function& o) = default;
    baryocentric_rational_function(baryocentric_rational_function&& o) = default;
    baryocentric_rational_function(const linalg::vector<complex_type>& z, const linalg::vector<complex_type>& f, const linalg::vector<complex_type>& w) : m_z(z), m_f(f), m_w(w)
    {
        ASSERT(m_z.size() == m_f.size() && m_z.size() == m_w.size(), "Failed to construct baryocentric rational function.  Invalid inputs.");
    }
    ~baryocentric_rational_function() {}

    baryocentric_rational_function& operator=(const baryocentric_rational_function& o) = default;
    baryocentric_rational_function& operator=(baryocentric_rational_function&& o) = default;

    void resize(size_t n)
    {
        m_z.resize(n);
        m_f.resize(n);
        m_w.resize(n);
    }

    template <typename U> 
    complex_type operator()(const U& z) const
    {
        complex_type p(0, 0);
        complex_type q(0, 0);
        for(size_t i = 0; i < m_z.size(); ++i)
        {
            complex_type CC = T(1.0)/(z - m_z[i]);
            p += CC * (m_f(i) * m_w(i));
            q += CC * m_w(i);
        }
        return p/q;
    }

    //function for computing the poles, residues and zeros of the baryocentric rational function
    void prz(linalg::vector<complex_type>& p, linalg::vector<complex_type>& r, linalg::vector<complex_type>& z, T tol = T(1e-6), size_t qorder = 24) const
    {
        size_t m = m_w.size();
        linalg::matrix<complex_type> B(m+1, m+1);   
        B.fill_zeros(); 
        for(size_t i = 1; i < m+1; ++i){B(i, i) = T(1.0);}

        linalg::matrix<complex_type> M(m+1, m+1);   
        M.fill_zeros(); 
        for(size_t i = 1; i < m+1; ++i){M(i, i) = m_z(i-1);}
        for(size_t i = 1; i < m+1; ++i){M(0, i) = m_w(i-1);}
        for(size_t i = 1; i < m+1; ++i){M(i, 0) = T(1.0);}

        linalg::generalised_eigensolver<linalg::matrix<complex_type>> gsolver;
        linalg::vector<complex_type> alpha(m+1);
        linalg::vector<complex_type> beta(m+1);

        gsolver(M, B, alpha, beta);

        //compute the poles
        size_t npoles = 0;
        for(size_t i = 0; i < m+1; ++i){if(linalg::abs(beta(i)) > std::numeric_limits<T>::epsilon()*T(10)){++npoles;}}

        p.resize(npoles);
        r.resize(npoles);

        size_t ind = 0;     for(size_t i = 0; i < m+1; ++i){if(std::abs(beta(i)) > std::numeric_limits<T>::epsilon()*T(10)){p(ind) = alpha(i)/beta(i);    ++ind;}}
    
        //compute the residues using numerical integration. 
        compute_residues(p, r, tol, qorder);

        //compute the zeros
        for(size_t i = 1; i < m+1; ++i){M(0, i) = m_w(i-1)*m_f(i-1);}
        gsolver(M, B, alpha, beta);
        size_t nzeros = 0;
        for(size_t i = 0; i < m+1; ++i){if(linalg::abs(beta(i)) > std::numeric_limits<T>::epsilon()*T(10)){++nzeros;}}

        z.resize(nzeros);

        ind = 0;    for(size_t i = 0; i < m+1; ++i){if(std::abs(beta(i)) > std::numeric_limits<T>::epsilon()*T(10)){z(ind) = alpha(i)/beta(i);    ++ind;}}

    }

private:
    void compute_residues(const linalg::vector<complex_type>& p, linalg::vector<complex_type>& r, T tol, size_t qorder) const
    {
        quad::gauss::legendre<T> gauss_leg(qorder);
        
        for(size_t i = 0; i < p.size(); ++i)
        {
            T rs = -T(1.0);
            for(size_t j = 0; j < p.size(); ++j)
            {
                if(i != j)  
                {
                    T dist = linalg::real(linalg::abs(p(i)-p(j)));
                    if(rs < 0 || rs > dist){rs = dist;}
                }
            }
            rs /= T(10);

            //now divide the distance by 10 so that we can evaluate the pole using cauchy's theorem without overlapping with other poles
            r(i) = quad::adaptive_integrate<complex_type>
            (
                [this, &rs, &p, &i](T x)
                {   
                    complex_type z(rs*std::cos(x), rs*std::sin(x));
                    complex_type val = this->operator()(p(i)+z)*(z/T(2.0*M_PI));
                    return val;
                }, 
                gauss_leg, T(0.0), T(2.0*M_PI), false, tol, true, tol
            );
        }

    }

protected:
    linalg::vector<complex_type> m_z;
    linalg::vector<complex_type> m_f;
    linalg::vector<complex_type> m_w;

};


//a function for applying the AAA algorithm to 
template <typename T> 
class AAA_algorithm
{
public:
    using complex_type = linalg::complex<T>;
    AAA_algorithm(T tol = 1e-12, int64_t nmax = -1, bool print_outputs = false, bool use_inf_norm = false) : m_tol(tol), m_nmax(nmax), m_print_outputs(print_outputs), m_use_inf_norm(use_inf_norm){}

    const int64_t& nmax() const{return m_nmax;}
    int64_t& nmax(){return m_nmax;}

    const T& tol() const{return m_tol;}
    T& tol(){return m_tol;}

    const bool& print_outputs() const{return m_print_outputs;}
    bool& print_outputs(){return m_print_outputs;}

    const bool& use_inf_norm() const{return m_use_inf_norm;}
    bool& use_inf_norm(){return m_use_inf_norm;}

    template <typename Func, typename U, typename ... Args>
    baryocentric_rational_function<T> operator()(Func&& func, const linalg::vector<U>& _Z, Args&& ... args)
    {
        size_t M = _Z.size();
        linalg::vector<complex_type> _FZ(M);
        for(size_t i = 0; i < M; ++i)
        {
            _FZ(i) = func(_Z(i), std::forward<Args>(args)...);
        }
        CALL_AND_HANDLE(return compute(_FZ, _Z), "Failed to compute AAA decomposition");
    }


    template <typename U2>
    baryocentric_rational_function<T> compute(const linalg::vector<complex_type>& _FZ, const linalg::vector<U2>& _Z)
    {
        try
        {
            ASSERT(_Z.size() == _FZ.size(), "Input vectors are not of the same size.");

            size_t M = _Z.size();

            //allocate the memory for all of the intermediate objects 
            size_t nmax = (m_nmax > 0 ? m_nmax : M);

            f.reallocate(nmax);     f.resize(0);
            z.reallocate(nmax);     z.resize(0);
            w.reallocate(nmax);     w.resize(0);
            wf.reallocate(nmax);    wf.resize(0);

            C.reallocate(M, nmax);      C.resize(M, 0);
            A.reallocate(M, nmax);      A.resize(M, 0);

            N.reallocate(M);         N.resize(M);
            D.reallocate(M);         D.resize(M);

            Z.resize(M);            
            FZ.resize(M);       
            R.resize(M);

            SF.reallocate(M);       SF.resize(M);
            Sf.reallocate(nmax);    Sf.resize(0);

            linalg::singular_value_decomposition<linalg::matrix<complex_type>> svd;
            linalg::matrix<complex_type> U, V;
            linalg::diagonal_matrix<T> S;

            complex_type Fzm(0, 0);
            for(size_t i = 0; i < M; ++i)
            {
                Z(i) = _Z(i);
                FZ(i) = _FZ(i);
                Fzm += FZ(i)/(M*T(1.0));
                SF(i, i) = FZ(i);
            }

            for(size_t i = 0; i < M; ++i)
            {   
                R(i) = Fzm;
            }
            

            for(size_t iter = 0; iter < nmax; ++iter)
            {
                add_point_to_active_set(get_maximum_error_index());
            
                A = SF*C;
                A -= C*Sf;

                svd(A, S, U, V, false);

                //now we set the w matrix from the elements of V
                for(size_t i = 0; i < w.size(); ++i)
                {
                    w(i) = linalg::conj(V(iter, i));
                    wf(i) = w(i)*f(i);
                }

                N = C*wf;
                D = C*w; 
                T err = 0;
                T Fznorm = 0;
                for(size_t i = 0; i < N.size(); ++i)
                {   
                    R(i) = N(i)/D(i);

                    if(m_use_inf_norm)
                    {
                        T eloc = linalg::real(linalg::abs(R(i) - FZ(i)));
                        if(eloc > err){err = eloc;}
                        eloc = linalg::real(linalg::abs(FZ(i)));
                        if(eloc > Fznorm){Fznorm = eloc;}
                    }
                    else
                    {
                        err += linalg::real(linalg::norm(R(i) - FZ(i)));
                        Fznorm += linalg::real(linalg::norm(FZ(i)));
                    }
                }
                if(m_print_outputs)
                {
                    std::cerr << iter << " " << std::sqrt(err/Fznorm) << std::endl;
                }
                if(std::sqrt(err) <= m_tol*std::sqrt(Fznorm))
                {
                    break;
                }
            }

            return baryocentric_rational_function(z, f, w);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply the AAA algorithm to a set of points Z with function values FZ.");
        }
    }

protected:
    void add_point_to_active_set(size_t index)
    {
        ASSERT(index < Z.size(), "Failed to remove index.");

        size_t M = FZ.size();
        size_t m = f.size();

        //expand the active space arrays
        f.resize(m+1);
        z.resize(m+1);
        w.resize(m+1);
        wf.resize(m+1);
        Sf.resize(m+1);

        //and add in the new f and z points
        z(m) = Z(index);
        f(m) = FZ(index);
        for(size_t i = 0; i < m+1; ++i) 
        {
            Sf(i, i) = f(i);
        }

        //remove the added element from the global set
        for(size_t i = index; i < M; ++i)
        {
            if(i+1 < M)
            {
                FZ(i) = FZ(i+1);
                Z(i) = Z(i+1);
            }
            SF(i, i) = FZ(i+1);
        }
    
            
        A.resize(M-1, m);
        //copy the cmatrix into a temporary array
        for(size_t i = 0; i < M-1; ++i)
        {
            size_t ib = i < index ? i : i+1;
            for(size_t j = 0; j < m; ++j)
            {
                A(i, j) = C(ib, j);
            }
        }

        //resize all of the global set buffer.
        FZ.resize(M-1);
        Z.resize(M-1);
        SF.resize(M-1);
        R.resize(M-1);
        C.resize(M-1, m+1);
        N.resize(M-1);
        D.resize(M-1);

        //copy C matrix back now that it has been resize and add in the new column
        for(size_t i = 0; i < M-1; ++i)
        {
            for(size_t j = 0; j < m; ++j)
            {
                C(i, j) = A(i, j);
            }
            C(i, m) = T(1.0)/(Z(i) - z(m));
        }
    
        //resize the A matrix so that it can fit the result
        A.resize(M-1, m+1);
    }
    
    size_t get_maximum_error_index()
    {
        size_t argmax = 0; 
        T max = -T(1.0);
        for(size_t i = 0; i < FZ.size(); ++i)
        {
            T val = linalg::real(linalg::abs(FZ(i)-R(i)));
            if(val > max && std::abs(val-max) > std::numeric_limits<T>::epsilon()*T(10))
            {
                max = val;
                argmax = i;
            }
        }
        return argmax;
    }

protected:
    T m_tol;
    int64_t m_nmax; 
    bool m_print_outputs;
    bool m_use_inf_norm;

    linalg::vector<complex_type> f;
    linalg::vector<complex_type> z;
    linalg::vector<complex_type> w;
    linalg::vector<complex_type> wf;

    linalg::matrix<complex_type> C;
    linalg::matrix<complex_type> A;

    linalg::vector<complex_type> N;
    linalg::vector<complex_type> D;

    linalg::vector<complex_type> Z;
    linalg::vector<complex_type> FZ;

    linalg::vector<complex_type> R;

    linalg::diagonal_matrix<complex_type> SF;
    linalg::diagonal_matrix<complex_type> Sf;
};

}   

#endif  //UTILS_AAA_HPP_//

