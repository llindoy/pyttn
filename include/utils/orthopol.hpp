#ifndef UTILS_ORTHOPOL_HPP_
#define UTILS_ORTHOPOL_HPP_

#include <limits>
#include <cmath>

#include <common/exception_handling.hpp>

#include <linalg/linalg.hpp>
#include <linalg/decompositions/eigensolvers/eigensolver.hpp>

#include "quadrature/adaptive_integrate.hpp"
#include "quadrature/gaussian_quadrature/gauss_jacobi_quadrature.hpp"

namespace utils
{

template <typename T>
class orthopol
{
public:
    orthopol() : m_max_order(0), m_lbound(1.0), m_rbound(-1.0), m_p0(0), m_scale(1), m_shift(0){}
    orthopol(size_t N) : orthopol()
    {   
        CALL_AND_HANDLE(resize(N), "Failed to resize orthopol object.");
    }
    orthopol(size_t N, T lbound, T rbound) : orthopol()
    {   
        CALL_AND_HANDLE(resize(N), "Failed to resize orthopol object.");
        m_lbound = lbound;
        m_rbound = rbound;
    }
    orthopol(size_t N, T lbound, T rbound, T p0) : orthopol()
    {   
        CALL_AND_HANDLE(resize(N), "Failed to resize orthopol object.");
        m_lbound = lbound;
        m_rbound = rbound;
        m_p0 = p0;
    }
    orthopol(const orthopol& o) = default;
    orthopol(orthopol&& o) = default;
   
    orthopol& operator=(const orthopol& o) = default;
    orthopol& operator=(orthopol&& o) = default;

    void resize(size_t n)
    {
        m_max_order = n;
        CALL_AND_HANDLE(m_poly_recurrence.resize(n), "Failed to resize polynomial recurrence relation.");
        //CALL_AND_HANDLE(m_w.resize(n), "Failed to resize quadrature weights.");    
        //CALL_AND_HANDLE(m_x.resize(n), "Failed to resize quadrature nodes.");    
    }
    
    void clear()
    {
        m_max_order = 0;
        m_poly_recurrence.clear();
        m_w.clear();
        m_x.clear();
    }

    
    void set_domain(T lbound, T rbound)
    {
        m_lbound = lbound;
        m_rbound = rbound;
    }

    void scale(const T& scale)
    {
        ASSERT(scale > 0, "Invalid scaling parameter.");
        m_scale *= scale;
        m_shift*=scale;
        m_p0*=scale;
    }

    void shift(const T& shift){m_shift= shift;}

    void set_weight_function_integral(T p0){m_p0 = p0;}

    T xmin() const{return m_lbound*m_scale + m_shift;}
    T xmax() const{return m_rbound*m_scale + m_shift;}

    //compute the value of the normalised orthonormal polynomial.  
    T operator()(const T& x, size_t norder) const
    {
        ASSERT(norder < m_max_order, "Failed to calculate value of the orthonormal polynomial: order is out of bounds.");
        ASSERT(m_p0 != 0, "The zeroth order normalised orthonormal polynomial has not been set.");

        T p0 = 1.0/sqrt(m_p0);
        if(norder == 0){return p0;}
        else
        {
            //compute the first term in the recurrence
            T px = (x - alpha(0))*p0/beta(0);
            T pxm = p0;

            for(size_t i=1; i < norder; ++i)
            {
                T pn = ((x-alpha(i))*px - beta(i-1)*pxm)/beta(i);
                pxm = px;
                px = pn;
            }
            return px;
        }
    }

    T monic(const T& x, size_t norder) const
    {
        ASSERT(norder < m_max_order, "Failed to calculate value of the orthonormal polynomial: order is out of bounds.");

        T p0 = 1.0;
        if(norder == 0){return p0;}
        else
        {
            T a = alpha(0);
            T b;
            //compute the first term in the recurrence
            T px = (x - a)*p0;
            T pxm = p0;

            for(size_t i=1; i < norder; ++i)
            {
                a = alpha(i);
                b = beta(i-1)*beta(i-1);
                T pn = ((x-a)*px - b*pxm);
                pxm = px;
                px = pn;
            }
            return px;
        }
    }

    template <typename At, typename Bt>
    void set_recurrence_relation(const At& alpha, const Bt& beta)
    {
        size_t n = alpha.size(); 
        ASSERT(beta.size() + 1 == alpha.size(), "beta array is incorrect size.");
        CALL_AND_HANDLE(resize(n), "Failed to resize buffers.");
        for(size_t i=0; i<n; ++i)
        {
            //size_t k = i+1;
            m_poly_recurrence(i, i) = alpha[i];
            if(i+1 < n)
            {
                m_poly_recurrence(i, i+1) = beta[i];
            }
        }
    }

    void set_recurrence_relation(const linalg::symmetric_tridiagonal_matrix<T>& o)
    {
        CALL_AND_HANDLE(resize(o.size()), "Failed to resize buffers.");
        m_poly_recurrence = o;
    }

    void set_alpha(size_t i, const T& a)
    {
        ASSERT(i < m_max_order, "Index out of bounds.");
        m_poly_recurrence(i, i ) = a;
    }

    void set_beta(size_t i, const T& b)
    {
        ASSERT(i+1 < m_max_order, "Index out of bounds.");
        m_poly_recurrence(i, i+1) = b;
    }


    void compute_nodes_and_weights(size_t npoints, T normalisation = 1)
    {
        ASSERT(npoints <= m_max_order, "Failed to compute quadrature nodes and weights. ");
        if(m_p0 == T(0)){m_p0 = normalisation;}

        linalg::symmetric_tridiagonal_matrix<T> poly_recurrence(npoints, npoints);
        //std::cerr << m_scale << std::endl;
        for(size_t i = 0; i < npoints; ++i)
        {
            poly_recurrence(i, i) = alpha(i);
            if(i+1 < npoints)
            {
                poly_recurrence(i, i+1) = beta(i);
            }
        }
        linalg::eigensolver<linalg::symmetric_tridiagonal_matrix<T>> eigsolver(m_max_order);

        m_w.resize(npoints);
        m_x.resize(npoints);
        linalg::vector<T> vals(npoints);
        linalg::matrix<T> vecs(npoints, npoints);

        eigsolver(poly_recurrence, vals, vecs);
        m_x = vals;
        for(size_t i=0; i<npoints; ++i)
        {
            vals(i) = vecs(0, i)*vecs(0, i) * m_p0;
        }
        m_w = vals;
    }

    void compute_nodes_and_weights(T normalisation = 1){CALL_AND_RETHROW(compute_nodes_and_weights(m_max_order, normalisation));}

    T alpha(size_t i) const
    {
        ASSERT(i < m_max_order, "Index out of bounds.");
        return m_poly_recurrence(i, i)*m_scale + m_shift;
    }

    T beta(size_t i) const
    {
        ASSERT(i+1 < m_max_order, "Index out of bounds.");
        return m_poly_recurrence(i, i+1)*m_scale;
    }

    const linalg::symmetric_tridiagonal_matrix<T>& recurrence_relation() const{return m_poly_recurrence;}
    const linalg::vector<T>& nodes() const{return m_x;}
    const linalg::vector<T>& weights() const{return m_w;}

    const T& node(size_t i) const{return m_x[i];}
    const T& weight(size_t i) const{return m_w[i];}
    const size_t& Nmax() const{return m_max_order;} 
    size_t npoints() const{return m_w.size();}
    const size_t& size() const{return m_max_order;}

    const T& pi0() const{return m_p0;}
public:
    template <typename vec_type>
    void set_nodes_and_weights(const vec_type& nodes, const vec_type& weights)
    {
        ASSERT(nodes.size() == weights.size(), "Nodes and weights are not the same sizes.");
        ASSERT(nodes.size() <= m_max_order, "Nodes and weights out of bounds.");
        
        m_x.resize(nodes.size());
        m_w.resize(nodes.size());

        for(size_t i = 0; i < nodes.size(); ++i)
        {
            m_x(i) = nodes[i];
            m_w(i) = weights[i];
        }
    }

protected:
    size_t m_max_order;
    linalg::symmetric_tridiagonal_matrix<T> m_poly_recurrence;
    linalg::vector<T> m_w;
    linalg::vector<T> m_x;
    
    T m_lbound;
    T m_rbound;

    T m_p0;

    T m_scale;
    T m_shift;

};


template <typename T> 
void jacobi_polynomial(orthopol<T>& orth, size_t nmax, const T& alpha, const T& beta)
{
    try
    {
        orth.resize(nmax);
        T p0 = std::exp( (alpha+beta+1)*std::log(2.0) + std::lgamma(alpha+1) + std::lgamma(beta+1) - std::lgamma(alpha+beta+2));
        orth.set_weight_function_integral(p0);
        orth.set_domain(-1.0, 1.0);

        T a = alpha;  T b = beta;
        if( std::abs(a+0.5) < 1e-14 && std::abs(b+0.5) < 1e-14)
        {
            for(size_t i=0; i<nmax; ++i)
            {
                orth.set_alpha(i, 0.0);
                if(i+1 < nmax)
                {
                    orth.set_beta(i, 0.5);
                }
                orth.set_beta(0, 1/sqrt(2.0));
            }
        }
        else if(std::abs(a) < 1e-14 && std::abs(b) < 1e-14)
        {
            for(size_t i=0; i<nmax; ++i)
            {
                size_t k = i+1;
                orth.set_alpha(i, 0.0);
                if(i+1 < nmax)
                {
                    orth.set_beta(i, sqrt( (4*k*(k+a)*(k+b)*(k+a+b))/((2*k+a+b-1)*(2*k+a+b)*(2*k+a+b)*(2*k+a+b+1))));
                }
            }
        }
        else
        {
            for(size_t i=0; i<nmax; ++i)
            {
                size_t k = i+1;
                orth.set_alpha(i, (b*b - a*a)/((2*k+a+b)*(2*k+a+b-2)));
                if(i+1 < nmax)
                {
                    orth.set_beta(i, sqrt( (4*k*(k+a)*(k+b)*(k+a+b))/((2*k+a+b-1)*(2*k+a+b)*(2*k+a+b)*(2*k+a+b+1))));
                }
            }
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct Jacobi polynomial object.");
    }
}

template <typename T> 
void gegenbauer_polynomial(orthopol<T>& orth, size_t nmax, const T& alpha )
{
    CALL_AND_HANDLE(jacobi_polynomial(orth, nmax, alpha, alpha), "Failed to construct gegenbauer polynomial object.");
}

template <typename T> 
void chebyshev_polynomial(orthopol<T>& orth, size_t nmax)
{
    CALL_AND_HANDLE(gegenbauer_polynomial(orth, nmax, -0.5), "Failed to construct chebyshev polynomial object.");
}

template <typename T> 
void chebyshev_second_kind_polynomial(orthopol<T>& orth, size_t nmax)
{
    CALL_AND_HANDLE(gegenbauer_polynomial(orth, nmax, 0.5), "Failed to construct chebyshev (second kind) polynomial object.");
}

template <typename T> 
void chebyshev_third_kind_polynomial(orthopol<T>& orth, size_t nmax)
{
    CALL_AND_HANDLE(jacobi_polynomial(orth, nmax, -0.5, 0.5), "Failed to construct chebyshev (second kind) polynomial object.");
}

template <typename T> 
void chebyshev_fourth_kind_polynomial(orthopol<T>& orth, size_t nmax)
{
    CALL_AND_HANDLE(jacobi_polynomial(orth, nmax, 0.5, -0.5), "Failed to construct chebyshev (second kind) polynomial object.");
}

template <typename T> 
void legendre_polynomial(orthopol<T>& orth, size_t nmax)
{
    CALL_AND_HANDLE(gegenbauer_polynomial(orth, nmax, 0.0), "Failed to construct Legendre polynomial object.");
}


template <typename T>
void associated_laguerre_polynomial(orthopol<T>& orth, size_t nmax, const T& alpha)
{
    try
    {
        orth.resize(nmax);
        T p0 = std::tgamma(alpha+1);
        orth.set_weight_function_integral(p0);
        orth.set_domain(0.0, std::numeric_limits<T>::infinity());

        for(size_t i=0; i<nmax; ++i)
        {
            size_t k = i+1;
            orth.set_alpha(i,  2.0*k-1.0+alpha);
            if(i+1 < nmax)
            {
                orth.set_beta(i, sqrt( k*(k+alpha)));
            }
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct Associated Laguerre polynomial object.");
    }
}

template <typename T>
void laguerre_polynomial(orthopol<T>& orth, size_t nmax)
{
    CALL_AND_HANDLE(associated_laguerre_polynomial(orth, nmax, 0.0), "Failed to construct Laguerre polynomial object.");
}


template <typename T>
void hermite_polynomial(orthopol<T>& orth, size_t nmax )
{
    try
    {
        orth.resize(nmax);
        T p0 = std::sqrt(M_PI);
        orth.set_weight_function_integral(p0);
        orth.set_domain(-std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity());

        for(size_t i=0; i<nmax; ++i)
        {
            size_t k = i+1;
            orth.set_alpha(i, 0.0);
            if(i+1 < nmax)
            {
                orth.set_beta(i,  sqrt(k/2.0));
            }
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct Hermite polynomial object.");
    }
}


//computes a set of nonclassical orthonormal polynomials.  Here with the modified moments (integrals of orthref with respect to the weight function of the new polynomials) as input.
template <typename T>
void nonclassical_polynomial(orthopol<T>& orth, const orthopol<T>& orthref, const linalg::vector<T>& modified_moments, T scale_factor = 1)
{
    size_t nmax = modified_moments.size()/2;
    orth.resize(nmax);

    ASSERT(orthref.size() >= 2*nmax, "The reference orthonormal polynomial object is not large enough to calculate all of the moments that are of interest.");
    //now that we have constructed the moments we need to use the wheeler approach to construct the new quadrature rule for this system
    //set up the initial values
    linalg::matrix<T> sigma(nmax+1, 2*nmax);    
    //set the k=-1 => j = 0 elements: 
    sigma.fill_zeros();

    //set the k=0 => j = 1 elements
    for(size_t i=0; i < 2*nmax; ++i){sigma(1, i) = modified_moments[i];}
    
    //need to modify the alpha and beta parameters given the shift
    linalg::vector<T> aref(nmax*2);  linalg::vector<T> bref(nmax*2);
    for(size_t i =0; i < 2*nmax; ++i){aref(i) = orthref.alpha(i);}
    bref(0) = 0;    for(size_t i =1; i < 2*nmax; ++i){bref(i) = orthref.beta(i-1);  bref(i) *= bref(i);}
    
    //set the j elements
    linalg::vector<T> a(nmax);  linalg::vector<T> b(nmax);
    a(0) = aref(0) + modified_moments(1)/modified_moments(0);
    b(0) = 0;
    
    //now we do the iterative steps
    for(size_t k=1; k<nmax; ++k)
    {
        size_t j = k+1;
        //first we update sigma
        for(size_t l = k; l < 2*nmax-k; ++l)
        {
            sigma(j, l) = sigma(j-1, l+1) - (a(k-1) - aref(l))*sigma(j-1, l) - b(k-1)*sigma(j-2, l) + bref(l)*sigma(j-1, l-1);
        }    
        a(k) = aref(k) - sigma(j-1, k)/sigma(j-1, k-1) + sigma(j, k+1)/sigma(j, k);
        b(k) = sigma(j,k)/sigma(j-1, k-1);
    }
    
    //now we set up the orthonormal polynomial object
    orth.set_weight_function_integral(modified_moments[0]/scale_factor);
    for(size_t i=0; i < nmax; ++i)
    {
        orth.set_alpha(i, a(i));
        if(i != 0)
        {
            std::cerr << "b" << i << " " << b(i) << "\r";
            ASSERT(b(i) >= 0, "Failed to construct nonclassical orthogonal polynomial object.  Invalid beta coefficients encountered.  This may be caused by inaccurate evaluation of the moments or overflow or underflow issues in the calculation.  These issues may be resolved by changing the monic polynomials used for the construction of the moments..");
            orth.set_beta(i-1, std::sqrt(b(i)));
        }
    }
}


//construct nonclassical orthogonal polynomial object from a weight function f using the polynomials in orthref to compute modified moments.
//This function will likely fail if the function f has a divergence at zero as the adaptive integration scheme does not necessarily handle this gracefully.
template <typename T, typename F> 
void nonclassical_polynomial(orthopol<T>& orth, const orthopol<T>& orthref, size_t nmax, F&& f, T rel_tol  = 1e-14, T tol = 1e-15, T scale_factor = 1, size_t quadrature_order = 100, size_t max_order = 100)
{
    try
    {
        orth.resize(nmax);
        ASSERT(orthref.size() >= 2*nmax, "The reference orthonormal polynomial object is not large enough to calculate all of the moments that are of interest.");
        
        T xmin = orthref.xmin();
        T xmax = orthref.xmax();

        ASSERT(std::abs(xmin) != std::numeric_limits<T>::infinity() && std::abs(xmax) != std::numeric_limits<T>::infinity(), "The modified Chebyshev method is not suitable for infinite ranges.");
        quad::gauss::legendre<T> gauss_leg(quadrature_order);

        std::cerr << std::setprecision(16);
        //compute the first 2N modified moments of the weight function F using the monic polynomials stored in orthref
        linalg::vector<T> modified_moments(2*nmax);
        for(size_t i = 0; i < 2*nmax; ++i)
        {
            modified_moments[i] = quad::adaptive_integrate([&](T x){return orthref.monic(x, i)*f(x)*scale_factor;}, gauss_leg, xmin, xmax, false, tol, true, rel_tol, 0.0, max_order, false);
            std::cerr << i << " " << modified_moments[i] << "\r";
        }
        CALL_AND_HANDLE(nonclassical_polynomial(orth, orthref, modified_moments, scale_factor), "An issues occured when applying modified Chebyshev algorithm.  This is likely a result of a poor choice of the monic polynomials, resulting in inaccurate evaluation of moments.");
        orth.set_domain(xmin, xmax);
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct orthonormal polynomials that are orthogonal to a non-classical weight function. ");
    }
}


//construct nonclassical orthogonal polynomial object from a weight function f using the polynomials in orthref to compute modified moments.
//This function will likely fail if the function f has a divergence at zero as the adaptive integration scheme does not necessarily handle this gracefully.
template <typename T, typename F> 
void nonclassical_polynomial(orthopol<T>& orth, const orthopol<T>& orthref, T xmin, T xmax, size_t nmax, F&& f, T rel_tol  = 1e-14, T tol = 1e-15, T scale_factor = 1, size_t quadrature_order = 100, size_t max_order = 100)
{
    try
    {
        orth.resize(nmax);
        ASSERT(orthref.size() >= 2*nmax, "The reference orthonormal polynomial object is not large enough to calculate all of the moments that are of interest.");
        
        //ASSERT(xmin >= orthref.xmin() && xmax <= orthref.xmax(), "Unable to compute nonclassical orthogonal polynomail object.  Xmin and xmax are not compatible with reference polynomial.");
        quad::gauss::legendre<T> gauss_leg(quadrature_order);
        ASSERT(std::abs(xmin) != std::numeric_limits<T>::infinity() && std::abs(xmax) != std::numeric_limits<T>::infinity(), "The modified Chebyshev method is not suitable for infinite ranges.");

        std::cerr << std::setprecision(16);
        //compute the first 2N modified moments of the weight function F using the monic polynomials stored in orthref
        linalg::vector<T> modified_moments(2*nmax);
        for(size_t i = 0; i < 2*nmax; ++i)
        {
            modified_moments[i] = quad::adaptive_integrate([&](T x){return orthref.monic(x, i)*f(x)*scale_factor;}, gauss_leg, xmin, xmax, false, tol, true, rel_tol, 0.0, max_order, false);
            std::cerr << i << " " << modified_moments[i] << "\r";
        }
        CALL_AND_HANDLE(nonclassical_polynomial(orth, orthref, modified_moments, scale_factor), "An issues occured when applying modified Chebyshev algorithm.  This is likely a result of a poor choice of the monic polynomials, resulting in inaccurate evaluation of moments.");
        orth.set_domain(xmin, xmax);
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to construct orthonormal polynomials that are orthogonal to a non-classical weight function. ");
    }
}


//Construct the orthonormal polynomial rule given a set of weights and nodes using the Lancsoz algorithm
template <typename T, typename vec_type>
void nonclassical_polynomial(orthopol<T>& orth, const vec_type& x, const vec_type& w)
{
    ASSERT(x.size() == w.size(), "Weights and nodes are not the same size.");
    size_t N = x.size() + 1;

    //now we go ahead and form the symmetric tridiagonal matrix using the RKPW algorithm
    linalg::vector<T> alpha(N);  alpha.fill_zeros();    alpha(0) = 1.0;
    linalg::vector<T> beta(w.size()); beta.fill_zeros();  
    

    for(size_t n = 0; n < w.size(); ++n)
    {
        alpha(n+1) = x[n];
        beta(n) = 0.0;

        T pi_p = w[n];
        T gp = 1.0;   T sk = 0.0;   T sp = 0.0;
        T tp = 0.0;  T tk = 0.0;
        

        for(size_t k=1; k <= n+1; ++k)
        {
            T bp = beta(k-1);   
            T rhok = bp + pi_p;
            beta(k-1) = gp*rhok;
            bool sk0 = false;

            if(std::abs(rhok) < 1e-15)
            {
                gp = 1.0;
                sk = 0.0;
                sk0 = true;
            }
            else
            {
                gp = bp/rhok;
                sk = pi_p/rhok;
            }

            tk = sk*(alpha(k) - x[n]) - gp*tp;
            alpha(k) = alpha(k) - (tk-tp);

            if(sk0){pi_p = sp*bp;}
            else{pi_p = tk*tk/sk;}

            //now advance previous variables
            tp = tk;
            sp = sk;
        }
    }


    orth.resize(w.size());

    //now we set 
    orth.set_weight_function_integral(beta(0));
    for(size_t i = 0; i < w.size(); ++i)
    {
        orth.set_alpha(i, alpha(i+1));
    }
    for(size_t i = 0; i+1 < w.size(); ++i)
    {
        orth.set_beta(i,std::sqrt(beta(i+1)));
    }

    orth.set_nodes_and_weights(x, w);
}

}

#endif  //UTILS_ORTHOPOL_HPP_//

