#ifndef UTILS_QUAD_GAUSS_LAGUERRE_QUADRATURE_HPP_
#define UTILS_QUAD_GAUSS_LAGUERRE_QUADRATURE_HPP_

#include <cstdint>
#include <common/exception_handling.hpp>

#include <linalg/linalg.hpp>
#include <linalg/decompositions/eigensolvers/eigensolver.hpp>

namespace utils{
namespace quad{
namespace gauss{

template<typename T, typename backend>
class laguerre_quad;

template <typename T>
class laguerre_quad<T, linalg::blas_backend>
{
    using RT = typename linalg::get_real_type<T>::type;
public:
    template <typename F> 
    static auto impl(F&& f, const RT& exponent, const RT& alpha, const linalg::vector<RT, linalg::blas_backend>& w, const linalg::vector<RT, linalg::blas_backend>& x) -> decltype(f(exponent))
    {
        decltype(f(exponent)) ret = 0.0;
        for(size_t i=0; i<w.size(); ++i)
        {
            ret += f(x(i)/exponent)*w(i);
        }
        ret /= exponent;//std::pow(exponent, alpha+1);
        return ret;
    }
};

//constructs and applies a gaussian quadrature rule of a given order for a specific orthogonal polynomial
template <typename T, typename backend = linalg::blas_backend>
class associated_laguerre
{
    using RT = typename linalg::get_real_type<T>::type;
public:
    associated_laguerre(size_t N, const RT& alpha, bool normalise_weights = true) :  m_alpha(alpha), m_normalise_weights(normalise_weights), m_w(N), m_x(N)
    {
        construct_quadrature_rule();
    }

    associated_laguerre(const associated_laguerre& o) = default;
    associated_laguerre(associated_laguerre&& o) = default;

    associated_laguerre& operator=(const associated_laguerre& o) = default;
    associated_laguerre& operator=(associated_laguerre&& o) = default;

    void construct_quadrature_rule()
    {
        linalg::symmetric_tridiagonal_matrix<RT, linalg::blas_backend> poly_recurrence(m_w.size(), m_w.size());
        linalg::eigensolver<linalg::symmetric_tridiagonal_matrix<RT, linalg::blas_backend>> eigsolver(m_w.size());
        linalg::vector<RT, linalg::blas_backend> vals(m_w.size());
        linalg::matrix<RT, linalg::blas_backend> vecs(m_w.size(), m_w.size());

        for(size_t i=0; i<m_w.size(); ++i)
        {
            size_t k = i+1;
            poly_recurrence(i, i) = 2.0*k-1.0+m_alpha;
            if(i+1 < m_w.size())
            {
                poly_recurrence(i, i+1) = sqrt( k*(k+m_alpha));
            }
        }

        eigsolver(poly_recurrence, vals, vecs);
        m_x = vals;
        for(size_t i=0; i<m_w.size(); ++i)
        {
            vals(i) = vecs(0, i)*vecs(0, i) * (m_normalise_weights ?  std::tgamma(m_alpha+1) : 1.0);
        }
        m_w = vals;
    }

    void resize(size_t N)
    {
        if(N != m_w.size())
        {
            m_w.resize(N);
            m_x.resize(N);
            construct_quadrature_rule();
        }
    }
    size_t size() const{return m_w.size();}

    const RT& alpha() const{return m_alpha;}

    void set_alpha(const RT& alpha)
    {
        if(alpha != m_alpha)
        {
            m_alpha = alpha;
            construct_quadrature_rule();
        }
    }

    template <typename F>
    auto operator()(F&& f, const RT& exponent = 1.0) const -> decltype(f(exponent))
    {
        ASSERT(exponent > 0, "Cannot perform Gauss-Laguerre quadrature with negative exponent.");
        return laguerre_quad<T, backend>::impl(std::forward<F>(f), exponent, m_alpha, m_w, m_x)*(m_normalise_weights ? 1.0 : std::tgamma(m_alpha+1));
    }

    RT weight_function(RT x, RT exponent = 1.0){return std::pow(x, m_alpha)*std::exp(-x*exponent);}

    const linalg::vector<RT, backend>& w() const{return m_w;}
    const linalg::vector<RT, backend>& x() const{return m_x;}

    RT w(size_t i, RT exponent = 1) const
    {
        ASSERT(i < m_w.size(), "Index out of bounds.");
        return m_w(i) /= std::pow(exponent, alpha+1);
    }

    RT x(size_t i, RT exponent = 1) const
    {
        ASSERT(i < m_x.size(), "Index out of bounds.");
        return m_x(i)/exponent;
    }
protected:
    //the alpha parameter for the laguerre polynomials
    RT m_alpha;
    bool m_normalise_weights;

    //the quadrature points
    linalg::vector<RT, backend> m_w;
    linalg::vector<RT, backend> m_x;
};


template <typename T, typename backend = linalg::blas_backend>
class laguerre : public associated_laguerre<T, backend>
{
public:
    laguerre(size_t N) : associated_laguerre<T, backend>(N, 0.0, true) {}

    laguerre(const laguerre& o) = default;
    laguerre(laguerre&& o) = default;

    laguerre& operator=(const laguerre& o) = default;
    laguerre& operator=(laguerre&& o) = default;
};

}  
}
}

#endif


