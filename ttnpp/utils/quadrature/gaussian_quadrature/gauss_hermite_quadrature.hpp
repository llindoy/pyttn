#ifndef UTILS_QUAD_GAUSS_HERMITE_QUADRATURE_HPP_
#define UTILS_QUAD_GAUSS_HERMITE_QUADRATURE_HPP_

#include <cstdint>
#include <common/exception_handling.hpp>

#include <linalg/linalg.hpp>
#include <linalg/decompositions/eigensolvers/eigensolver.hpp>

namespace utils{
namespace quad{
namespace gauss{

template<typename T, typename backend>
class hermite_quad;

template <typename T>
class hermite_quad<T, linalg::blas_backend>
{
    using RT = typename linalg::get_real_type<T>::type;
public:
    template <typename F> 
    static auto impl(F&& f, const RT& exponent, const linalg::vector<RT, linalg::blas_backend>& w, const linalg::vector<RT, linalg::blas_backend>& x) -> decltype(f(exponent))
    {
        decltype(f(exponent)) ret = 0.0;
        for(size_t i=0; i<w.size(); ++i)
        {
            ret += f(x(i)/sqrt(exponent))*w(i);
        }
        ret /= sqrt(exponent);
        return ret;
    }
};

//constructs and applies a gaussian quadrature rule of a given order for a specific orthogonal polynomial
template <typename T, typename backend = linalg::blas_backend>
class hermite
{
    using RT = typename linalg::get_real_type<T>::type;
public:
    hermite(size_t N, bool normalise_weights = true) : m_w(N), m_x(N), m_normalise_weights(normalise_weights)
    {
        construct_quadrature_rule();
    }

    hermite(const hermite& o) = default;
    hermite(hermite&& o) = default;

    hermite& operator=(const hermite& o) = default;
    hermite& operator=(hermite&& o) = default;

    void construct_quadrature_rule()
    {
        linalg::symmetric_tridiagonal_matrix<RT, linalg::blas_backend> poly_recurrence(m_w.size(), m_w.size());
        linalg::eigensolver<linalg::symmetric_tridiagonal_matrix<RT, linalg::blas_backend>> eigsolver(m_w.size());
        linalg::vector<RT, linalg::blas_backend> vals(m_w.size());
        linalg::matrix<RT, linalg::blas_backend> vecs(m_w.size(), m_w.size());

        for(size_t i=0; i<m_w.size(); ++i)
        {
            size_t k = i+1;
            poly_recurrence(i, i) = 0.0;
            if(i+1 < m_w.size())
            {
                poly_recurrence(i, i+1) = sqrt(k/2.0);
            }
        }

        eigsolver(poly_recurrence, vals, vecs);
        m_x = vals;
        for(size_t i=0; i<m_w.size(); ++i)
        {
            vals(i) = vecs(0, i)*vecs(0, i) * (m_normalise_weights ?  sqrt(M_PI) : 1.0);
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

    template <typename F>
    auto operator()(F&& f, const RT& exponent = 1.0) const -> decltype(f(exponent))
    {
        ASSERT(exponent > 0, "Cannot perform Gauss-Laguerre quadrature with negative exponent.");
        return hermite_quad<T, backend>::impl(std::forward<F>(f), exponent, m_w, m_x)*(m_normalise_weights ? 1.0 : sqrt(M_PI));
    }


    RT weight_function(RT x, RT exponent = 1.0){return std::exp(-x*x*exponent);}

    const linalg::vector<RT, backend>& w() const{return m_w;}
    const linalg::vector<RT, backend>& x() const{return m_x;}

    RT w(size_t i, RT exponent = 1) const
    {
        ASSERT(i < m_w.size(), "Index out of bounds.");
        return m_w(i)/sqrt(exponent);
    }

    RT x(size_t i, RT exponent = 1) const
    {
        ASSERT(i < m_x.size(), "Index out of bounds.");
        return m_x(i)/sqrt(exponent);
    }
protected:
    //the quadrature points
    linalg::vector<RT, backend> m_w;
    linalg::vector<RT, backend> m_x;

    //the alpha parameter for the hermite polynomials
    bool m_normalise_weights;
};

}
}
}

#endif


