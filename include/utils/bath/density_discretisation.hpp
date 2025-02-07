#ifndef UTILS_DENSITY_DISCRETISATION_HPP
#define UTILS_DENSITY_DISCRETISATION_HPP

#include <type_traits>
#include <linalg/linalg.hpp>
#include <cmath>

#include "../quadrature/adaptive_integrate.hpp"
#include "../find_root.hpp"

namespace utils
{

//a function for applying the AAA algorithm to 
class density_discretisation
{
public:
    template <typename Jfunc, typename rhoFunc, typename real_type,  typename T>
    static void discretise(Jfunc&& J, rhoFunc&& rho, real_type wmin, real_type wmax, size_t N, std::vector<T>& g, std::vector<real_type>& w, real_type atol=1e-12, real_type rtol=1e-14, size_t nquad = 100, real_type wtol = 1e-12, real_type ftol=1e-12, size_t niters = 100)
    {
        ASSERT(wmax > wmin, "wmax <= wmin");
        w.resize(N);
        g.resize(N);
        quad::gauss::legendre<real_type> gauss_leg(nquad);
        real_type scaling_factor = 0.0;
        CALL_AND_HANDLE(scaling_factor = quad::adaptive_integrate<T>(std::forward<rhoFunc>(rho), gauss_leg, wmin, wmax, false, atol, true, rtol), "Failed to evaluate scaling factor for the density discretisation.");

        w[N-1] = wmax;
        g[N-1] = std::sqrt(1.0/M_PI*J(wmax)*scaling_factor/rho(wmax)/N);

        for(size_t i = 1; i < N; ++i)
        {
            real_type w0 = w[N-i];
            auto bf = [&gauss_leg, scaling_factor, wmin, atol, rtol, i, N, &rho](real_type wj){return quad::adaptive_integrate<T>(rho, gauss_leg, wmin, wj, false, atol, true, rtol)/scaling_factor - (N-i+0.0)/N;};

            real_type wn = find_root_monotonic
              (
                  w0,
                  bf, 
                  [&rho, scaling_factor](real_type x){return rho(x)/scaling_factor;},
                  niters,
                  wtol,
                  ftol,
                  wmin,
                  w0
              );

            w[N-(i+1)] = wn;
            g[N-(i+1)] = std::sqrt(1.0/M_PI*J(wn)*scaling_factor/rho(wn)/N);
        }
    }
};

}   

#endif  //UTILS_DENSITY_DISCRETISATION_HPP

