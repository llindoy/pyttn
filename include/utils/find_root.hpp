#ifndef UTILS_ROOT_FINDER_HPP_
#define UTILS_ROOT_FINDER_HPP_

#include <cmath>
#include <limits>

namespace utils
{
template <typename T, typename F, typename DF> 
T find_root_monotonic(const T& x0, F&& f, DF&& df, size_t niters = 100, T xtol = 1e-6, T ftol = 1e-6, T xmin = -std::numeric_limits<T>::infinity(), T xmax = std::numeric_limits<T>::infinity())
{
    //now we attempt to find the root of f near x0 using a newton's method solver with bounds.
    T xp = x0;
    T fp = f(xp);

    for(size_t i = 0; i < niters; ++i)
    {
        T derf = df(xp);
        T x = xp - fp / derf;

        //if the newton step has taken us out of range we use a bisection step instead - moving towards the direction the newton step has sent us
        if(x < xmin)
        {
            x = (xp + xmin)/2.0;
        }
        if(x > xmax)
        {
            x = (xp + xmax)/2.0;
        }

        T _f = f(x);
    
        if(_f > 0)
        {
            xmax = x;
        }
        else if(_f < 0)
        {
            xmin = x;
        }

        T df2 = std::abs(_f);
        T dx = std::abs(xp - x);

        fp = _f;
        xp = x;

        if(dx < xtol || df2 < ftol)
        {
            return x;
        }
    }
    return -1;
    std::cerr << "max_iter exceeded: failed to converge roots." << std::endl;
}
}


#endif  //UTILS_ROOT_FINDER_HPP_//

