#ifndef UTILS_QUAD_ADAPTIVE_INTEGRATE_HPP_
#define UTILS_QUAD_ADAPTIVE_INTEGRATE_HPP_

#include <stdexcept>
#include <cstdint>

#include "levin/levin_collocation.hpp"
#include "gaussian_quadrature/gauss_legendre_quadrature.hpp"
#include "gaussian_quadrature/gauss_jacobi_quadrature.hpp"

#include <common/exception_handling.hpp>

#include <linalg/utils/linalg_utils.hpp>

namespace utils
{
namespace quad
{

template <typename T, typename Integ, typename F, typename SF, typename RT = typename linalg::get_real_type<T>::type>
T adaptive_integrate_internal(F&& f, const Integ& integ, T remainder, T r, RT a, RT b, bool use_abs, RT int_tol, bool rel, RT relerror, size_t max_iterations, bool print_out, SF&& subdivision)
{
    RT c = subdivision(a, b);
    T s, t;
    CALL_AND_HANDLE(s = integ(f, a, c), "Failed to evaluate integral.");
    CALL_AND_HANDLE(t = integ(f, c, b), "Failed to evaluate integral.");


    if(std::abs(s) == std::numeric_limits<RT>::infinity() || std::abs(t) == std::numeric_limits<RT>::infinity()){return r;}
    if(print_out)
    {
        std::cerr << "gauss: " << max_iterations << " " << a << " " << b << " " << c << " " << r << " " << s << " " << t << " " << std::abs(r - (s+t)) << std::endl; 
    }
    bool err_tol_satisfied = false;
    if(use_abs){err_tol_satisfied = ((std::abs(r - (s+t)) < int_tol) ? true : err_tol_satisfied);}
    if(rel){err_tol_satisfied = ((std::abs(r - (s+t))/std::abs(remainder+s+t) < relerror) ? true : err_tol_satisfied);}
    if(err_tol_satisfied){return s+t;}
    else
    {
        if(max_iterations == 0)
        {
            throw std::runtime_error("Adaptive_integrate failed.");
        }
        try{return adaptive_integrate_internal(std::forward<F>(f), integ, t+remainder, s, a, c, use_abs, int_tol, rel, relerror, max_iterations-1, print_out, std::forward<SF>(subdivision)) 
                 + adaptive_integrate_internal(std::forward<F>(f), integ, s+remainder, t, c, b, use_abs, int_tol, rel, relerror, max_iterations-1, print_out, std::forward<SF>(subdivision));}
        catch(const std::exception& ex){throw;}
    }
}

template <typename T, typename Integ, typename F, typename SF, typename RT = typename linalg::get_real_type<T>::type>
T adaptive_integrate_subdivision(F&& f, const Integ& integ, RT a, RT b, SF&& subdivision, bool use_abs = true, RT int_tol = 1e-10, bool rel = false, RT relerror =  1e-12, T remainder = 0.0, size_t max_iterations = 100, bool print_out = false)
{
    RT c = subdivision(a, b);
    T r, s, t;
    CALL_AND_HANDLE(r = integ(f, a, b), "Failed to evaluate integral.");
    CALL_AND_HANDLE(s = integ(f, a, c), "Failed to evaluate integral.");
    CALL_AND_HANDLE(t = integ(f, c, b), "Failed to evaluate integral.");
    if(print_out)
    {
        std::cerr << "gauss: " << a << " " << b << " " << c << " " << r << " " << s << " " << t << " " << std::abs(r - (s+t)) << std::endl;
    }

    if(std::abs(s) == std::numeric_limits<RT>::infinity() || std::abs(t) == std::numeric_limits<RT>::infinity()){return r;}

    bool err_tol_satisfied = false;
    if(use_abs){err_tol_satisfied = ((std::abs(r - (s+t)) < int_tol) ? true : err_tol_satisfied);}
    if(rel){err_tol_satisfied = ((std::abs(r - (s+t))/std::abs(remainder+s+t) < relerror) ? true : err_tol_satisfied);}
    if(err_tol_satisfied){return s+t;}
    else
    {
        if(max_iterations == 0)
        {
            throw std::runtime_error("Adaptive_integrate failed.");
        }
        try{return adaptive_integrate_internal(std::forward<F>(f), integ, t+remainder, s, a, c, use_abs, int_tol, rel, relerror, max_iterations-1, print_out, std::forward<SF>(subdivision)) 
                 + adaptive_integrate_internal(std::forward<F>(f), integ, s+remainder, t, c, b, use_abs, int_tol, rel, relerror, max_iterations-1, print_out, std::forward<SF>(subdivision));}
        catch(const std::exception& ex){throw;}
    }
}

template <typename T, typename Integ, typename F, typename RT = typename linalg::get_real_type<T>::type>
T adaptive_integrate(F&& f, const Integ& integ, RT a, RT b, bool use_abs = true, RT int_tol = 1e-10, bool rel = false, RT relerror =  1e-12, T remainder = 0.0, size_t max_iterations = 100, bool print_out = false)
{
    try{return adaptive_integrate_subdivision(std::forward<F>(f), integ, a, b, [](RT _a, RT _b){return (_a+_b)/2.0;}, use_abs, int_tol, rel, relerror, remainder, max_iterations, print_out);}
    catch(const std::exception& ex){throw;}
}




template <template <typename> class levin_impl, typename T>
class adaptive_levin
{
public:
    static_assert(std::is_base_of<levin_base<levin_impl<T>>, levin_impl<T>>::value, "Invalid type to adaptive_levin.");
    using value_type = typename levin_impl<T>::value_type;
    using result_type = typename levin_impl<T>::result_type;
    using real_type = typename levin_impl<T>::real_type;
    using lev_type = levin_impl<T>;
public:

    template <typename F, typename Integ>
    static result_type integrate(F&& f, lev_type& integ, const Integ& fallback, real_type x, real_type a, real_type b, real_type int_tol = 1e-14, result_type remainder = 0.0, bool rel = false, real_type relerror =  1e-12, bool print_out = false, size_t max_iterations = 500)
    {
        try{return integrate_subdivision(std::forward<F>(f), integ, fallback, x, a, b, [](real_type _a, real_type _b){return (_a+_b)/2.0;}, int_tol, remainder, rel, relerror, print_out, max_iterations);}
        catch(const std::exception& ex){throw;}
    }


    template <typename F, typename SF, typename Integ>
    static result_type integrate_subdivision(F&& f, lev_type& integ, const Integ& fallback, real_type x, real_type a, real_type b, SF&& subdivision, real_type int_tol = 1e-14, result_type remainder = 0.0, bool rel = false, real_type relerror =  1e-12, bool print_out = false, size_t max_iterations = 500)
    {
        if(b < a){real_type c = b;  b = a; a = c;}
        result_type r,s,t;
        if(a == b){return 0.0;}

        auto fcyl = [x, &f, &integ](real_type _r){return f(_r)*integ.oscillatory_function(x*_r);};

        //if the integration region is over a region comparable to the oscillation frequency then just use the fallback routine
        real_type c = subdivision(a, b);//(a+b)/2.0;
        if( (b-a) * x < 10*M_PI)
        {
            CALL_AND_HANDLE(r = fallback(fcyl, a, b), "Failed to evaluate integral.");  
            CALL_AND_HANDLE(s = fallback(fcyl, a, c), "Failed to evaluate integral.");  
            CALL_AND_HANDLE(t = fallback(fcyl, c, b), "Failed to evaluate integral.");  
        }
        else
        {
            bool failed = false;
            try{r = integ( f, x, a, b);}
            catch(const std::exception& ex){failed = true;}
            if(std::isnan(std::abs(r)) || failed){CALL_AND_HANDLE(r = fallback(fcyl, a, b), "Failed to evaluate integral."); std::cerr << "fallback" << std::endl;}
        
            failed = false;
            try{s = integ(f, x, a, c);}
            catch(const std::exception& ex){failed = true;}
            if(std::isnan(std::abs(s)) || failed){CALL_AND_HANDLE(s = fallback(fcyl, a, c), "Failed to evaluate integral."); std::cerr << "fallback" << std::endl;}

            failed = false;
            try{t = integ(f, x, c, b);}
            catch(const std::exception& ex){failed = true;}
            if(std::isnan(std::abs(t)) || failed){CALL_AND_HANDLE(t = fallback(fcyl, c, b), "Failed to evaluate integral."); std::cerr << "fallback" << std::endl;}
        }


        real_type scale = rel ? std::abs(remainder + s+t) + int_tol : 1.0;
        scale = scale < relerror/int_tol ? relerror/int_tol : scale;
        if(print_out)
        {
            std::cerr << "lev: " << a << " " << b << " " << c << " " << r << " " << s << " " << t << " " << std::abs(r - (s+t)) << std::endl;
        }
        if(std::abs(r - (s+t))/scale < int_tol){return s+t;}
        else
        {
            if(max_iterations == 0)
            {
                throw std::runtime_error("Adaptive_integrate_levin failed.");
            }
            try{return integrate_internal(std::forward<F>(f), integ, fallback, remainder + t, s, x, a, c, int_tol, rel, relerror, print_out, max_iterations - 1, std::forward<SF>(subdivision)) 
                     + integrate_internal(std::forward<F>(f), integ, fallback, remainder + s, t, x, c, b, int_tol, rel, relerror, print_out, max_iterations - 1, std::forward<SF>(subdivision));}
            catch(const std::exception& ex){throw;}
        }
    }

protected:
    template <typename F, typename SF, typename Integ>
    static result_type integrate_internal(F&& f, lev_type& integ, const Integ& fallback, result_type remainder, result_type r, real_type x, real_type a, real_type b, real_type int_tol, bool rel, real_type relerror, bool print_out, size_t max_iterations, SF&& subdivision)
    {
        result_type s,t;
        auto fcyl = [x, &f, &integ](real_type _r){return f(_r)*integ.oscillatory_function(x*_r);};
        if(a == b){return 0.0;}

        //if the integration region is over a region comparable to the oscillation frequency then just use the fallback routine
        real_type c = subdivision(a, b);//(a+b)/2.0;
        if( (b-a) * x < 10*M_PI)
        {
            CALL_AND_HANDLE(r = fallback(fcyl, a, b), "Failed to evaluate integral.");  
            CALL_AND_HANDLE(s = fallback(fcyl, a, c), "Failed to evaluate integral.");  
            CALL_AND_HANDLE(t = fallback(fcyl, c, b), "Failed to evaluate integral.");  
        }
        else
        {
            bool failed = false;
            try{r = integ( f, x, a, b);}
            catch(const std::exception& ex){failed = true;}
            if(std::isnan(std::abs(r)) || failed){CALL_AND_HANDLE(r = fallback(fcyl, a, b), "Failed to evaluate integral."); std::cerr << "fallback" << std::endl;}
        
            failed = false;
            try{s = integ(f, x, a, c);}
            catch(const std::exception& ex){failed = true;}
            if(std::isnan(std::abs(s)) || failed){CALL_AND_HANDLE(s = fallback(fcyl, a, c), "Failed to evaluate integral."); std::cerr << "fallback" << std::endl;}

            failed = false;
            try{t = integ(f, x, c, b);}
            catch(const std::exception& ex){failed = true;}
            if(std::isnan(std::abs(t)) || failed){CALL_AND_HANDLE(t = fallback(fcyl, c, b), "Failed to evaluate integral."); std::cerr << "fallback" << std::endl;}
        }


        if(print_out)
        {
            std::cerr << "lev: " << max_iterations << " " << a << " " << b << " " << c << " " << r << " " << s << " " << t << " " << std::abs(r - (s+t)) << std::endl;
        }
        real_type scale = rel ? std::abs(remainder + s+t) + int_tol : 1.0;
        scale = scale < relerror/int_tol ? relerror/int_tol : scale;
        if(std::abs(r - (s+t))/scale < int_tol){return s+t;}
        else
        {
            if(max_iterations == 0)
            {
                throw std::runtime_error("Adaptive_integrate_levin failed.");
            }
            try{return integrate_internal(std::forward<F>(f), integ, fallback, remainder + t, s, x, a, c, int_tol, rel, relerror, print_out, max_iterations - 1, std::forward<SF>(subdivision)) 
                     + integrate_internal(std::forward<F>(f), integ, fallback, remainder + s, t, x, c, b, int_tol, rel, relerror, print_out, max_iterations - 1, std::forward<SF>(subdivision));}
            catch(const std::exception& ex){throw;}
        }
    }


};

//need to implement a class that wraps the adaptive integrate functions
template <typename T> 
class adaptive_fourier_integrals
{
public:
    using real_type = T;
    using result_type = typename levin_cosine<T>::result_type;
    using fresult_type = typename levin_fourier<T>::result_type;
public:
    adaptive_fourier_integrals(size_t Nlev = 10, size_t Ngauss = 100) : m_levf(Nlev), m_levc(Nlev), m_levs(Nlev), m_gauss(Ngauss) {}
    adaptive_fourier_integrals(const adaptive_fourier_integrals& o) = default;
    adaptive_fourier_integrals(adaptive_fourier_integrals&& o) = default;
    adaptive_fourier_integrals& operator=(const adaptive_fourier_integrals& o) = default;
    adaptive_fourier_integrals& operator=(adaptive_fourier_integrals&& o) = default;

    //functions for evaluating non oscillatory integrals
    template <typename F>
    result_type integrate(F&& f, real_type a, real_type b, real_type iguess = 1e3, real_type int_tol = 1e-14, result_type remainder = 0.0, bool rel = false, real_type relerror =  1e-12, bool print_out = false, size_t max_iterations = 500)
    {
        if(b == std::numeric_limits<real_type>::infinity())
        {
            CALL_AND_RETHROW(return upper_real_integral(std::forward<F>(f), a, iguess, int_tol, remainder, rel, relerror, print_out, max_iterations));
        }
        else
        {
            CALL_AND_RETHROW(return definite_integral(std::forward<F>(f), a, b, int_tol, remainder, rel, relerror, print_out, max_iterations));
        }
    }    
    
    //functions for evaluating the fouriet type integrals
    template <typename F>
    fresult_type fourier(F&& f, real_type x, real_type a, real_type b, real_type iguess = 1e3, real_type int_tol = 1e-14, fresult_type remainder = 0.0, bool rel = false, real_type relerror =  1e-12, bool print_out = false, size_t max_iterations = 500)
    {
        if(b == std::numeric_limits<real_type>::infinity())
        {
            if(std::abs(x) < 1e-5)
            {
                CALL_AND_RETHROW(return upper_real_integral([&](const real_type& w){return f(w)*fresult_type(std::cos(x*w), std::sin(x*w));}, a, iguess, int_tol, remainder, rel, relerror, print_out, max_iterations));
            }
            else
            {
                CALL_AND_RETHROW(return upper_real_integral(std::forward<F>(f), m_levf, x, a, iguess, int_tol, remainder, rel, relerror, print_out, max_iterations));
            }
        }
        else
        {
            if(std::abs(x)/(b-a) < 10)
            {
                CALL_AND_RETHROW(return definite_integral([&](const real_type& w){return f(w)*fresult_type(std::cos(x*w), std::sin(x*w));}, a, b, int_tol, remainder, rel, relerror, print_out, max_iterations));
            }
            else
            {
                CALL_AND_RETHROW(return definite_integral(std::forward<F>(f), m_levf, x, a, b, int_tol, remainder, rel, relerror, print_out, max_iterations));
            }
        }
    }

    template <typename F>
    result_type cosine(F&& f, real_type x, real_type a, real_type b, real_type iguess = 1e3, real_type int_tol = 1e-14, result_type remainder = 0.0, bool rel = false, real_type relerror =  1e-12, bool print_out = false, size_t max_iterations = 500)
    {
        if(b == std::numeric_limits<real_type>::infinity())
        {
            if(std::abs(x) < 1e-5)
            {
                CALL_AND_RETHROW(return upper_real_integral([&](const real_type& w){return f(w)*std::cos(x*w);}, a, iguess, int_tol, remainder, rel, relerror, print_out, max_iterations));
            }
            else
            {
                CALL_AND_RETHROW(return upper_real_integral(std::forward<F>(f), m_levc, x, a, iguess, int_tol, remainder, rel, relerror, print_out, max_iterations));
            }
        }
        else
        {
            if(std::abs(x)/(b-a) < 10)
            {
                CALL_AND_RETHROW(return definite_integral([&](const real_type& w){return f(w)*std::cos(x*w);}, a, b, int_tol, remainder, rel, relerror, print_out, max_iterations));
            }
            else
            {
                CALL_AND_RETHROW(return definite_integral(std::forward<F>(f), m_levc, x, a, b, int_tol, remainder, rel, relerror, print_out, max_iterations));
            }
        }
    }
    template <typename F>
    result_type sine(F&& f, real_type x, real_type a, real_type b, real_type iguess = 1e3, real_type int_tol = 1e-14, result_type remainder = 0.0, bool rel = false, real_type relerror =  1e-12, bool print_out = false, size_t max_iterations = 500)
    {
        if(b == std::numeric_limits<real_type>::infinity())
        {
            if(std::abs(x) < 1e-5)
            {
                CALL_AND_RETHROW(return upper_real_integral([&](const real_type& w){return f(w)*std::sin(x*w);}, a, iguess, int_tol, remainder, rel, relerror, print_out, max_iterations));
            }
            else
            {
                CALL_AND_RETHROW(return upper_real_integral(std::forward<F>(f), m_levs, x, a, iguess, int_tol, remainder, rel, relerror, print_out, max_iterations));
            }
        }
        else
        {
            if(std::abs(x)/(b-a) < 10)
            {
                CALL_AND_RETHROW(return definite_integral([&](const real_type& w){return f(w)*std::sin(x*w);}, a, b, int_tol, remainder, rel, relerror, print_out, max_iterations));
            }
            else
            {
                CALL_AND_RETHROW(return definite_integral(std::forward<F>(f), m_levs, x, a, b, int_tol, remainder, rel, relerror, print_out, max_iterations));
            }
        }
    }


    const gauss::legendre<T>& quad() const{return m_gauss;}
protected:
    template <typename F, typename U>
    U definite_integral(F&& f,real_type a, real_type b, real_type int_tol = 1e-14, U remainder = 0.0, bool rel = false, real_type relerror =  1e-12, bool print_out = false, size_t max_iterations = 500)
    {
        try
        {
            return adaptive_integrate(std::forward<F>(f), m_gauss, a, b, true, int_tol, rel, relerror, remainder, max_iterations, print_out);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate integral.");
        }
    }

    template <typename F, typename U>
    U upper_real_integral(F&& f,real_type a, real_type b, real_type int_tol = 1e-14, U remainder = 0.0, bool rel = false, real_type relerror =  1e-12, bool print_out = false, size_t max_iterations = 500)
    {
        try
        {
            U res = remainder;
            real_type ubound = b+a;
            real_type lbound = a;
            real_type diff = b;

            bool continue_iter = true;

            while(continue_iter)
            {
                U local = adaptive_integrate(std::forward<F>(f), m_gauss, lbound, ubound, true, int_tol, rel, relerror, res, max_iterations, print_out);
        
                if(std::abs(local) < relerror*std::abs(res) || std::abs(local) < 1e-15){continue_iter = false;}
                res += local;
                diff*=2;
                lbound = ubound;
                ubound += diff;
                  
            }
            return res;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate integral over entire upper real line.");
        }
    }



    template <typename F, typename U, template <typename > class Lev>
    U definite_integral(F&& f, Lev<T>& lev, real_type x, real_type a, real_type b, real_type int_tol = 1e-14, U remainder = 0.0, bool rel = false, real_type relerror =  1e-12, bool print_out = false, size_t max_iterations = 500)
    {
        try
        {
            return adaptive_levin<Lev, T>::integrate(std::forward<F>(f), lev, m_gauss, x, a, b, int_tol, remainder, rel, relerror, print_out, max_iterations);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate integral.");
        }
    }

    template <typename F, typename U, template <typename > class Lev>
    U upper_real_integral(F&& f, Lev<T>& lev, real_type x, real_type a, real_type b, real_type int_tol = 1e-14, U remainder = 0.0, bool rel = false, real_type relerror =  1e-12, bool print_out = false, size_t max_iterations = 500)
    {
        try
        {
            U res = remainder;
            real_type ubound = b+a;
            real_type lbound = a;
            real_type diff = b;

            bool continue_iter = true;

            while(continue_iter)
            {
                U local = adaptive_levin<Lev, T>::integrate(std::forward<F>(f), lev, m_gauss, x, lbound, ubound, int_tol, res, rel, relerror, print_out, max_iterations);
    
                if(std::abs(local) < relerror*std::abs(res) || std::abs(local) < 1e-15){continue_iter = false;}
                res += local;
                diff*=2;
                lbound = ubound;
                ubound += diff;
                  
            }
            return res;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to evaluate integral over entire upper real line.");
        }
    }
 
    
protected:
    levin_fourier<T> m_levf;
    levin_cosine<T> m_levc;
    levin_sine<T> m_levs;
    gauss::legendre<T> m_gauss;
};

}   //namespace quad
}

#endif //UTILS_QUAD_ADAPTIVE_INTEGRATE_HPP//

