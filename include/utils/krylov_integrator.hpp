#ifndef UTILS_KRYLOV_SUBSPACE_INTEGRATOR_HPP_
#define UTILS_KRYLOV_SUBSPACE_INTEGRATOR_HPP_

#include <limits>
#include <utility>
#include <algorithm>

#include <common/tmp_funcs.hpp>
#include <common/exception_handling.hpp>

#include <linalg/linalg.hpp>
#include <linalg/decompositions/sparse/arnoldi_iteration.hpp>
#include <linalg/decompositions/eigensolvers/eigensolver.hpp>

namespace utils
{

template <typename T, typename backend = linalg::blas_backend>
class krylov_integrator;

template <typename T, typename backend>
class krylov_integrator<complex<T>, backend>
{
public:
    using value_type = complex<T>;
    using real_type = T;
    using backend_type = backend;
    using size_type = typename backend_type::size_type;

protected:
    linalg::arnoldi_iteration<value_type, backend> m_arnoldi;

    //we construct the eigenvalues of the upper hessenberg matrix obtained from arnoldi on the gpu
    linalg::eigensolver<linalg::upper_hessenberg_matrix<value_type, linalg::blas_backend> > m_eigensolver;
    linalg::diagonal_matrix<value_type, linalg::blas_backend> m_vals;
    linalg::diagonal_matrix<value_type, linalg::blas_backend> m_expmat;

    std::vector<std::pair<value_type, size_type>> m_sorted_vals;
    std::vector<std::pair<value_type, size_type>> m_sorted_vals2;

    linalg::vector<value_type, linalg::blas_backend> m_e1;
    linalg::matrix<value_type, linalg::blas_backend> m_rvecs;
    linalg::matrix<value_type, linalg::blas_backend> m_rvecsd;
    linalg::matrix<value_type, linalg::blas_backend> m_lvecs;
    linalg::vector<value_type, backend> m_vkry;
    linalg::vector<value_type, backend> m_vkry_prev;
    linalg::vector<value_type, backend> m_tempr;
        
    linalg::vector<value_type, linalg::blas_backend> m_temp1;
    linalg::vector<value_type, linalg::blas_backend> m_temp2;

    size_type m_krylov_dim;
    size_type m_cur_order;
    size_type m_istride;
    real_type m_eps;
    real_type m_gamma;
    real_type m_delta;

public:
    krylov_integrator() : m_arnoldi(), m_krylov_dim(0), m_cur_order(0), m_istride(1), m_eps(std::numeric_limits<real_type>::epsilon()*1e3), m_gamma(0.8), m_delta(0.9) {}
    krylov_integrator(size_type krylov_dim, size_type dim, real_type eps = std::numeric_limits<real_type>::epsilon()*1e3) : m_arnoldi(), m_krylov_dim(krylov_dim), m_cur_order(0), m_istride(1), m_eps(eps), m_gamma(0.8), m_delta(0.9) {CALL_AND_HANDLE(resize(krylov_dim, dim), "Failed to construct krylov subspace integrator.");}
    krylov_integrator(const krylov_integrator& o) = default;
    krylov_integrator(krylov_integrator&& o) = default;

    krylov_integrator& operator=(const krylov_integrator& o) = default;
    krylov_integrator& operator=(krylov_integrator&& o) = default;

    void resize(size_type krylov_dim, size_type dim)
    {
        try
        {
            m_krylov_dim = krylov_dim;
            CALL_AND_HANDLE(m_arnoldi.resize(krylov_dim, dim), "Failed to resize arnoldi iteration engine.");
            CALL_AND_HANDLE(m_eigensolver.resize(krylov_dim, false), "Failed to resize upper_hessenberg_eigensolver.");
            CALL_AND_HANDLE(m_e1.resize(krylov_dim), "Failed to resize the e1 vector.");

            CALL_AND_HANDLE(m_vals.resize(krylov_dim, krylov_dim), "Failed to resize the temp1 vector.");
            CALL_AND_HANDLE(m_expmat.resize(krylov_dim, krylov_dim), "Failed to resize the temp1 vector.");
            CALL_AND_HANDLE(m_sorted_vals.resize(krylov_dim), "Failed to resize the temp1 vector.");
            CALL_AND_HANDLE(m_sorted_vals2.resize(krylov_dim), "Failed to resize the temp1 vector.");

            CALL_AND_HANDLE(m_rvecs.resize(krylov_dim, krylov_dim), "Failed to resize the temp1 vector.");
            CALL_AND_HANDLE(m_rvecsd.resize(krylov_dim, krylov_dim), "Failed to resize the temp1 vector.");
            CALL_AND_HANDLE(m_lvecs.resize(krylov_dim, krylov_dim), "Failed to resize the temp1 vector.");
            CALL_AND_HANDLE(m_temp1.resize(krylov_dim), "Failed to resize the temp1 vector.");
            CALL_AND_HANDLE(m_temp2.resize(krylov_dim), "Failed to resize the temp2 vector.");
            CALL_AND_HANDLE(m_tempr.resize(dim), "Failed to resize the temp2 vector.");
            m_e1.fill_zeros();
            m_e1(0)=1;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize krylov integrator.");
        }
    }

    void clear()
    {
        try
        {
            m_krylov_dim = 0;
            CALL_AND_HANDLE(m_arnoldi.clear(), "Failed to resize arnoldi iteration engine.");
            CALL_AND_HANDLE(m_eigensolver.clear(), "Failed to resize upper_hessenberg_eigensolver.");
            CALL_AND_HANDLE(m_e1.clear(), "Failed to resize the e1 vector.");
            CALL_AND_HANDLE(m_temp1.clear(), "Failed to resize the temp1 vector.");
            CALL_AND_HANDLE(m_temp2.clear(), "Failed to resize the temp2 vector.");
            CALL_AND_HANDLE(m_tempr.clear(), "Failed to resize the temp2 vector.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize krylov integrator.");
        }
    }

    template <typename vec_type, typename ... Args>
    typename std::enable_if<linalg::is_same_backend<vec_type, linalg::vector<value_type, backend_type>>::value, size_type>::type 
    operator()(vec_type& x, real_type dt, value_type coeff, Args&& ... args)
    {
        try
        {
            size_type krylov_dim = m_krylov_dim < x.size() ? m_krylov_dim : x.size();
            ASSERT(m_istride <= krylov_dim, "The stride for determining error estimates is too large.");

            //construct the krylov subspace and store the final matrix element required for computing error estimates and matrix exponentials
            CALL_AND_HANDLE(m_tempr.resize(x.size()), "Failed to resize the working buffer.");
            CALL_AND_HANDLE(m_arnoldi.resize(krylov_dim, x.size()), "Failed to resize krylov subspace.");

            if(dt == real_type(0)){return 0.0;}
            
            real_type dt_completed = 0;
            real_type dt_trial_next = 0;
            real_type dt_trial = dt;

            size_type nevals = 0;
            real_type err(0);
            size_type count = 0;
            while(dt_completed/dt < 1)
            {
                real_type scale_factor = dt_trial;
                bool nan_encountered;
                real_type err_res;
                determine_order(x, dt_trial, coeff, scale_factor, nan_encountered, err_res, std::forward<Args>(args)...);
                nevals += m_cur_order;

                if(!nan_encountered)
                {
                    err = err_res/std::abs(dt_trial);

                    ASSERT(m_cur_order >= 1, "The iend value obtained is not allowed.");
                    
                    bool error_tol_satisfied = err < m_delta*m_eps;
                    while(!error_tol_satisfied)
                    {
                        dt_trial_next = m_gamma*std::pow(m_eps/err, real_type(1.0/m_cur_order))*(dt_trial);
                        if(std::abs(dt_trial_next) < 0.1*std::abs(dt_trial)){dt_trial_next = 0.1*dt_trial;}
                        if(std::abs(dt_trial_next) > 10*std::abs(dt_trial)){dt_trial_next = 10*dt_trial;}
                        dt_trial = dt_trial_next;

                        CALL_AND_HANDLE(err_res = local_error_estimate(std::abs(dt_trial), coeff/scale_factor), "Failed to compute local error estimate."); 
                        err = err_res/std::abs(dt_trial);
                        error_tol_satisfied = err < m_delta*m_eps;
                    }

                    CALL_AND_HANDLE(integrate_step(x, dt_trial, coeff/scale_factor), "Failed to perform the integration step.");
                    dt_completed += dt_trial;   

                    dt_trial_next = m_gamma*std::pow(m_eps/err_res, real_type(1.0/m_cur_order))*(dt_trial);
                    if(std::abs(dt_trial_next) < 0.1*std::abs(dt_trial)){dt_trial_next = 0.1*dt_trial;}
                    if(std::abs(dt_trial_next) > 10*std::abs(dt_trial)){dt_trial_next = 10*dt_trial;}
                    dt_trial = dt_trial_next;

                    if(std::abs(dt_trial + dt_completed) > std::abs(dt)){dt_trial = dt-dt_completed;}
                }
                else
                {
                    auto H = m_arnoldi.H();
                    if(std::isnan(abs(H(0,0)))){RAISE_NUMERIC("Nan encountered at first element of krylov subspace iteration.");}
                    scale_factor /= 10;
                }
                ++count;
            }
            return nevals;
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("performing krylov subspace integration");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to perform krylov subspace integration.");
        }
    }

    const size_type& stride() const {return m_istride;}
    size_type& stride() {return m_istride;}

    const real_type& error_tolerance() const {return m_eps;}
    real_type& error_tolerance() {return m_eps;}

    const size_type& krylov_dim() const{return m_krylov_dim;}

    //accessors for the safety parameters
    const real_type& gamma() const {return m_gamma;}
    real_type& gamma() {return m_gamma;}
    const real_type& delta() const {return m_delta;}
    real_type& delta() {return m_delta;}

    const size_type& current_order() const{return m_cur_order;}

    //need to output the eigenvectors correctly ordered
    template <typename vec_type, typename ... Args>
    typename std::enable_if<linalg::is_same_backend<vec_type, linalg::vector<value_type, backend_type>>::value, size_type>::type 
    eigs(vec_type& x, linalg::diagonal_matrix<value_type, backend_type>& vals, linalg::matrix<value_type, backend_type>& vecs, real_type scalefactor, size_type neigs, size_type maxdim, real_type errtol, Args&& ... args)
    {
        try
        {
            size_type krylov_dim = m_krylov_dim < x.size() ? m_krylov_dim : x.size();
            if(maxdim > krylov_dim){maxdim = krylov_dim;}
            ASSERT(m_istride <= krylov_dim, "The stride for determining error estimates is too large.");
            size_type order_start = neigs;

            ASSERT(neigs < m_krylov_dim, "Cannot compute more eigenvalues than dimension of object.");  
            if(neigs > x.size()){neigs = x.size();}

            //construct the krylov subspace and store the final matrix element required for computing error estimates and matrix exponentials
            CALL_AND_HANDLE(m_tempr.resize(x.size()), "Failed to resize the working buffer.");
            CALL_AND_HANDLE(m_arnoldi.resize(krylov_dim, x.size()), "Failed to resize krylov subspace.");
            CALL_AND_HANDLE(m_arnoldi.reset_zeros(), "Failed to reset arnoldi iteration.");
            CALL_AND_HANDLE(m_sorted_vals.resize(krylov_dim), "Failed to resize the working buffer.");
            CALL_AND_HANDLE(m_sorted_vals2.resize(neigs), "Failed to resize the eigenvalue storage array.");
        
            //compute the arnoldi iteration. 

            //now compute the eigenvalues in the arnoldi subspace
            CALL_AND_HANDLE(m_vals.resize(krylov_dim), "Failed to resize the working buffer.");
            CALL_AND_HANDLE(vecs.resize(neigs, x.size()), "Failed to resize the working buffer.");

            size_type istart = 0;
            size_type iend = maxdim;
            bool keep_running = true;
            bool first_call = true;
            for(iend = order_start; iend <= krylov_dim+m_istride && keep_running; iend+=m_istride)
            {
                if(iend > maxdim){iend = maxdim; keep_running = false;}   
                try
                {
                    bool ended_early = false;
                    CALL_AND_HANDLE(ended_early = m_arnoldi.partial_krylov_step(x, scalefactor, istart, iend, std::forward<Args>(args)...), "Failed to construct the krylov subspace using a arnoldi iteration");
                    auto H = m_arnoldi.H();      

                    m_eigensolver(H, m_vals, m_rvecs, m_lvecs, true);
                    
                    //sort the eigenvalues based on norm
                    {
                        m_sorted_vals.resize(m_vals.size());
                        for(size_type i = 0; i < m_vals.size(); ++i)
                        {
                            m_sorted_vals[i] = std::make_pair(m_vals[i], i);
                        }
                        std::sort(m_sorted_vals.begin(), m_sorted_vals.end(), 
                            [](const std::pair<value_type, size_type>& a, const std::pair<value_type, size_type>& b)
                            {
                                return abs(std::get<0>(a)) > abs(std::get<0>(b));
                            });
                    }

                    //now we check the error 
                    bool err_tol_satisfied = true;
                    if(first_call)
                    {
                        CALL_AND_HANDLE(m_sorted_vals2 = m_sorted_vals, "Failed to copy eigenvalue array.");
                        first_call = false;
                        err_tol_satisfied = false;
                    }
                    else
                    {
                        for(size_type i = 0; i < neigs; ++i)
                        {
                            if(abs((std::get<0>(m_sorted_vals[i]) - std::get<0>(m_sorted_vals2[i]))/std::get<0>(m_sorted_vals[i])) > errtol){err_tol_satisfied = false;}
                            m_sorted_vals2[i] = m_sorted_vals[i];
                        }
                    }
                    if(err_tol_satisfied || ended_early){keep_running = false;}
                } 
                catch(const std::exception& ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Error when attempting to compute eigenvalues.");
                }
            }
            //copy the eigenvalues to the device
            CALL_AND_HANDLE(sort_vecs_and_vals(vals, vecs, scalefactor), "Failed to sort the eigenvalues.");
            return iend;
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("performing krylov subspace integration");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to perform krylov subspace integration.");
        }
    }

protected:
    void sort_vecs_and_vals(linalg::diagonal_matrix<value_type, backend_type>& vals, linalg::matrix<value_type, backend_type>& vecs, real_type scalefactor)
    {
        try
        {
            //sort and scale eigenvalues
            CALL_AND_HANDLE(m_vals.resize(m_sorted_vals2.size()), "Failed to resize vals array.");
            for(size_type i = 0; i < m_sorted_vals2.size(); ++i)
            {
                m_vals[i] = std::get<0>(m_sorted_vals2[i]);
            }
            CALL_AND_HANDLE(vals = m_vals/scalefactor, "Failed to copy the eigenvalues.");

            //sort eigenvectors in krylov subspace
            CALL_AND_HANDLE(m_lvecs = linalg::adjoint(m_rvecs), "Failed to store rvecs in temporary.");
            for(size_type i = 0; i < m_sorted_vals2.size(); ++i)         
            {
                m_rvecs[i] = m_lvecs[std::get<1>(m_sorted_vals2[i])];
            }
            CALL_AND_HANDLE(m_rvecsd = m_rvecs, "Failed to copy rvecs to device.");

            //and transform to original space
            CALL_AND_HANDLE(vecs = m_rvecsd*m_arnoldi.Q(), "Failed to compute eigenvectors.");
            for(size_type i = 0; i < vecs.size(0); ++i)
            {
                real_type norm = std::sqrt( linalg::real(linalg::dot_product(linalg::conj(vecs[i]), vecs[i])));
                vecs[i] /= norm;
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to sort eigenvalues and eigenvectors.");
        }
    }


protected:
    template <typename vec_type, typename ... Args>
    void determine_order(vec_type& x, real_type dt, value_type coeff, real_type scale_factor, bool& nan_encountered,  real_type& err_res, Args&& ... args)
    {
        //now we determine the initial 
        size_type order_start = 2;
        real_type emin = -1e10;
        size_t imin = 0;
        
        size_type krylov_dim = m_krylov_dim < x.size() ? m_krylov_dim : x.size();
        CALL_AND_HANDLE(m_arnoldi.reset_zeros(), "Failed to reset arnoldi iteration.");
        size_type istart = 0;
        nan_encountered = false;
        size_type iend;
        std::vector<std::pair<real_type, size_type> > mdtres(m_krylov_dim);
        size_type counter = 0;
        bool keep_running = true;
        for(iend = order_start; iend < m_krylov_dim+m_istride; iend+=m_istride)
        {
            if(iend > krylov_dim){iend = krylov_dim;    keep_running = false;}   
            try
            {
                bool ended_early = false;
                CALL_AND_HANDLE(ended_early = m_arnoldi.partial_krylov_step(x, scale_factor, istart, iend, std::forward<Args>(args)...), "Failed to construct the krylov subspace using a arnoldi iteration");
                if(ended_early){keep_running = false;}
                auto H = m_arnoldi.H();      
                m_eigensolver(H, m_vals, m_rvecs, m_lvecs, true);
                err_res = local_error_estimate(std::abs(dt), coeff/scale_factor);
                //std::cerr << iend << " " << dt << " " << err_res << std::endl;
                if(ended_early){m_cur_order = m_arnoldi.current_krylov_dim(); return;}
                mdtres[counter] = std::make_pair(std::abs(std::pow(m_eps/(err_res/std::abs(dt)), real_type(1.0/m_arnoldi.current_krylov_dim()))*(std::abs(dt))), m_arnoldi.current_krylov_dim());
                ++counter;
            } 
            catch(const std::exception& ex)
            {
                nan_encountered = true;
                m_cur_order = 0;
                return ;
            }
        
            if(!nan_encountered)    
            {
                if(err_res/std::abs(dt) < m_delta*m_eps || keep_running == false)
                {
                    m_cur_order = iend; return;
                }
                if(emin < 0 || err_res < emin)
                {
                    emin = err_res;
                    imin = m_arnoldi.current_krylov_dim();
                }

                //if the current error is much larger than the previous smallest error then we stop attempting to adapt the order at this stage and attemp
                //to make the krylov subspace step with the previous smallest order
                if(emin > 0 && err_res > emin*1e5)
                {
                    m_cur_order = imin;
                    m_arnoldi.finalise_krylov_rep(m_cur_order);
                    auto H2 = m_arnoldi.H();      
                    CALL_AND_HANDLE(m_eigensolver(H2, m_vals, m_rvecs, m_lvecs, true), "Failed to diagonalise upper hessenberg matrix.");
                    CALL_AND_HANDLE(err_res = local_error_estimate(std::abs(dt), coeff/scale_factor), "Failed to compute local error estimate."); 
                    return;
                }
                istart = iend + 1;
            }
        }
        real_type maxdt = 0;    size_type order = 0;
        for(size_type i = 0; i < counter; ++i)
        {
            if(std::abs(std::get<0>(mdtres[i])) > maxdt){maxdt = std::abs(std::get<0>(mdtres[i]));  order = std::get<1>(mdtres[i]);}
        }
        m_cur_order = order;

        if(imin != iend)
        {
            m_arnoldi.finalise_krylov_rep(m_cur_order);
            auto H2 = m_arnoldi.H();      
            CALL_AND_HANDLE(m_eigensolver(H2, m_vals, m_rvecs, m_lvecs, true), "Failed to diagonalise upper hessenberg matrix.");
            CALL_AND_HANDLE(err_res = local_error_estimate(std::abs(dt), coeff/scale_factor), "Failed to compute local error estimate."); 
        }
    }
    

    template <typename vec_type>
    void integrate_step(vec_type& x, real_type dt, value_type coeff)
    {
        try
        {
            //resize the working arrays
            m_e1.resize(m_cur_order);
            m_temp1.resize(m_cur_order);
            m_temp2.resize(m_cur_order);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize working arrays.");
        }
        try
        {
            //now we compute the action of the exponential on our matrix
            //evaluate u = exp(dt*coeff*H)*e1
            m_expmat = elemental_exp(m_vals*dt*coeff);
            m_lvecs  = conj(m_lvecs);
            m_temp1 = trans(m_lvecs)*m_e1;
            m_temp2 = m_expmat*m_temp1;
            m_temp1 = m_arnoldi.beta()*m_rvecs*m_temp2;
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing krylov subspace representation of the propagator.");
        }
        catch(const std::exception& ex)
        {
            RAISE_EXCEPTION("Failed to compute krylov subspace representation of the propagator.");
        } 
        try
        {
            //now we transform this vector to the device 
            m_vkry = m_temp1;

            //and we construct our time evolved vector
            m_tempr = trans(m_arnoldi.Q())*m_vkry;
            x.set_buffer(m_tempr);
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("applying propagator.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to apply propagator.");
        }
    }

    value_type phi1(value_type x)
    {
        value_type expfactor = 1.0;
        
        //compute phi_1(tau E)
        if(abs(x) < 1e-3)
        {
            value_type term = 1.0;
            for(size_type j=1; j<50; ++j)
            {
                term *= (x/static_cast<real_type>(j+1));
                expfactor += term;
            }
        }
        else{expfactor = (exp(x)-1.0)/(x);}
        return expfactor;
    }

    real_type local_error_estimate(real_type dt, value_type coeff)
    {
        try
        {
            real_type err = 0;
            value_type accum = value_type(0)*0.0;
            value_type g = coeff*dt;

            for(size_type k=0; k<m_vals.shape(0); ++k)
            {
                value_type rekc = g*m_vals[k];
                accum += m_rvecs(m_vals.size(0)-1, k)*conj(m_lvecs(0, k))*phi1(rekc);
            }
            err = abs(accum)*abs(g*m_arnoldi.hk1k());
            //err = m_arnoldi.beta()*abs(accum)*abs(g*m_arnoldi.hk1k());

            real_type ret = 0.0;
            if(m_arnoldi.hk1k() != 0){ret = err;}
            if(ret/dt < 1e-30){ret = 1e-30*dt;}
            if(std::isnan(ret))
            {
                RAISE_NUMERIC("Nan encountered in calcaulation of local error estimate.");
            }
            return ret;
        }
        catch(const common::invalid_value& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_NUMERIC("computing local error estimate.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to compute local error estimate.");
        }
    }
};

}

#endif  //UTILS_KRYLOV_SUBSPACE_INTEGRATOR_HPP_

