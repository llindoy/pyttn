#ifndef UTILS_ARNOLDI_HPP
#define UTILS_ARNOLDI_HPP


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
class arnoldi;

enum eigenvalue_target
{
    largest_magnitude = 0,
    smallest_real = 1,
    largest_real
};

template <typename T, typename backend>
class arnoldi
{
public:
    using value_type = T;
    using real_type = typename linalg::get_real_type<T>::type;
    using complex_type = linalg::complex<real_type>;
    using backend_type = backend;
    using size_type = typename backend_type::size_type;

protected:
    linalg::arnoldi_iteration<value_type, backend> m_arnoldi;

    //we construct the eigenvalues of the upper hessenberg matrix obtained from arnoldi on the gpu
    linalg::eigensolver<linalg::upper_hessenberg_matrix<value_type, linalg::blas_backend> > m_eigensolver;
    linalg::diagonal_matrix<complex_type, linalg::blas_backend> m_vals;

    linalg::matrix<complex_type, linalg::blas_backend> m_rvecs;
    linalg::matrix<complex_type, backend> m_rvecsd;
    linalg::matrix<complex_type, linalg::blas_backend> m_lvecs;

    linalg::matrix<real_type> m_residues;
    linalg::vector<bool> m_converged;

    size_type m_krylov_dim;
    size_type m_istride;
    size_type m_max_iter;
    size_type m_niters;
    size_type m_neigs;
    real_type m_eps;
    real_type m_rel_eps;

    eigenvalue_target m_mode;

    bool m_verbose;
    bool m_invert_mode;
public:

    arnoldi() : m_arnoldi(), m_krylov_dim(0), m_istride(1), m_max_iter(10), m_neigs(0), m_eps(std::numeric_limits<real_type>::epsilon()*1e3), m_rel_eps(0.0), m_mode(eigenvalue_target::largest_magnitude), m_verbose(false), m_invert_mode(false){}
    arnoldi(size_type krylov_dim, size_type dim, real_type eps = std::numeric_limits<real_type>::epsilon()*1e3) : m_arnoldi(), m_krylov_dim(krylov_dim), m_istride(1), m_max_iter(10), m_neigs(0), m_eps(eps), m_rel_eps(0.0), m_mode(eigenvalue_target::largest_magnitude), m_verbose(false), m_invert_mode(false) {CALL_AND_HANDLE(resize(krylov_dim, dim), "Failed to construct krylov subspace integrator.");}
    arnoldi(const arnoldi& o) = default;
    arnoldi(arnoldi&& o) = default;

    arnoldi& operator=(const arnoldi& o) = default;
    arnoldi& operator=(arnoldi&& o) = default;

    void resize(size_type krylov_dim, size_type dim)
    {
        try
        {
            m_krylov_dim = krylov_dim;
            CALL_AND_HANDLE(m_arnoldi.resize(krylov_dim, dim), "Failed to resize arnoldi iteration engine.");
            CALL_AND_HANDLE(m_eigensolver.resize(krylov_dim, false), "Failed to resize upper_hessenberg_eigensolver.");

            CALL_AND_HANDLE(m_vals.resize(krylov_dim, krylov_dim), "Failed to resize the temp1 vector.");

            CALL_AND_HANDLE(m_rvecs.resize(krylov_dim, krylov_dim), "Failed to resize the temp1 vector.");
            CALL_AND_HANDLE(m_rvecsd.resize(1, krylov_dim), "Failed to resize the temp1 vector.");
            CALL_AND_HANDLE(m_lvecs.resize(krylov_dim, krylov_dim), "Failed to resize the temp1 vector.");
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
            m_niters = 0;
            m_krylov_dim = 0;
            CALL_AND_HANDLE(m_arnoldi.clear(), "Failed to resize arnoldi iteration engine.");
            CALL_AND_HANDLE(m_eigensolver.clear(), "Failed to resize upper_hessenberg_eigensolver.");
            m_mode = eigenvalue_target::largest_magnitude;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize krylov integrator.");
        }
    }

    const bool& invert_mode() const{return m_invert_mode;}
    bool& invert_mode(){return m_invert_mode;}

    const bool& verbose() const{return m_verbose;}
    bool& verbose(){return m_verbose;}

    const eigenvalue_target& mode() const{return m_mode;}
    eigenvalue_target& mode(){return m_mode;}
  
    const size_type& stride() const {return m_istride;}
    size_type& stride() {return m_istride;}

    const real_type& error_tolerance() const {return m_eps;}
    real_type& error_tolerance() {return m_eps;}

    const real_type& tol() const {return m_eps;}
    real_type& tol() {return m_eps;}

    const real_type& rel_tol() const {return m_rel_eps;}
    real_type& rel_tol() {return m_rel_eps;}

    const size_type& krylov_dim() const{return m_krylov_dim;}

    const linalg::matrix<real_type>& residues() const{return m_residues;}
    const real_type& residue(size_t i, size_t j) const{ASSERT(i < m_residues.size(0) && j < m_niters, "Index out of bounds."); return m_residues(i,j);}
    const size_type& niter() const{return m_niters;}
    const size_type& neigs() const{return m_neigs;}
    size_type& neigs(){return m_neigs;}

    size_type& max_iter() {return m_max_iter;}
    const size_type& max_iter() const{return m_max_iter;}


    template <typename vec_type, typename ... Args>
    typename std::enable_if<linalg::is_same_backend<vec_type, linalg::vector<value_type, backend_type>>::value, size_t>::type 
    operator()(vec_type& x, complex_type& E, Args&& ... args)
    {
        try
        {
            size_type neigs = 1;
            size_type size = x.size();
            
            if(m_neigs > 0)
            {
                neigs = std::min(neigs, m_neigs);
            }
            m_neigs = neigs;

            size_type krylov_dim = std::min(m_krylov_dim, size);
            size_type istride = std::min(m_istride, krylov_dim);

            ASSERT(neigs <= m_krylov_dim, "Cannot compute more eigenvalues than the dimension of the krylov subspace.");  
            if(neigs > size){neigs = size;}

            //construct the krylov subspace and store the final matrix element required for computing error estimates and matrix exponentials
            CALL_AND_HANDLE(m_arnoldi.resize(krylov_dim, size), "Failed to resize krylov subspace.");
            CALL_AND_HANDLE(m_arnoldi.reset_zeros(), "Failed to reset arnoldi iteration.");
        
            //compute the arnoldi iteration. 
            real_type scalefactor = 1.0;

            E = value_type(0);
            CALL_AND_HANDLE(m_residues.resize(neigs, m_max_iter), "Failed to resize residues array.");  m_residues.fill_zeros();
            CALL_AND_HANDLE(m_converged.resize(neigs), "Failed to resize converged array.");    m_converged.fill_value(false);

            //now compute the eigenvalues in the arnoldi subspace
            CALL_AND_HANDLE(m_vals.resize(krylov_dim), "Failed to resize the working buffer.");

            bool all_converged = true;
            size_type max_iters = 0;

            size_type iter = 0;

            size_type eigenvalue_index=0;
            bool do_restart = true;
            for(iter = 0; iter < m_max_iter && do_restart; ++iter)
            {
                size_type istart = 0;
                size_type iend = krylov_dim;
                bool keep_running = true;
                size_type iend_start = 2 < krylov_dim ? 2 : krylov_dim;

                for(iend = iend_start; iend < krylov_dim+istride && keep_running; iend+=istride)
                {
                    if(iend > krylov_dim){iend = krylov_dim; keep_running = false;}   
                    try
                    {
                        bool ended_early = false;

                        //perform partial arnoldi step 
                        CALL_AND_HANDLE(ended_early = m_arnoldi.partial_krylov_step(x, scalefactor, istart, iend, std::forward<Args>(args)...), "Failed to construct the krylov subspace using a arnoldi iteration");

                        if(iend > 1)
                        {
                            if(ended_early){keep_running = false;}
                            auto H = m_arnoldi.H();      

                            m_eigensolver(H, m_vals, m_rvecs, m_lvecs, false);
                            size_type n = get_eig_index();
                            eigenvalue_index = n;

                            m_residues(0, iter) = m_arnoldi.hk1k()*linalg::abs(m_rvecs(m_vals.size()-1, n));
                            if(m_verbose)
                            {
                                std::cerr << 0 << " " << iter << " " << iend << " " << m_residues(0, iter) << " " << (m_invert_mode ? 1.0/m_vals(n, n) : m_vals(n, n)) << std::endl;
                            }
                            if(m_residues(0, iter) < m_eps || m_residues(0, iter) < m_rel_eps*linalg::abs((m_invert_mode ? 1.0/m_vals(n, n) : m_vals(n, n))))
                            {
                                do_restart = false;
                                keep_running=false;
                                m_converged(0) = true;
                            }
                        }

                        istart = iend + 1;
                        if(istart > krylov_dim){keep_running = false;}
                    } 
                    catch(const std::exception& ex)
                    {
                        std::cerr << ex.what() << std::endl;
                        RAISE_EXCEPTION("Error when attempting to compute eigenvalues.");
                    }
                }

                get_vecs_and_vals(E, x, eigenvalue_index);
                if(max_iters < iter)
                {
                    max_iters = iter;
                }
            }
            if(!m_converged(0)){all_converged = false;}
            m_niters = max_iters+1;

            return all_converged;
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

    template <typename vals_type, typename vecs_type,  typename ... Args>
    typename std::enable_if<linalg::is_same_backend<vals_type, linalg::vector<value_type, backend_type>>::value && linalg::is_same_backend<vecs_type, linalg::vector<value_type, backend_type>>::value, size_t>::type 
    operator()(vecs_type& x, vals_type& E, Args&& ... args)
    {
        try
        {
            size_type neigs = x.shape(0);
            size_type size =  x.size()/neigs;
            
            if(m_neigs > 0)
            {
                neigs = std::min(neigs, m_neigs);
            }
            m_neigs = neigs;

            size_type krylov_dim = std::min(m_krylov_dim, size);
            size_type istride = std::min(m_istride, krylov_dim);

            ASSERT(neigs <= m_krylov_dim, "Cannot compute more eigenvalues than the dimension of the krylov subspace.");  
            if(neigs > size){neigs = size;}
            if(E.size(0) != neigs){E.resize(neigs);}

            //construct the krylov subspace and store the final matrix element required for computing error estimates and matrix exponentials
            CALL_AND_HANDLE(m_arnoldi.resize(krylov_dim, size), "Failed to resize krylov subspace.");
            CALL_AND_HANDLE(m_arnoldi.reset_zeros(), "Failed to reset arnoldi iteration.");
        
            //compute the arnoldi iteration. 
            real_type scalefactor = 1.0;

            CALL_AND_HANDLE(m_residues.resize(krylov_dim, m_max_iter), "Failed to resize residues array.");  m_residues.fill_zeros();

            //now compute the eigenvalues in the arnoldi subspace
            CALL_AND_HANDLE(m_vals.resize(krylov_dim), "Failed to resize the working buffer.");

            size_type max_iters = 0;
            size_type iter = 0;

            size_type eigenvalues_evaluated = 0;
            std::vector<size_type> eigorder(krylov_dim);
            std::vector<std::pair<T, size_type>> temp(krylov_dim);
            bool do_restart = true;

            for(iter = 0; iter < m_max_iter && do_restart; ++iter)
            {
                size_type istart = 0;
                size_type iend = krylov_dim;
                bool keep_running = true;
                size_type iend_start = std::min(neigs, krylov_dim);

                for(iend = iend_start; iend < krylov_dim+istride && keep_running; iend+=istride)
                {
                    if(iend > krylov_dim){iend = krylov_dim; keep_running = false;}   
                    try
                    {
                        bool ended_early = false;

                        //perform partial arnoldi step 
                        CALL_AND_HANDLE(ended_early = m_arnoldi.partial_krylov_step(x, 0, scalefactor, istart, iend, false, std::forward<Args>(args)...), "Failed to construct the krylov subspace using a arnoldi iteration");

                        if(iend > 1)
                        {
                            if(ended_early){keep_running = false;}
                            auto H = m_arnoldi.H();      

                            m_eigensolver(H, m_vals, m_rvecs, m_lvecs, false);
                            eigenvalues_evaluated = eigenvalue_ordering(temp, eigorder);

                            //get the last eigenvalue index
                            size_type eigindex = eigorder[eigenvalues_evaluated-1];

                            //and compute the associated residue
                            m_residues(iter) = m_arnoldi.hk1k()*linalg::abs(m_rvecs(m_vals.size()-1, eigindex));

                            if(m_residues(eigindex, iter) < m_eps || m_residues(eigindex, iter) < m_rel_eps*linalg::abs((m_invert_mode ? 1.0/m_vals(eigindex, eigindex) : m_vals(eigindex, eigindex))))
                            {
                                do_restart = false;
                                keep_running=false;
                            }
                        }
                    } 
                    catch(const std::exception& ex)
                    {
                        std::cerr << ex.what() << std::endl;
                        RAISE_EXCEPTION("Error when attempting to compute eigenvalues.");
                    }
                }

                for(size_type i = 0; i < std::min(eigenvalues_evaluated, neigs); ++i)
                {
                    auto xeig = x[i];
                    size_type eigenvalue_index = eigorder[i];
                    get_vecs_and_vals(E[i], xeig, eigenvalue_index);

                }
                if(max_iters < iter)
                {
                    max_iters = iter;
                }
                istart = iend + 1;
                if(istart > krylov_dim){keep_running = false;}
            }
            m_niters = max_iters+1;
            return eigenvalues_evaluated;
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
    //TODO: Need to fix arnoldi with explicit deflation.  Currently not using this code
    //template <typename vals_type, typename vecs_type,  typename ... Args>
    //typename std::enable_if<linalg::is_same_backend<vals_type, linalg::vector<value_type, backend_type>>::value && linalg::is_same_backend<vecs_type, linalg::vector<value_type, backend_type>>::value, size_t>::type 
    //operator()(vecs_type& x, vals_type& E, Args&& ... args)
    //{
    //    try
    //    {
    //        size_type neigs = x.shape(0);
    //        size_type size =  x.size()/neigs;
    //        
    //        if(m_neigs > 0)
    //        {
    //            neigs = std::min(neigs, m_neigs);
    //        }
    //        m_neigs = neigs;

    //        size_type krylov_dim = std::min(m_krylov_dim, size);
    //        size_type istride = std::min(m_istride, krylov_dim);

    //        ASSERT(neigs <= m_krylov_dim, "Cannot compute more eigenvalues than the dimension of the krylov subspace.");  
    //        if(neigs > size){neigs = size;}
    //        if(E.size(0) != neigs){E.resize(neigs);}

    //        //construct the krylov subspace and store the final matrix element required for computing error estimates and matrix exponentials
    //        CALL_AND_HANDLE(m_arnoldi.resize(krylov_dim, size), "Failed to resize krylov subspace.");
    //        CALL_AND_HANDLE(m_arnoldi.reset_zeros(), "Failed to reset arnoldi iteration.");
    //    
    //        //compute the arnoldi iteration. 
    //        real_type scalefactor = 1.0;

    //        CALL_AND_HANDLE(m_residues.resize(neigs, m_max_iter), "Failed to resize residues array.");  m_residues.fill_zeros();
    //        CALL_AND_HANDLE(m_converged.resize(neigs), "Failed to resize converged array.");    m_converged.fill_value(false);

    //        //now compute the eigenvalues in the arnoldi subspace
    //        CALL_AND_HANDLE(m_vals.resize(krylov_dim), "Failed to resize the working buffer.");

    //        bool all_converged = true;
    //        size_type max_iters = 0;
    //        for(size_type eigindex = 0; eigindex < neigs; ++eigindex)
    //        {
    //            size_type iter = 0;

    //            size_type eigenvalue_index=0;
    //            bool do_restart = true;
    //            for(iter = 0; iter < m_max_iter && do_restart; ++iter)
    //            {
    //                size_type istart = 0;
    //                size_type iend = krylov_dim;
    //                bool keep_running = true;
    //                size_type iend_start = std::min(size_type(2), krylov_dim);

    //                for(iend = iend_start; iend < krylov_dim+istride && keep_running; iend+=istride)
    //                {
    //                    if(iend > krylov_dim){iend = krylov_dim; keep_running = false;}   
    //                    try
    //                    {
    //                        bool ended_early = false;

    //                        //perform partial arnoldi step 
    //                        CALL_AND_HANDLE(ended_early = m_arnoldi.partial_krylov_step_deflation(x, eigindex, scalefactor, istart, iend, std::forward<Args>(args)...), "Failed to construct the krylov subspace using a arnoldi iteration");

    //                        if(iend > 1)
    //                        {
    //                            if(ended_early){keep_running = false;}
    //                            auto H = m_arnoldi.H();      

    //                            m_eigensolver(H, m_vals, m_rvecs, m_lvecs, false);
    //                            size_type n = get_eig_index();
    //                            eigenvalue_index = n;

    //                            m_residues(eigindex, iter) = m_arnoldi.hk1k()*linalg::abs(m_rvecs(m_vals.size()-1, n));
    //                            if(m_verbose)
    //                            {
    //                                std::cerr << eigindex << " " << iter << " " << iend << " " << m_residues(eigindex, iter) << " " << (m_invert_mode ? 1.0/m_vals(n, n) : m_vals(n, n)) << std::endl;
    //                            }
    //                            if(m_residues(eigindex, iter) < m_eps || m_residues(eigindex, iter) < m_rel_eps*linalg::abs((m_invert_mode ? 1.0/m_vals(n, n) : m_vals(n, n))))
    //                            {
    //                                do_restart = false;
    //                                keep_running=false;
    //                                m_converged(eigindex) = true;
    //                            }
    //                        }
    //                    } 
    //                    catch(const std::exception& ex)
    //                    {
    //                        std::cerr << ex.what() << std::endl;
    //                        RAISE_EXCEPTION("Error when attempting to compute eigenvalues.");
    //                    }
    //                }

    //                auto xeig = x[eigindex];
    //                get_vecs_and_vals(E[eigindex], xeig, eigenvalue_index);
    //                if(max_iters < iter)
    //                {
    //                    max_iters = iter;
    //                }
    //                istart = iend + 1;
    //                if(istart > krylov_dim){keep_running = false;}
    //            }
    //            if(!m_converged(eigindex)){all_converged = false;}
    //        }
    //        m_niters = max_iters+1;
    //        return all_converged;
    //    }
    //    catch(const common::invalid_value& ex)
    //    {
    //        std::cerr << ex.what() << std::endl;
    //        RAISE_NUMERIC("performing krylov subspace integration");
    //    }
    //    catch(const std::exception& ex)
    //    {
    //        std::cerr << ex.what() << std::endl;
    //        RAISE_EXCEPTION("Failed to perform krylov subspace integration.");
    //    }
    //}

protected:
    size_t get_eig_index()
    {
        size_type index = 0;
        real_type m_val = -1;
        if(m_mode == eigenvalue_target::largest_magnitude)
        {   
            for(size_type i = 0; i < m_vals.size(0); ++i)
            {
                if(linalg::abs(m_vals(i,i)) > m_val){m_val = linalg::abs(m_vals(i,i)); index = i;}
            }
        }
        else if(m_mode == eigenvalue_target::smallest_real)
        {
            for(size_type i = 0; i < m_vals.size(0); ++i)
            {
                if(linalg::real(m_vals(i,i)) < m_val || i == 0 ){m_val = linalg::real(m_vals(i,i)); index = i;}
            }
        }
        else if(m_mode == eigenvalue_target::largest_real)
        {
            for(size_type i = 0; i < m_vals.size(0); ++i)
            {
                if(linalg::real(m_vals(i,i)) > m_val || i == 0 ){m_val = linalg::real(m_vals(i,i)); index = i;}
            }
        }
        return index;
    }


    size_type eigenvalue_ordering(std::vector<std::pair<T, size_type>>& eigs, std::vector<size_type>& inds) const
    {
        for(size_type i = 0; i < m_vals.size(0); ++i)
        {
            eigs[i] = std::make_pair(m_vals(i, i), i);
        }

        if(m_mode == eigenvalue_target::largest_magnitude)
        {   
            std::sort(eigs.begin(), eigs.end(), [](const std::pair<T, size_type>& a, const std::pair<T, size_type>& b){return linalg::abs(a.first) > linalg::abs(b.first);});
        }
        else if(m_mode == eigenvalue_target::smallest_real)
        {
            std::sort(eigs.begin(), eigs.end(), [](const std::pair<T, size_type>& a, const std::pair<T, size_type>& b){return linalg::real(a.first) < linalg::real(b.first);});
        }
        else if(m_mode == eigenvalue_target::largest_real)
        {
            std::sort(eigs.begin(), eigs.end(), [](const std::pair<T, size_type>& a, const std::pair<T, size_type>& b){return linalg::real(a.first) > linalg::real(b.first);});
        }

        for(size_type i =0; i < m_vals.size(0); ++i)
        {
            inds[i] = eigs[i].second;
        }
        return m_vals.size(0);
    }


    template <typename vectype>
    void get_vecs_and_vals(value_type& E, vectype& x, size_t n)
    {
        try
        {
            //sort and scale eigenvalues
            E = m_invert_mode ? 1.0/m_vals(n, n) : m_vals(n, n);

            //sort eigenvectors in krylov subspace
            CALL_AND_HANDLE(m_rvecsd.resize(1, m_lvecs.size(0)), "Failed to resize rvecs array.");
            CALL_AND_HANDLE(m_lvecs = linalg::adjoint(m_rvecs), "Failed to store rvecs in temporary.");
            m_rvecsd[0] = m_lvecs[n];

            //and transform to original space
            auto xm = x.reinterpret_shape(1, x.size());
            auto xv = x.reinterpret_shape(x.size());
            CALL_AND_HANDLE(xm = m_rvecsd*m_arnoldi.Q(), "Failed to compute eigenvectors.");
            real_type norm = std::sqrt( linalg::real(linalg::dot_product(linalg::conj(xv), xv)));
            x /= norm;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to sort eigenvalues and eigenvectors.");
        }
    }
};

}


#endif

