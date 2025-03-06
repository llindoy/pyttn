#ifndef UTILS_EXPLICIT_ARNOLDI_HPP
#define UTILS_EXPLICIT_ARNOLDI_HPP


#include <limits>
#include <utility>
#include <algorithm>

#include <common/exception_handling.hpp>
#include <common/tmp_funcs.hpp>

#include <linalg/linalg.hpp>
#include <linalg/decompositions/sparse/arnoldi_iteration.hpp>
#include <linalg/decompositions/eigensolvers/eigensolver.hpp>

namespace utils
{

template <typename T, typename backend = linalg::blas_backend>
class explicitly_restarted_arnoldi;

template <typename T, typename backend>
class explicitly_restarted_arnoldi<complex<T>, backend>
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
    linalg::vector<value_type> m_x;

    std::vector<std::pair<value_type, size_type>> m_sorted_vals;

    linalg::vector<value_type, linalg::blas_backend> m_e1;
    linalg::matrix<value_type, linalg::blas_backend> m_rvecs;
    linalg::matrix<value_type, linalg::blas_backend> m_rvecsd;
    linalg::matrix<value_type, linalg::blas_backend> m_lvecs;
    linalg::vector<value_type, backend> m_tempr;
        

    linalg::vector<real_type> m_residues;

    size_type m_krylov_dim;
    size_type m_istride;
    size_type m_max_iter;
    real_type m_eps;

public:
    explicitly_restarted_arnoldi() : m_arnoldi(), m_krylov_dim(0), m_istride(1), m_max_iter(1), m_eps(std::numeric_limits<real_type>::epsilon()*1e3) {}
    explicitly_restarted_arnoldi(size_type krylov_dim, size_type dim, real_type eps = std::numeric_limits<real_type>::epsilon()*1e3) : m_arnoldi(), m_krylov_dim(krylov_dim), m_istride(1), m_max_iter(1), m_eps(eps) {CALL_AND_HANDLE(resize(krylov_dim, dim), "Failed to construct krylov subspace integrator.");}
    explicitly_restarted_arnoldi(const explicitly_restarted_arnoldi& o) = default;
    explicitly_restarted_arnoldi(explicitly_restarted_arnoldi&& o) = default;

    explicitly_restarted_arnoldi& operator=(const explicitly_restarted_arnoldi& o) = default;
    explicitly_restarted_arnoldi& operator=(explicitly_restarted_arnoldi&& o) = default;

    void resize(size_type krylov_dim, size_type dim)
    {
        try
        {
            m_krylov_dim = krylov_dim;
            CALL_AND_HANDLE(m_arnoldi.resize(krylov_dim, dim), "Failed to resize arnoldi iteration engine.");
            CALL_AND_HANDLE(m_eigensolver.resize(krylov_dim, false), "Failed to resize upper_hessenberg_eigensolver.");
            CALL_AND_HANDLE(m_e1.resize(krylov_dim), "Failed to resize the e1 vector.");

            CALL_AND_HANDLE(m_vals.resize(krylov_dim, krylov_dim), "Failed to resize the temp1 vector.");
            CALL_AND_HANDLE(m_sorted_vals.resize(krylov_dim), "Failed to resize the temp1 vector.");

            CALL_AND_HANDLE(m_rvecs.resize(krylov_dim, krylov_dim), "Failed to resize the temp1 vector.");
            CALL_AND_HANDLE(m_rvecsd.resize(1, krylov_dim), "Failed to resize the temp1 vector.");
            CALL_AND_HANDLE(m_lvecs.resize(krylov_dim, krylov_dim), "Failed to resize the temp1 vector.");
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
            CALL_AND_HANDLE(m_tempr.clear(), "Failed to resize the temp2 vector.");
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to resize krylov integrator.");
        }
    }
  
    const size_type& stride() const {return m_istride;}
    size_type& stride() {return m_istride;}

    const real_type& tol() const {return m_eps;}
    real_type& tol() {return m_eps;}

    const size_type& krylov_dim() const{return m_krylov_dim;}

    const linalg::vector<real_type>& residues() const{return m_residues;}
    const real_type& residue(size_t i) const{ASSERT(i < m_residues.size(), "Index out of bounds."); return m_residues[i];}
    const size_type& niter() const{return m_residues.size();}

    size_type& max_iter() {return m_max_iter;}
    const size_type& max_iter() const{return m_max_iter;}

    template <typename vec_type, typename ... Args>
    typename std::enable_if<linalg::is_same_backend<vec_type, linalg::vector<value_type, backend_type>>::value, bool>::type 
    operator()(vec_type& x, value_type& E, Args&& ... args)
    {
        try
        {
            size_type neigs = 1;
            size_type krylov_dim = m_krylov_dim < x.size() ? m_krylov_dim : x.size();
            ASSERT(m_istride <= krylov_dim, "The stride for determining error estimates is too large.");
            ASSERT(neigs <= m_krylov_dim, "Cannot compute more eigenvalues than the dimension of the krylov subspace.");  
            if(neigs > x.size()){neigs = x.size();}

            //construct the krylov subspace and store the final matrix element required for computing error estimates and matrix exponentials
            CALL_AND_HANDLE(m_tempr.resize(x.size()), "Failed to resize the working buffer.");
            CALL_AND_HANDLE(m_arnoldi.resize(krylov_dim, x.size()), "Failed to resize krylov subspace.");
            CALL_AND_HANDLE(m_arnoldi.reset_zeros(), "Failed to reset arnoldi iteration.");
            CALL_AND_HANDLE(m_sorted_vals.resize(krylov_dim), "Failed to resize the working buffer.");
        
            //compute the arnoldi iteration. 
            real_type scalefactor = 1.0;

            E = value_type(0);
            CALL_AND_HANDLE(m_residues.resize(m_max_iter), "Failed to resize residues array.");

            //now compute the eigenvalues in the arnoldi subspace
            CALL_AND_HANDLE(m_vals.resize(krylov_dim), "Failed to resize the working buffer.");
            for(size_t iter = 0; iter <= m_max_iter; ++iter)
            {
                size_type istart = 0;
                size_type iend = krylov_dim;
                bool keep_running = true;
                size_t iend_start = neigs;

                if(iter > 0)
                {
                    m_x.resize(x.size());
                    auto mx = m_x.reinterpret_shape(x.shape());
                    m_arnoldi.compute_action(x, m_x, std::forward<Args>(args)...);
      
                    mx -= E*x;
                    m_residues[iter-1] = std::sqrt( linalg::real(linalg::dot_product(linalg::conj(m_x), m_x)));
                    //std::cerr << E << " " << m_residues[iter-1] << std::endl;
                    if(m_residues[iter-1]/linalg::abs(E) < m_eps)
                    {
                        m_residues.resize(iter);
                        return true;
                    }
                }

                if(iter != m_max_iter)
                {
                    for(iend = iend_start; iend < krylov_dim+m_istride && keep_running; iend+=m_istride)
                    {
                        if(iend > krylov_dim){iend = krylov_dim; keep_running = false;}   
                        try
                        {
                            bool ended_early = false;
                            CALL_AND_HANDLE(ended_early = m_arnoldi.partial_krylov_step(x, scalefactor, istart, iend, std::forward<Args>(args)...), "Failed to construct the krylov subspace using a arnoldi iteration");
                            if(ended_early){keep_running = false;}
                            auto H = m_arnoldi.H();      

                            m_eigensolver(H, m_vals, m_rvecs, m_lvecs, false);
                            
                            //sort the eigenvalues based on norm and take the one with the largest as our value
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
                            //std::cerr << m_vals << std::endl;
                        } 
                        catch(const std::exception& ex)
                        {
                            std::cerr << ex.what() << std::endl;
                            RAISE_EXCEPTION("Error when attempting to compute eigenvalues.");
                        }
                    }

                    //get the largest eigenvalue and corresponding eigenvector
                    get_vecs_and_vals(E, x);
                    //std::cerr << 1.0/E << std::endl;
                }
            }
            return false;
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
    void get_vecs_and_vals(value_type& E, linalg::vector<value_type, backend_type>& x)
    {
        try
        {
            //sort and scale eigenvalues
            E = std::get<0>(m_sorted_vals[0]);

            //sort eigenvectors in krylov subspace
            CALL_AND_HANDLE(m_lvecs = linalg::adjoint(m_rvecs), "Failed to store rvecs in temporary.");
            m_rvecsd[0] = m_lvecs[std::get<1>(m_sorted_vals[0])];

            //and transform to original space
            auto xm = x.reinterpret_shape(1, x.size());
            CALL_AND_HANDLE(xm = m_rvecsd*m_arnoldi.Q(), "Failed to compute eigenvectors.");
            real_type norm = std::sqrt( linalg::real(linalg::dot_product(linalg::conj(x), x)));
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

