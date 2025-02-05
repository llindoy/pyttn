#ifndef TTNS_TENSOR_NODE_HELPER_HPP
#define TTNS_TENSOR_NODE_HELPER_HPP


#include <common/exception_handling.hpp>
#include <linalg/linalg.hpp>

#include <vector>
#include <stdexcept>
#include <random>

namespace ttns
{

namespace helper
{

template <typename T, typename = void>
class random_number;

template <typename T>
class random_number<T, typename std::enable_if<not linalg::is_complex<T>::value, void>::type>
{
public:
    using real_type = T;

public:
    static inline T generate(std::mt19937& rng, std::normal_distribution<T>& length_dist, std::uniform_real_distribution<T>& /* angle_dist */)
    {
        return length_dist(rng);
    }
};

template <typename T>
class random_number<T, typename std::enable_if<linalg::is_complex<T>::value, void>::type>
{
public:
    using real_type = typename linalg::get_real_type<T>::type;

public:
    static inline T generate(std::mt19937& rng, std::normal_distribution<real_type>& length_dist, std::uniform_real_distribution<real_type>& angle_dist)
    {
        real_type r =length_dist(rng);
        real_type theta = angle_dist(rng);
        return T(r*std::cos(theta), r*std::sin(theta));
    }
};


template <typename T, typename backend>
class orthogonal_vector;

template <typename T>
class orthogonal_vector<T, linalg::blas_backend>
{
public:
    using real_type = typename linalg::get_real_type<T>::type;
    using size_type = typename linalg::blas_backend::size_type;
public:
    template <typename mattype>
    static void pad_random_vectors(mattype&& ct, size_type iskip, std::mt19937& rng)
    {
        std::uniform_real_distribution<real_type> dist(0, 2.0*acos(real_type(-1.0)));
        std::normal_distribution<real_type> length_dist(0, 1);
        for(size_type i=iskip; i<ct.shape(0); ++i)
        {
            bool vector_generated = false;
        
            while(!vector_generated)
            {
                //generate a random vector
                for(size_type j=0; j<ct.shape(1); ++j)
                {
                    ct(i, j) = random_number<T>::generate(rng, length_dist, dist);
                }

                //now we normalise it
                ct[i] /= std::sqrt(linalg::real(linalg::dot_product(linalg::conj(ct[i]), ct[i])));

                //now we attempt to modified gram-schmidt this
                //if we run into linear dependence then we need to try another random vector
                for(size_type j=0; j < i; ++j)
                {
                    ct[i] -= linalg::dot_product(linalg::conj(ct[j]), ct[i])*ct[j];
                }

                //now we compute the norm of the new vector
                real_type norm = std::sqrt(linalg::real(linalg::dot_product(linalg::conj(ct[i]), ct[i])));
                if(norm > 1e-12)
                {
                    ct[i] /= norm;
                    vector_generated = true;
                }
            }
        }
    }

    template <typename mattype, typename rettype>
    static void generate_random_vector(mattype&& ct, rettype&& x, std::mt19937& rng)
    {
        std::uniform_real_distribution<real_type> dist(0, 2.0*acos(real_type(-1.0)));
        std::normal_distribution<real_type> length_dist(0, 1);
        bool vector_generated = false;
        
        while(!vector_generated)
        {
            //generate a random vector
            for(size_type j=0; j<ct.shape(1); ++j)
            {
                x(j) = random_number<T>::generate(rng, length_dist, dist);
            }

            //now we normalise it
            x /= std::sqrt(linalg::real(linalg::dot_product(linalg::conj(x), x)));

            //now we attempt to modified gram-schmidt this
            //if we run into linear dependence then we need to try another random vector
            for(size_type j=0; j < ct.shape(0); ++j)
            {
                x -= linalg::dot_product(linalg::conj(ct[j]), x)*ct[j];
            }

            //now we compute the norm of the new vector
            real_type norm = std::sqrt(linalg::real(linalg::dot_product(linalg::conj(x), x)));
            if(norm > 1e-12)
            {
                x /= norm;
                vector_generated = true;
            }
        }
    }
};

#ifdef PYTTN_BUILD_CUDA
template <typename T>
class orthogonal_vector<T, linalg::cuda_backend>
{
public:
    template <typename r3type, typename vec2_type>
    inline void append(const r3type& atens, vec2_type&& x)
    {
        RAISE_EXCEPTION("Generating orthogonal trial vector current not supported for cuda backend.");
    }
};
#endif
}
}

#endif  //TTNS_TENSOR_NODE_HPP//


