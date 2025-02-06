#ifndef LINALG_UTILS_ORTHOGONAL_VECTOR_HPP
#define LINALG_UTILS_ORTHOGONAL_VECTOR_HPP

#include "linalg_utils.hpp"
#include "genrandom.hpp"
#include "../dense.hpp"

namespace linalg
{

template <typename T, typename backend>
class orthogonal_vector
{
public:
    using real_type = typename get_real_type<T>::type;
    using size_type = typename backend::size_type;
public:
    template <typename mattype>
    static void pad_random(mattype&& ct, size_type iskip, random_engine<backend>& rng, real_type tol=1e-12)
    {

        for(size_type i=iskip; i<std::min(ct.shape(0), ct.shape(1)); ++i)
        {
            bool vector_generated = false;
        
            while(!vector_generated)
            {
                //generate a random vector
                rng.fill_normal(ct[i]);

                //now we normalise it
                ct[i] /= std::sqrt(real(dot_product(conj(ct[i]), ct[i])));

                //now we attempt to modified gram-schmidt this
                //if we run into linear dependence then we need to try another random vector
                for(size_type j=0; j < i; ++j)
                {
                    ct[i] -= dot_product(conj(ct[j]), ct[i])*ct[j];
                }

                //now we compute the norm of the new vector
                real_type norm = std::sqrt(real(dot_product(conj(ct[i]), ct[i])));
                if(norm > tol)
                {
                    ct[i] /= norm;
                    vector_generated = true;
                }
            }
        }
    }

    template <typename mattype, typename rettype>
    static void generate(mattype&& ct, rettype&& x, random_engine<backend>& rng, real_type tol=1e-12)
    {
        bool vector_generated = false;

        ASSERT(x.size() == ct.shape(1), "Cannot generate random vector output vector is not the correct size.");
        ASSERT(ct.shape(0) < ct.shape(1), "Cannot generate random orthogonal output vector if the matrix fully spans the space.");
        
        while(!vector_generated)
        {
            //generate a random vector
            rng.fill_normal(x);

            //now we normalise it
            x /= std::sqrt(real(dot_product(conj(x), x)));

            //now we attempt to modified gram-schmidt this
            //if we run into linear dependence then we need to try another random vector
            for(size_type j=0; j < ct.shape(0); ++j)
            {
                x -= dot_product(conj(ct[j]), x)*ct[j];
            }

            //now we compute the norm of the new vector
            real_type norm = std::sqrt(real(dot_product(conj(x), x)));
            if(norm > tol)
            {
                x /= norm;
                vector_generated = true;
            }
        }
    }
};

}   //linalg

#endif  //LINALG_UTILS_ORTHOGONAL_VECTOR_HPP

