#ifndef LINALG_UTILS_RANDOM_ENGINE_HPP
#define LINALG_UTILS_RANDOM_ENGINE_HPP

#include <random>
#include "../backends/curand_wrapper.hpp"
#include "../linalg_forward_decl.hpp"
#include "../linalg_traits.hpp"

namespace linalg
{
template <typename backend>
class random_engine;

template <>
class random_engine<linalg::blas_backend>
{
protected:
    template <typename T, typename = void>
    class random_normal;

    template <typename T>
    class random_normal<T, typename std::enable_if<not linalg::is_complex<T>::value, void>::type>
    {
    public:
        using real_type = T;
    public:
        static inline T generate(std::mt19937& rng, std::normal_distribution<T>& dist)
        {
            return dist(rng);
        }
    };

    template <typename T>
    class random_normal<T, typename std::enable_if<linalg::is_complex<T>::value, void>::type>
    {
    public:
        using real_type = typename linalg::get_real_type<T>::type;
    public:
        static inline T generate(std::mt19937& rng, std::normal_distribution<real_type>& dist)
        {
            real_type div = 1.0/std::sqrt(2.0);
            real_type a =dist(rng)/div;
            real_type b = dist(rng)/div;
            return T(a,b);
        }
    };


public:
    //constructors for random number generation
    random_engine(){}
    template <typename sseq>
    random_engine(sseq& seed){m_rng.seed(seed);}
    random_engine(const random_engine& o) = default;
    random_engine(random_engine&& o) = default;
    random_engine& operator=(const random_engine& o) = default;
    random_engine& operator=(random_engine&& o) = default;

    template <typename sseq>
    void set_seed(sseq& seed){m_rng.seed(seed);}

    template <typename T>
    T generate_normal()
    {
        using real_type = typename linalg::get_real_type<T>::type;
        std::normal_distribution<real_type> dist(0, 1);

        return random_normal<T>::generate(m_rng, dist);
    }

    template <typename T>
    void fill_normal(std::vector<T>& arr)
    {
        using real_type = typename linalg::get_real_type<T>::type;
        std::normal_distribution<real_type> dist(0, 1);

        for(size_t i = 0; i < arr.size(); ++i)
        {
            arr[i] = random_normal<T>::generate(m_rng, dist);
        }
    }

    template <typename ArrType, typename = typename std::enable_if<is_dense_tensor<ArrType>::value && has_backend<ArrType, blas_backend>::value, void>::type>
    void fill_normal(ArrType& arr)
    {
        using T = typename linalg::traits<ArrType>::value_type;
        using real_type = typename linalg::get_real_type<T>::type;
        std::normal_distribution<real_type> dist(0, 1);

        for(size_t i = 0; i < arr.size(); ++i)
        {
            arr(i) = random_normal<T>::generate(m_rng, dist);
        }
    }

    template <typename ArrType, typename = typename std::enable_if<is_dense_tensor<ArrType>::value && has_backend<ArrType, blas_backend>::value, void>::type>
    void fill_normal(ArrType&& arr)
    {
        using T = typename linalg::traits<ArrType>::value_type;
        using real_type = typename linalg::get_real_type<T>::type;
        std::normal_distribution<real_type> dist(0, 1);

        for(size_t i = 0; i < arr.size(); ++i)
        {
            arr(i) = random_normal<T>::generate(m_rng, dist);
        }
    }
protected:
    std::mt19937 m_rng;
};


#ifdef PYTTN_BUILD_CUDA

#include <curand.h>


    template <typename T>
    struct generate_norm_vec;

    template <>
    struct generate_norm_vec<float>
    {
        static inline uint64_t generate(curandGenerator_t gen, float* buffer, size_t n)
        {
            float mean=0;
            float stdev=1;
            curand_safe_call(curandGenerateNormal(gen, buffer, n, mean, stdev));
            return n;
        }
    };

    template <>
    struct generate_norm_vec<double>
    {
        static inline uint64_t generate(curandGenerator_t gen, double* buffer, size_t n)
        {
            double mean=0;
            double stdev=1;
            curand_safe_call(curandGenerateNormalDouble(gen, buffer, n, mean, stdev));
            return n;
        }
    };

    //in order to generate normal distributed complex numbers we generate 2 times as many real numbers
    //with a standard deviation that is sqrt(2) smaller.
    template <>
    struct generate_norm_vec<complex<float>>
    {
        static inline uint64_t generate(curandGenerator_t gen, complex<float>* buffer, size_t n)
        {
            float mean=0;
            float stdev=1/std::sqrt(2.0);
            curand_safe_call(curandGenerateNormal(gen, reinterpret_cast<float*>(buffer), 2*n, mean, stdev));
            return 2*n;
        }
    };

    template <>
    struct generate_norm_vec<complex<double>>
    {
        static inline uint64_t generate(curandGenerator_t gen, complex<double>* buffer, size_t n)
        {
            double mean=0;
            double stdev=1/std::sqrt(2.0);
            curand_safe_call(curandGenerateNormalDouble(gen, reinterpret_cast<double*>(buffer), 2*n, mean, stdev));
            return 2*n;
        }
    };

template <>
class random_engine<linalg::cuda_backend>
{
public:
    //constructors for random number generation
    random_engine() : m_active(false) {initialise();}
    template <typename I>
    random_engine(I seed) : m_active(false) 
    {
        initialise();
        set_seed(seed);
    }

    random_engine(const random_engine& o)
    {
        if(o.active())
        {
            m_active=false;
            initialise();
        }
        set_seed(o.seed());
        m_ngenerated = o.ngenerated();
        m_active=o.active();
    }
    random_engine(random_engine&& o)
    {
        if(o.active())
        {            
            m_active=false;
            initialise();
        }
        set_seed(o.seed());
        m_ngenerated = o.ngenerated();
        m_active=o.active();

        o.clear();
    }
    ~random_engine(){clear();}
    random_engine& operator=(const random_engine& o)
    {
        clear();
        if(o.active())
        {
            m_active=false;
            initialise();
        }
        set_seed(o.seed());
        m_ngenerated = o.ngenerated();
        m_active=o.active();
        return *this;
    }
    random_engine& operator=(random_engine&& o)
    {
        clear();
        if(o.active())
        {
            m_active=false;
            initialise();
        }
        set_seed(o.seed());
        m_ngenerated = o.ngenerated();
        m_active=o.active();

        o.clear();
        return *this;
    }

    bool active() const {return m_active;}
    std::uint64_t ngenerated() const {return m_ngenerated;}
    unsigned long long seed () const{return m_seed;}

    template <typename I>
    void set_seed(I seed)
    {
        ASSERT(m_active, "Cannot set seed of inactive pseudo random number generator.");
        m_seed = seed;
        curand_safe_call(curandSetPseudoRandomGeneratorSeed(m_gen, m_seed));
    }

    template <typename ArrType, typename = typename std::enable_if<is_dense_tensor<ArrType>::value && has_backend<ArrType, cuda_backend>::value, void>::type>
    void fill_normal(ArrType& array)
    {
        using T = typename linalg::traits<ArrType>::value_type;
        m_ngenerated += generate_norm_vec<T>::generate(m_gen, array.buffer(), array.size());
    }

    template <typename ArrType, typename = typename std::enable_if<is_dense_tensor<ArrType>::value && has_backend<ArrType, cuda_backend>::value, void>::type>
    void fill_normal(ArrType&& array)
    {
        using T = typename linalg::traits<ArrType>::value_type;
        m_ngenerated += generate_norm_vec<T>::generate(m_gen, array.buffer(), array.size());
    }
protected:
    void initialise()
    {
        if(!m_active)
        {
            m_ngenerated=0;
            m_active=true;
            curand_safe_call(curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT));
        }
    }

    void clear()
    {
        if(m_active)
        {
            curand_safe_call(curandDestroyGenerator(m_gen));
        }
        m_active=false;
        m_ngenerated = 0;
        m_seed = 0;
    }
protected:
    curandGenerator_t m_gen;
    bool m_active;
    std::uint64_t m_ngenerated;
    unsigned long long m_seed;
};
#endif

}   //namespace linalg

#endif  //LINALG_UTILS_RANDOM_ENGINE_HPP
