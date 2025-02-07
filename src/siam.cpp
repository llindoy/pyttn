//#define TIMING 0
//#define USE_OLD
#define TTNS_REGISTER_COMPLEX_DOUBLE_OPERATOR

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>
#endif

#include <ttns_lib/ttns.hpp>

#include <ttns_lib/sop/sSOP.hpp>
#include <ttns_lib/sop/SOP.hpp>
#include <ttns_lib/sop/multiset_SOP.hpp>
#include <ttns_lib/sop/compressedSOP.hpp>
#include <ttns_lib/sop/system_information.hpp>

#include <utils/io/input_wrapper.hpp>

#include <ttns_lib/operators/sop_operator.hpp>
#include <ttns_lib/operators/multiset_sop_operator.hpp>
#include <ttns_lib/sweeping_algorithm/tdvp.hpp>

#include <ttns_lib/sop/models/spin_boson.hpp>

#include <chrono>
#include <map>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <algorithm>

using namespace ttns;

inline size_t nU(size_t i, size_t N){return N-(i+1);}
inline size_t nD(size_t i, size_t N){return N+i;}
inline size_t nind(size_t i, size_t N, size_t s)
{
    if(s == 0){return nU(i, N);}
    else{return nD(i, N);}
}

int main(int argc, char* argv[])
{
    try
    {
        using real_type = double;
        using complex_type = linalg::complex<double>;
        using backend_type = linalg::blas_backend;
        using namespace utils;
        backend_type::initialise();

        if(argc < 8)
        {
            std::cerr << argv[0] << " <N> <Gamma> <W> <eps> <U> <chi> <nbranch>" << std::endl;
            return 1;
        }

        size_t N = std::atoi(argv[1]);
        real_type Gamma = std::atof(argv[2]);
        real_type W = std::atoi(argv[3]);
        real_type eps = std::atoi(argv[4]);
        real_type U = std::atoi(argv[5]);
        size_t nchi = std::atoi(argv[6]);
        size_t nbranch = std::atoi(argv[7]);
        INIT_TIMER;

        {
            START_TIMER;

            std::vector<real_type> t(N);
            std::vector<real_type> e(N);

            for(size_t i = 0; i < N; ++i)
            {
                t[i] = W/2.0*i;
                e[i] = i+1.0;

            }
            t[0] = std::sqrt(Gamma*W/2.0);

            size_t mdim = nchi;

            system_modes sysinf(2*(N+1));
            for(size_t i = 0; i < 2*(N+1); ++i)
            {
                sysinf[i] = qubit_mode();
            }

            SOP<complex_type> sop(2*(1+N));
            sop += U*sOP("z", nU(0, N+1)) * sOP("z", nD(0, N+1));
            for(size_t s=0; s < 2; ++s)
            {
                sop += eps*sOP("z", nind(0, N+1, s));
                for(size_t i = 0; i  < N; ++i)
                {
                    sop += t[i]*sOP("sigma+", nind(i, N+1, s)) * sOP("sigma-", nind(i+1, N+1, s));
                    sop += t[i]*sOP("sigma+", nind(i+1, N+1, s)) * sOP("sigma-", nind(i, N+1, s));
                    sop += e[i]*sOP("z", nind(i+1, N+1, s));
                }
            }
            //cdagsop.jordan_wigner();

            std::vector<size_t> dims(2*(N+1));
            for(size_t i = 0; i < dims.size(); ++i)
            {
                dims[i] = 2;
            }
            ntree<size_t> topology = ntree_builder<size_t>::htucker_tree(dims, nbranch, nchi);
            //ntree<size_t> topology;
            //topology.insert(1);
            //topology().insert(2); topology()[0].insert(2);
            //topology().insert(2);
            //ntree_builder<size_t>::htucker_subtree(topology()[1], sbm.mode_dims(), nbranch, nchi);
            ntree_builder<size_t>::sanitise_tree(topology, false);


            ntree<size_t> topologyb = ntree_builder<size_t>::htucker_tree(dims, nbranch, nchi*2);
            //ntree<size_t> topologyb;
            //topologyb.insert(1);
            //topologyb().insert(2); topologyb()[0].insert(2);
            //topologyb().insert(2);
            //ntree_builder<size_t>::htucker_subtree(topologyb()[1], sbm.mode_dims(), nbranch, nchi*2);
            ntree_builder<size_t>::sanitise_tree(topologyb, false);
            std::cout << topology << std::endl;
            std::vector<size_t> zeros(2*(N+1));   std::fill(zeros.begin(), zeros.end(), 1);


            ttn<complex_type, backend_type> A(topology);      A.random();
            ttn<complex_type, backend_type> B(topologyb);     B.set_state(zeros);

            {
            sop_operator<complex_type, backend_type> sop_op(sop, A, sysinf, false);
            std::cerr << "uncompressed" << std::endl;

            std::mt19937 rng;
            std::uniform_int_distribution<size_t> dist(0, A.ntensors()-1);
            for(size_t i = 0; i < 100; ++i)
            {
                size_t ind= dist(rng);
                A.set_orthogonality_centre(ind);
            }

            {
                using sweeping_type = sweeping_algorithm<complex_type, backend_type, ttn, energy_debug_engine, sop_environment>;
                //typename sweeping_type::env_type env;
                sweeping_type sweep(A, sop_op);
                CALL_AND_HANDLE(sweep.prepare_environment(A, sop_op), "Failed to prepare the hamiltonian buffer for evolution.");

                for(size_t i = 0; i < 1; ++i)
                {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    sweep(A, sop_op);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
                    std::cout << duration.count() << std::endl;
                }
            }
            }
            {
            sop_operator<complex_type, backend_type> sop_op(sop, A, sysinf, true);
            std::cerr << "compressed" << std::endl;

            std::mt19937 rng;
            std::uniform_int_distribution<size_t> dist(0, A.ntensors()-1);
            for(size_t i = 0; i < 100; ++i)
            {
                size_t ind= dist(rng);
                A.set_orthogonality_centre(ind);
            }

            {
                using sweeping_type = sweeping_algorithm<complex_type, backend_type, ttn, energy_debug_engine, sop_environment>;
                //typename sweeping_type::env_type env;
                sweeping_type sweep(A, sop_op);
                CALL_AND_HANDLE(sweep.prepare_environment(A, sop_op), "Failed to prepare the hamiltonian buffer for evolution.");

                for(size_t i = 0; i < 1; ++i)
                {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    sweep(A, sop_op);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
                    std::cout << duration.count() << std::endl;
                }
            }
            }


            return 0;
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

}





