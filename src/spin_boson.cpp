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
#include <ttns_lib/sweeping_algorithm/dmrg.hpp>
#include <ttns_lib/sweeping_algorithm/subspace_expansion/variance_subspace_expansion_engine.hpp>

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


int main(int argc, char* argv[])
{
    try
    {
        using real_type = double;
        using complex_type = linalg::complex<double>;
        using backend_type = linalg::blas_backend;
        using namespace utils;
        backend_type::initialise();

        if(argc < 12)
        {
            std::cerr << argv[0] << " <N> <alpha> <wc> <s> <beta> <nchi> <nbranch> <nbose> <eps> <weight trunc> <svd trunc>" << std::endl;
            return 1;
        }

        size_t N = std::atoi(argv[1]);
        real_type alpha = std::atof(argv[2]);
        real_type wc = std::atof(argv[3]);
        real_type s = std::atof(argv[4]);
        real_type beta = std::atof(argv[5]);
        size_t nchi = std::atoi(argv[6]);
        size_t nbranch = std::atoi(argv[7]);
        size_t mdim = std::atoi(argv[8]);
        real_type eps = std::atof(argv[9]);
        real_type unocc_trunc = std::atof(argv[10]);
        real_type svd_trunc = std::atof(argv[11]);
        INIT_TIMER;

        {
            START_TIMER;

            size_t Nb;
            if(beta < 0){Nb = N;}
            else{Nb = 2*N;}
            std::vector<complex_type> g(Nb);
            std::vector<real_type> w(Nb);

            for(size_t i = 0; i < N; ++i)
            {
                if(beta < 0)
                {
                    w[i] = -wc*std::log(1.0-(i+1)/(N+1.));
                    g[i] = std::sqrt(2*alpha*wc*wc*std::pow(w[i]/wc, s)/(N+1.0));
                }
                else
                {
                    real_type wi = -wc*std::log(1.0-(i+1)/(N+1.));
                    real_type gi = std::sqrt(2*alpha*wc*wc*std::pow(wi/wc, s)/(N+1.0));
                    w[2*i] = wi;
                    w[2*i+1] = -wi;
                    g[2*i] = gi*0.5*(1+1.0/std::tanh(wi*beta/2.0));
                    g[2*i+1] = gi*0.5*(1+1.0/std::tanh(-wi*beta/2.0));
                }
            }

            real_type delta = 2.0;
            spin_boson_star<complex_type> sbm(eps, delta, w, g);

            for(size_t i = 0; i < Nb; ++i)
            {
                sbm.mode_dim(i) = mdim;
            }


            SOP<complex_type> sop(1+Nb);
            system_modes sysinf;

            sop += eps*sOP("sz", 0) + delta*sOP("sx", 0);
            for(size_t i = 0; i < Nb; ++i)
            {
                complex_type gi = g[i];
                sop +=  std::sqrt(2.0)*gi*(sOP("sz", 0)*sOP("q", i+1));
                sop += w[i]*sOP("n", i+1);
            }

            //sbm.hamiltonian(sop);
            sbm.system_info(sysinf);

            std::vector<size_t> dims(Nb+1);
            dims[0] = 2;
            for(size_t i = 0; i < Nb; ++i)
            {
                dims[i+1] = sbm.mode_dims()[i];
            }
            ntree<size_t> topology;
            topology.insert(1);
            topology().insert(2); topology()[0].insert(2);
            topology().insert(2);
            ntree_builder<size_t>::htucker_subtree(topology()[1], sbm.mode_dims(), nbranch, nchi);
            ntree_builder<size_t>::sanitise_tree(topology, false);

            ntree<size_t> capacity;
            capacity.insert(1);
            capacity().insert(2); capacity()[0].insert(2);
            capacity().insert(2);
            ntree_builder<size_t>::htucker_subtree(capacity()[1], sbm.mode_dims(), nbranch, mdim);
            ntree_builder<size_t>::sanitise_tree(capacity, false);

            std::cout << topology << std::endl;
            std::vector<size_t> zeros(Nb+1);   std::fill(zeros.begin(), zeros.end(), 0);

            ttn<complex_type, backend_type> A(topology, capacity);      A.set_state(zeros);

            sop_operator<complex_type, backend_type> sop_op(sop, A, sysinf);

            sOP sz = sOP("sz", 0);
            site_operator<complex_type> op(sz, sysinf);

            //matrix_element<complex_type> mel;//(msA, ms_sop_op);
            matrix_element<complex_type> mel;//(A, sop_op);
            mel.resize(A, sop_op);

            //sweeping_algorithm<complex_type, backend_type, ttn, tdvp_engine, sop_environment, variance_subspace_expansion> sweep(A, sop_op, {16}, {}, {4, 2});
            adaptive_one_site_tdvp<complex_type, backend_type> sweep(A, sop_op, 16, 1, 4, 2);
            //one_site_tdvp<complex_type, backend_type> sweep(A, sop_op, 16);
            sweep.dt() = 0.0025;
            sweep.coefficient() = complex_type(0, 1);
            sweep.krylov_steps() = 1;
            sweep.minimum_unoccupied() = 1;
            sweep.unoccupied_threshold() = unocc_trunc;
            sweep.spawning_threshold() = svd_trunc;
            sweep.truncation_mode() = orthogonality::truncation_mode::weight_truncation;
            sweep.use_time_dependent_hamiltonian() = true;
            CALL_AND_HANDLE(sweep.prepare_environment(A, sop_op), "Failed to prepare the hamiltonian buffer for evolution.");

            std::cout << std::setprecision(16);
            std::cout << 0 << " " << std::real(mel(op, A)) << std::endl;
            size_t counter = 4;
            for(size_t i = 0; i < 1000; ++i)
            {
                sweep.maximum_bond_dimension() = counter;
                auto t1 = std::chrono::high_resolution_clock::now();
                sweep(A, sop_op);
                auto t2 = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
                std::cout << (i+1)*sweep.dt() << " " << std::real(mel(op, A)) << " " << A.maximum_bond_dimension() << " " << duration.count() << std::endl;
                if(i%100 == 0)
                {
                    counter += 1;
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





