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


int main(int argc, char* argv[])
{
    try
    {
        using real_type = double;
        using complex_type = linalg::complex<double>;
        using backend_type = linalg::blas_backend;
        using namespace utils;
        backend_type::initialise();

        if(argc < 6)
        {
            std::cerr << argv[0] << " <input filename>" << std::endl;
            return 1;
        }

        size_t N = std::atoi(argv[1]);
        real_type alpha = std::atof(argv[2]);
        real_type wc = std::atoi(argv[3]);
        size_t nchi = std::atoi(argv[4]);
        size_t nbranch = std::atoi(argv[5]);
        INIT_TIMER;

        {
            START_TIMER;

            std::vector<complex_type> g(N);
            std::vector<real_type> w(N);

            std::vector<complex_type> g2(N);
            std::vector<real_type> w2(N);

            std::vector<complex_type> g3(N);
            std::vector<real_type> w3(N);
            for(size_t i = 0; i < N; ++i)
            {
                w[i] = -wc*std::log(1.0-i/(N+1.));
                g[i] = std::sqrt(4*alpha* w[i]);

                w2[i] = -wc/10*std::log(1.0-i/(N+1.));
                g2[i] = std::sqrt(4*alpha* w[i]);

                w3[i] = -wc*1.5*std::log(1.0-i/(N+1.));
                g3[i] = std::sqrt(4*alpha* w[i]);
            }

            size_t mdim = nchi;

            real_type eps = 0.0;
            real_type delta = 2.0;
            spin_boson_star<complex_type> sbm(eps, delta, w, g);
            spin_boson_star<complex_type> sbm2(eps, delta, w2, g3);
            spin_boson_star<complex_type> sbm3(eps, delta, w2, g3);

            for(size_t i = 0; i < N; ++i)
            {
                sbm.mode_dim(i) = mdim;
            }


            SOP<complex_type> sop(1+N);
            system_modes sysinf;
            sbm.hamiltonian(sop);
            sbm.system_info(sysinf);

            SOP<complex_type> sop2(1+N);
            sbm2.hamiltonian(sop2);
            SOP<complex_type> sop3(1+N);
            sbm3.hamiltonian(sop3);


            multiset_SOP<complex_type> msSOP(2, 1+N);
            //msSOP(0, 0) = sop2;
            msSOP(0, 1) = sop2;
            msSOP(1, 0) = sop2; 
            //msSOP(1, 1) = sop2; 



            std::vector<size_t> dims(N+1);
            dims[0] = 2;
            for(size_t i = 0; i < N; ++i)
            {
                dims[i+1] = sbm.mode_dims()[i];
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
            std::vector<size_t> zeros(N+1);   std::fill(zeros.begin(), zeros.end(), 1);


            std::vector<complex_type> coeff(2);
            coeff[0] = 1.;
            coeff[1] = -0.5;
            std::vector<std::vector<size_t>> msState(2);    msState[0] = zeros; msState[1]=zeros;
            ms_ttn<complex_type, backend_type> msA(topology, 2);      msA.set_state(coeff, msState);
            ttn<complex_type, backend_type> A(topology);      A.set_state(zeros);
            ttn<complex_type, backend_type> B(topologyb);     B.set_state(zeros);

            sop_operator<complex_type, backend_type> sop_op(sop, A, sysinf);
            multiset_sop_operator<complex_type, backend_type> ms_sop_op(msSOP, A, sysinf);

            sOP sz = sOP("sz", 0);
            site_operator<complex_type> op(sz, sysinf);

            //matrix_element<complex_type> mel;//(msA, ms_sop_op);
            matrix_element<complex_type> mel;//(A, sop_op);
            mel.resize(A, sop_op);
            mel.resize(msA, ms_sop_op);
            ms_ttn<complex_type, backend_type> msB(msA);
            std::cout << std::real(mel(sop_op, A)) << std::endl;// " " << std::real(mel(msA, msB)) << std::endl;
            //std::cout << std::real(mel(sop_op, A)) << " " << std::real(mel(msA, msB)) << std::endl;
            //std::cout << std::real(mel(sop_op, A, B)) << " " << std::real(mel(msA, msB)) << std::endl;
            //std::cout << std::real(mel(ms_sop_op, msA)) << " " << std::real(mel(msA, msB)) << std::endl;
            //std::cout << std::real(mel(ms_sop_op, msA, msB)) << " " << std::real(mel(msA, msB)) << std::endl;
            //std::cout << std::real(mel(op, A)) << " " << std::real(mel(msA, msB)) << std::endl;
            //std::cout << std::real(mel(op, msA)) << " " << std::real(mel(msA, msB)) << std::endl;


            std::mt19937 rng;
            std::uniform_int_distribution<size_t> dist(0, A.ntensors()-1);
            for(size_t i = 0; i < 100; ++i)
            {
                size_t ind= dist(rng);
                A.set_orthogonality_centre(ind);
                msA.set_orthogonality_centre(ind);
            }

            //std::cout << std::real(mel(sop_op, A)) << " " << std::real(mel(msA, msB)) << std::endl;
            //std::cout << std::real(mel(sop_op, A)) << " " << std::real(mel(msA, msB)) << std::endl;
            //std::cout << std::real(mel(sop_op, A, B)) << " " << std::real(mel(msA, msB)) << std::endl;
            //std::cout << std::real(mel(ms_sop_op, msA, msB)) << " " << std::real(mel(msA, msB)) << std::endl;
            //std::cout << std::real(mel(op, A)) << " " << std::real(mel(msA, msB)) << std::endl;
            //std::cout << std::real(mel(op, msA)) << " " << std::real(mel(msA, msB)) << std::endl;

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


            one_site_tdvp<complex_type, backend_type> sweep(A, sop_op, 16);
            sweep.dt() = 0.001;
            sweep.coefficient() = complex_type(0, 1);
            sweep.krylov_steps() = 1;
            CALL_AND_HANDLE(sweep.prepare_environment(A, sop_op), "Failed to prepare the hamiltonian buffer for evolution.");

            for(size_t i = 0; i < 100; ++i)
            {
                auto t1 = std::chrono::high_resolution_clock::now();
                sweep(A, sop_op);
                auto t2 = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
                ttn<complex_type, backend_type> B(A);
                std::cout << (i+1)*sweep.dt() << " " << std::real(mel(op, A)) << " " << std::real( mel(sop_op, A)) << " " << std::real(mel(A, B)) << " " << duration.count() << std::endl;
            }
            


            //{
            //    using sweeping_type = sweeping_algorithm<complex_type, backend_type, ms_ttn, energy_debug_engine, sop_environment>;
            //    //typename sweeping_type::env_type env;
            //    sweeping_type mssweep(msA, ms_sop_op);
            //    CALL_AND_HANDLE(mssweep.prepare_environment(msA, ms_sop_op), "Failed to prepare the hamiltonian buffer for evolution.");
            //
            //    for(size_t i = 0; i < 1; ++i)
            //    {
            //        auto t1 = std::chrono::high_resolution_clock::now();
            //        mssweep(msA, ms_sop_op);
            //        std::cout << std::real(mel(ms_sop_op, msA)) << " " << std::real(mel(msA, msB)) << std::endl;
            //        auto t2 = std::chrono::high_resolution_clock::now();
            //        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
            //        std::cout << duration.count() << std::endl;
            //    }
            //}

            return 0;
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

}





