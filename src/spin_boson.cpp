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

        if(argc < 9)
        {
            std::cerr << argv[0] << "<N> <alpha> <wc> <s> <beta> <nchi> <nbranch> <nbose> <eps>" << std::endl;
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

            sop += ttns::literal::coeff<complex_type>([eps](real_type t){return eps*std::cos(10*t);})*sOP("sz", 0) + ttns::literal::coeff<complex_type>([delta](real_type t){return (std::fmod(t , 0.2)< 0.1 ? 0.0 : delta);})*sOP("sx", 0);
            for(size_t i = 0; i < Nb; ++i)
            {
                complex_type gi = g[i];
                sop +=  ttns::literal::coeff<complex_type>([gi](real_type t){return (std::fmod(t , 0.2)< 0.1 ? 0.0 : std::sqrt(2.0)*gi);})*(sOP("sz", 0)*sOP("q", i+1));
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

            std::cout << topology << std::endl;
            std::vector<size_t> zeros(Nb+1);   std::fill(zeros.begin(), zeros.end(), 0);

            ttn<complex_type, backend_type> A(topology);      A.set_state(zeros);

            sop_operator<complex_type, backend_type> sop_op(sop, A, sysinf);

            sOP sz = sOP("sz", 0);
            site_operator<complex_type> op(sz, sysinf);

            //matrix_element<complex_type> mel;//(msA, ms_sop_op);
            matrix_element<complex_type> mel;//(A, sop_op);
            mel.resize(A, sop_op);

            one_site_tdvp<complex_type, backend_type> sweep(A, sop_op, 16);
            sweep.dt() = 0.0025;
            sweep.coefficient() = complex_type(0, 1);
            sweep.krylov_steps() = 1;
            sweep.use_time_dependent_hamiltonian() = true;
            CALL_AND_HANDLE(sweep.prepare_environment(A, sop_op), "Failed to prepare the hamiltonian buffer for evolution.");

            std::cout << std::setprecision(16);
            std::cout << 0 << " " << std::real(mel(op, A)) << std::endl;
            for(size_t i = 0; i < 1000; ++i)
            {
                auto t1 = std::chrono::high_resolution_clock::now();
                sweep(A, sop_op);
                auto t2 = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
                std::cout << (i+1)*sweep.dt() << " " << std::real(mel(op, A)) << " " << duration.count() << std::endl;
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





