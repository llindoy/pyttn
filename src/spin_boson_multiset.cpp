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

        if(argc < 10)
        {
            std::cerr << argv[0] << "<N> <alpha> <wc> <s> <beta> <nchi> <nbranch> <nbose> <nthreads>" << std::endl;
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
        size_t nthreads = std::atoi(argv[9]);
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
                    w[i] = -wc*std::log(1.0-(i+1.0)/(N+1.));
                    g[i] = std::sqrt(2*alpha*wc*w[i]/(N+1.0));
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

            real_type eps = 0.0;
            real_type delta = 1.0;

            SOP<complex_type> sop(Nb);
            system_modes sysinf(Nb);
            for(size_t i = 0; i < Nb; ++i)
            {
                sysinf[i] = boson_mode(mdim);
            }

            multiset_SOP<complex_type> msSOP(2, Nb);
            msSOP(0, 0) = SOP<complex_type>(Nb);
            msSOP(0, 1) = SOP<complex_type>(Nb);
            msSOP(1, 0) = SOP<complex_type>(Nb); 
            msSOP(1, 1) = SOP<complex_type>(Nb); 

            //set up the system terms
            msSOP(0, 0) += eps;
            msSOP(0, 1) += ttns::literal::coeff<complex_type>([delta](real_type t){return (std::fmod(t , 0.2)< 0.1 ? 0.0 : delta);});
            msSOP(1, 0) += ttns::literal::coeff<complex_type>([delta](real_type t){return (std::fmod(t , 0.2)< 0.1 ? 0.0 : delta);});
            msSOP(1, 1) -= eps;

            //now add on the system bath and bath terms
            for(size_t i =0; i < Nb; ++i)
            {
                complex_type gi = g[i];
                msSOP(0,0) += ttns::literal::coeff<complex_type>([gi](real_type t){return (std::fmod(t , 0.2)< 0.1 ? 0.0 : 0.5*std::sqrt(2.0)*gi);}) * sOP("q", i);   
                msSOP(0,0) += w[i]*sOP("n", i);
                msSOP(1,1) -= ttns::literal::coeff<complex_type>([gi](real_type t){return (std::fmod(t , 0.2)< 0.1 ? 0.0 : 0.5*std::sqrt(2.0)*gi);}) * sOP("q", i);   
                msSOP(1,1) += w[i]*sOP("n", i);
            }

            std::vector<size_t> dims(Nb);
            for(size_t i = 0; i < Nb; ++i)
            {
                dims[i] = mdim;
            }
            ntree<size_t> topology = ntree_builder<size_t>::htucker_tree(dims, nbranch, nchi);
            ntree_builder<size_t>::sanitise_tree(topology, false);

            std::cout << topology << std::endl;
            std::vector<size_t> zeros(Nb);   std::fill(zeros.begin(), zeros.end(), 0);

            std::vector<complex_type> coeff(2);
            coeff[0] = 1.0;
            coeff[1] = 0.0;
            std::vector<std::vector<size_t>> msState(2);    msState[0] = zeros; msState[1]=zeros;


            ttn<complex_type, backend_type> A(topology);      
            ttn<complex_type, backend_type> B(topology);      

            ms_ttn<complex_type, backend_type> msA(topology, 2);      
            msA.set_state(coeff, msState);

            sop_operator<complex_type, backend_type> sop_op(sop, A, sysinf);
            multiset_sop_operator<complex_type, backend_type> ms_sop_op(msSOP, msA, sysinf);

            matrix_element<complex_type> mel(msA, ms_sop_op);
            ms_ttn<complex_type, backend_type>msB(msA);
            std::cout << std::setprecision(16);
            std::cout << mel(msA, msB) << std::endl;

            {
                using sweeping_type = multiset_one_site_tdvp<complex_type, backend_type>;
                //typename sweeping_type::env_type env;
                sweeping_type mssweep(msA, ms_sop_op, 8, 1, nthreads);
                mssweep.dt() = 0.0025;
                mssweep.coefficient() = complex_type(0, 1);
                //mssweep.krylov_steps() = 1;
                CALL_AND_HANDLE(mssweep.prepare_environment(msA, ms_sop_op), "Failed to prepare the hamiltonian buffer for evolution.");
            
                A = msA.slice(0);
                B = msA.slice(1);
                std::cout << 0.0 << " " << std::real(mel(A)-mel(B)) << std::endl;
                for(size_t i = 0; i < 1000; ++i)
                {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    mssweep(msA, ms_sop_op);
                    A = msA.slice(0);
                    B = msA.slice(1);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
                    std::cout << (i+1.0)*mssweep.dt() << " " << std::real(mel(A)-mel(B)) << " " << duration.count() << std::endl;
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





