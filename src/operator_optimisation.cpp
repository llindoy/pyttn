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
#include <ttns_lib/sop/compressedSOP.hpp>
#include <ttns_lib/sop/system_information.hpp>

#include <utils/io/input_wrapper.hpp>

#include <ttns_lib/operators/sop_operator.hpp>
#include <ttns_lib/sweeping_algorithm/tdvp.hpp>

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

        size_t nimp = std::atoi(argv[1]);
        size_t N = std::atoi(argv[2]);
        size_t nbranch = std::atoi(argv[3]);
        size_t option = std::atoi(argv[4]);
        size_t option2 = std::atoi(argv[5]);
        bool compute_energy = option == 0;
        bool use_id_op = option2 == 0;
        INIT_TIMER;
        std::mt19937 rng;
        std::uniform_real_distribution<real_type> dist(0, 1);

        {
            START_TIMER;
            SOP<complex_type> sop(nimp+N);//(nimp*nimp*nimp*nimp + nimp*N*2);
            //sop.reserve(nimp*nimp*nimp*nimp + nimp*N*2);
            //add on the impurity interaction terms
            size_t count = 0;
            for(size_t i = 0; i < nimp; ++i)
            {
                for(size_t j = i; j < nimp; ++j)
                {
                    real_type coeff = dist(rng);
                    sop += coeff*fermion_operator("cdag", i)*fermion_operator("c", j);
                    if(i != j)
                    {
                        sop += coeff*fermion_operator("cdag", j)*fermion_operator("c", i);
                    }
                    ++count;
                }
                for(size_t j = i+1; j < nimp; ++j)
                {
                    ++count;
                    for(size_t k = 0; k < nimp; ++k)
                    {
                        for(size_t l = k+1; l < nimp; ++l)
                        {
                            real_type coeff = dist(rng);
                            sop += coeff*fermion_operator("cdag", i)*fermion_operator("cdag", j)*fermion_operator("c", k)*fermion_operator("c", l);
                            ++count;
                        }    
                    }
                }
            }

            //add on the one-body terms
            for(size_t i = 0; i < nimp; ++i)
            {
                for(size_t j=nimp; j< N+nimp; ++j)
                {
                    sop += (j+5.0)*fermion_operator("cdag", i)*fermion_operator("c", j);
                    sop += (j+5.0)*fermion_operator("cdag", j)*fermion_operator("c", i);
                    sop += (j+5)*10.0*fermion_operator("n", j);
                }
            }
            STOP_TIMER("SOP built");
              
            START_TIMER;
            STOP_TIMER("jordan_wigner");

            std::vector<size_t> dims(N+nimp);  std::fill(dims.begin(), dims.end(), 2);


            ntree<size_t> topology = ntree_builder<size_t>::htucker_tree(dims, nbranch, 16);

            system_modes inf(N+nimp);
            for(size_t i = 0; i < inf.nmodes(); ++i)
            {
                inf[i] = fermion_mode();
            }

            ntree_builder<size_t>::sanitise_tree(topology, false);
            std::cout << topology << std::endl;
            std::vector<size_t> ones(N+nimp);   std::fill(ones.begin(), ones.end(), 1);
            ttn<complex_type, backend_type> A(topology);      A.set_state(ones);

            sop_operator<complex_type, backend_type> sop_op(sop, A, inf);
            one_site_tdvp<complex_type, backend_type> sweep(A, sop_op);
            sweep.dt() = 0.01;
            CALL_AND_HANDLE(sweep.prepare_environment(A, sop_op), "Failed to prepare the hamiltonian buffer for evolution.");

            for(size_t i = 0; i < 10; ++i)
            {
                auto t1 = std::chrono::high_resolution_clock::now();
                sweep(A, sop_op);
                auto t2 = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
                std::cout << "Time: " << duration.count() << std::endl;
            }

            //aSOP.primitive(sop, A);
            //energy_est_test(A, dims, aSOP, use_id_op, compute_energy);
            //aSOP.construct_sop_tree(use_id_op);

            //aSOP.compressed(sop, A);
            //energy_est_test(A, dims, aSOP, use_id_op, compute_energy);
            //aSOP.construct_sop_tree(use_id_op);



            return 0;
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

}





