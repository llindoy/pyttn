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
#include <ttns_lib/sweeping_algorithm/dmrg.hpp>

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

int main(int /*argc*/, char* /*argv*/[])
{
    try
    {
        using real_type = double;
        using complex_type = std::complex<double>;
        using backend_type = linalg::blas_backend;
        using namespace utils;
        backend_type::initialise();

        INIT_TIMER;
        std::mt19937 rng;
        std::uniform_real_distribution<real_type> dist(0, 1);

        {
            START_TIMER;
            size_t N = 16;
            SOP<complex_type> sop(N);//(nimp*nimp*nimp*nimp + nimp*N*2);
            //sop.reserve(nimp*nimp*nimp*nimp + nimp*N*2);
            //add on the impurity interaction terms
            for(size_t i = 0; i < N; ++i)
            {
                sop += -1.0*sOP("sx", i);
            }
            for(size_t i = 0; i < N-1; ++i)
            {
                sop += -1.0*sOP("sz", i)*sOP("sz", i+1);
            }
            STOP_TIMER("SOP built");
              
            START_TIMER;
            STOP_TIMER("jordan_wigner");

            std::vector<size_t> dims(N);  std::fill(dims.begin(), dims.end(), 2);

            ntree<size_t> topology = ntree_builder<size_t>::htucker_tree(dims, 2, 2);
            ntree<size_t> capacity = ntree_builder<size_t>::htucker_tree(dims, 2, 16);

            system_modes inf(N);
            for(size_t i = 0; i < inf.nmodes(); ++i)
            {
                inf[i] = spin_mode(2);
            }

            ntree_builder<size_t>::sanitise_tree(topology, false);
            ntree_builder<size_t>::sanitise_tree(capacity, false);

            std::cout << topology << std::endl;
            std::cout << capacity << std::endl;
            std::vector<size_t> ones(N);   std::fill(ones.begin(), ones.end(), 1);
            ttn<complex_type, backend_type> A(topology, capacity);      A.set_state(ones);
            

            sop_operator<complex_type, backend_type> sop_op(sop, A, inf);
            adaptive_one_site_dmrg<complex_type, backend_type> sweep(A, sop_op);
            sweep.spawning_threshold() = 1e-5;
            sweep.unoccupied_threshold() = 1e-5;
            sweep.minimum_unoccupied() = 3;
            CALL_AND_HANDLE(sweep.prepare_environment(A, sop_op), "Failed to prepare the hamiltonian buffer for evolution.");

            for(size_t i = 0; i < 10; ++i)
            {
                auto t1 = std::chrono::high_resolution_clock::now();
                sweep(A, sop_op);
                auto t2 = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
                std::cout << "Time: " << duration.count() << std::endl;
            }
            linalg::matrix<complex_type, backend_type> mat(4,4, [](size_t i, size_t j){return i*4.0+j;});
            //mat(0,3) = 1.0/4;
            //mat(1,2) = 1.0/4;
            //mat(2,1) = 1.0/4;
            //mat(3,0) = 1.0/4;

            std::vector<size_t> inds(2);
            std::vector<size_t> ldims(2);
            inds[0] = 0;
            inds[1] = 15;
            ldims[0] = 2;
            ldims[1] = 2;

            Op<complex_type, backend_type> op(mat, inds, ldims);

            auto op_mpo = op.as_mpo();
            auto C = op_mpo[0].reinterpret_shape(op_mpo[0].shape(1), op_mpo[0].shape(2), op_mpo[0].shape(3));
            auto D = op_mpo[1].reinterpret_shape(op_mpo[1].shape(0), op_mpo[1].shape(1), op_mpo[1].shape(2));   

            std::cerr << op_mpo[0].shape(0) << " " <<  op_mpo[0].shape(1) << " " <<  op_mpo[0].shape(2)   << " " <<  op_mpo[0].shape(3) << std::endl;       
            std::cerr << op_mpo[1].shape(0) << " " <<  op_mpo[1].shape(1) << " " <<  op_mpo[1].shape(2)   << " " <<  op_mpo[1].shape(3) << std::endl;       

           
            std::cerr << "C: " <<  C << std::endl;
            std::cerr << "D: " << D << std::endl;
            linalg::tensor<complex_type, 4> ret = linalg::tensordot(C, D, std::array<int, 1>{{2}}, std::array<int, 1>{{0}});
            linalg::tensor<complex_type, 4> rt = linalg::transpose(ret, {0, 2, 1, 3});
            linalg::matrix<complex_type> rmat = rt.reinterpret_shape(4, 4);
            std::cerr << "ret: " << rt << std::endl;
            std::cerr << "ret: " << rmat << std::endl;

            ttn<complex_type, backend_type> B(A);    
            ttn<complex_type, backend_type> E(A);    

            A.apply_operator(op, 1e-8);
            E.apply_operator(op, 1e-8, 0, true);

            matrix_element<complex_type, backend_type> mel(A);

            product_operator<complex_type, backend_type> pop(sOP("sx", inds[0]) * sOP("sx", inds[1]), inf);
            std::cout << mel(A, B) << " " << mel(pop, B) << " " << mel(E, B) << " " << mel(A, A) << std::endl;

            return 0;
        }
    }
    catch(const std::exception& ex)
    {
        logging::error(ex.what());
        return 1;
    }

}





