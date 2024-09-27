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

        if(argc < 9)
        {
            std::cerr << argv[0] << " <N> <alpha> <wc> <s> <beta> <nchi> <nbranch> <nbose>" << std::endl;
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
        INIT_TIMER;

        {
            START_TIMER;

            size_t Nb = N;
            std::vector<complex_type> g(Nb);
            std::vector<real_type> E(Nb);
            std::vector<real_type> V(Nb);
            std::vector<real_type> M(Nb);

            for(size_t i = 0; i < N; ++i)
            {
                if(beta < 0)
                {
                    E[i] = -wc*std::log(1.0-(i+1)/(N+1.));
                    g[i] = std::sqrt(2*alpha*wc*wc*std::pow(E[i]/wc, s)/(N+1.0));
                    V[i]  = 1.0;
                    M[i]  = 1.0;
                }
            }

            real_type delta = 2.0;


            SOP<complex_type> sop(1+2*Nb);
            system_modes sysinf(1+Nb);
            sysinf[0] = spin_mode(2);
            for(size_t i = 0; i < Nb; ++i)
            {
                std::vector<primitive_mode_data> modes(2);
                modes[0] = boson_mode(mdim);
                modes[1] = boson_mode(mdim);
                sysinf[i+1] = modes;
            }

            sop +=  delta*sOP("sx", 0);
            for(size_t i = 0; i < Nb; ++i)
            {
                size_t i1 = 2*i+1;
                size_t i2 = 2*i+2;
                sop += E[i]*sOP("n", i1) - E[i]*sOP("n", i2);
                sop += complex_type(0, 2)*g[i]*sOP("a", i1)*sOP("a", i2) - complex_type(0, 1)*g[i] *sOP("n", i1) - complex_type(0, 1)*g[i]*sOP("n", i2);
                sop += V[i]*sOP("sz", 0)*sOP("adag", i1) + V[i]*sOP("sz", 0)*sOP("a", i1) - V[i]*sOP("sy", 0) *sOP("adag", i2) - V[i]*sOP("sy", 0)*sOP("a", i2);
                sop += complex_type(0, 2)*M[i]*sOP("sy", 0)*sOP("a", i1) - complex_type(0, 1)*M[i]*sOP("sz", 0)*sOP("a", i1) - complex_type(0, 1)*M[i]*sOP("sy",0)*sOP("adag", i2);
                sop += complex_type(0, 2)*M[i]*sOP("sz", 0)*sOP("a", i2) - complex_type(0, 1)*M[i]*sOP("sz", 0)*sOP("adag", i1) - complex_type(0, 1)*M[i]*sOP("sy",0)*sOP("a", i2);
            }

            std::vector<size_t> dims(Nb+1);
            std::vector<size_t> bdims(Nb);
            dims[0] = 2;
            for(size_t i = 0; i < Nb; ++i)
            {
                dims[i+1] = mdim*mdim;
                bdims[i] = mdim*mdim;
            }
            ntree<size_t> topology;
            topology.insert(1);
            topology().insert(2); topology()[0].insert(2);
            topology().insert(2);
            ntree_builder<size_t>::htucker_subtree(topology()[1], bdims, nbranch, nchi);
            ntree_builder<size_t>::sanitise_tree(topology, false);

            ntree<size_t> capacity;
            capacity.insert(1);
            capacity().insert(2); capacity()[0].insert(2);
            capacity().insert(2);
            ntree_builder<size_t>::htucker_subtree(capacity()[1], bdims, nbranch, mdim);
            ntree_builder<size_t>::sanitise_tree(capacity, false);

            std::cout << topology << std::endl;
            std::vector<size_t> zeros(Nb+1);   std::fill(zeros.begin(), zeros.end(), 0);

            ttn<complex_type, backend_type> A(topology, capacity);      A.set_state(zeros);

            sop_operator<complex_type, backend_type> sop_op(sop, A, sysinf);
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

}





