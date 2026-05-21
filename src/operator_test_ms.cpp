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
#include <ttns_lib/sop/multiset_SOP.hpp>
#include <ttns_lib/sop/system_information.hpp>

#include <utils/io/input_wrapper.hpp>

#include <ttns_lib/operators/multiset_sop_operator.hpp>

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
            multiset_SOP<complex_type> H(2, N-1);//(nimp*nimp*nimp*nimp + nimp*N*2);
            //sop.reserve(nimp*nimp*nimp*nimp + nimp*N*2);
            //add on the impurity interaction terms
            real_type h = 1;
            real_type J = 1;
            H(0, 1) += -1.0 * h;
            H(1, 0) += -1.0 * h;

            H(0, 0) += -1.0 * J * sOP("sxz", 0);
            H(1, 1) += 1.0 * J * sOP("sxz", 0);

            for(size_t i = 0; i < N-1; ++i)
            {
                H(0, 0) += -1.0 * h * sOP("sxx", i);
                H(1, 1) += -1.0 * h * sOP("sxx", i);
            }
            for(size_t i = 0; i < N-2; ++i)
            {
                H(0, 0) += -1.0 * J * sOP("sxz", i) * sOP("sxz", i + 1);
                H(1, 1) += -1.0 * J * sOP("sxz", i) * sOP("sxz", i + 1);
            }
            
            STOP_TIMER("SOP built");
            operator_dictionary<complex_type, backend_type> opdict(N-1);
            linalg::matrix<complex_type> sx(2, 2);
            sx(0,1) = 1.0;
            sx(1,0) = 1.0;
            ops::dense_matrix_operator<complex_type, backend_type> _sxx(sx);
            linalg::matrix<complex_type> sz(2, 2);
            sz(0,0) = 1.0;
            sz(1,1) =-1.0;
            ops::dense_matrix_operator<complex_type, backend_type> _sxz(sz);

            
            for(size_t i = 0; i < N-1; ++i)
            {
                opdict.insert(i, std::string("sxx"), site_operator<complex_type, backend_type>(_sxx, i));
                opdict.insert(i, std::string("sxz"), site_operator<complex_type, backend_type>(_sxz, i));
            }

            std::vector<size_t> dims(N-1);  std::fill(dims.begin(), dims.end(), 2);

            ntree<size_t> topology = ntree_builder<size_t>::htucker_tree(dims, 2, 2);
            ntree<size_t> capacity = ntree_builder<size_t>::htucker_tree(dims, 2, 16);

            system_modes inf(N-1);
            for(size_t i = 0; i < inf.nmodes(); ++i)
            {
                inf[i] = spin_mode(2);
            }

            ntree_builder<size_t>::sanitise_tree(topology, false);
            ntree_builder<size_t>::sanitise_tree(capacity, false);

            std::cout << topology << std::endl;
            std::cout << capacity << std::endl;
            ms_ttn<complex_type, backend_type> A(2, topology, capacity);      A.random();

            multiset_sop_operator<complex_type, backend_type> sop_op(H, A, inf, opdict);
           
            return 0;
        }
    }
    catch(const std::exception& ex)
    {
        logging::error(ex.what());
        return 1;
    }

}





