//#define TIMING 0
//#define USE_OLD
#define TTNS_REGISTER_COMPLEX_DOUBLE_OPERATOR

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>
#endif

#include <ttns_lib/ttns.hpp>
#include <ttns_lib/ttn/sop_ttn_contraction.hpp>

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
#include <ttns_lib/sop/toDense.hpp>

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


template <typename T>
void test_1( std::string label)
{
    system_modes sysinf(1);
    sysinf[0] = qubit_mode();

    sOP H = sOP(label, 0);
    site_operator<T> siteop = site_operator<T>(H, sysinf);

    auto mat = siteop.todense();
    std::cout << label << std::endl;
    std::cout << mat << std::endl;
}

template <typename T>
void test_2(std::string label, size_t mi)
{
    mi = mi % 3;
    system_modes sysinf(1);
    sysinf[0].append(qubit_mode());
    sysinf[0].append(qubit_mode());
    sysinf[0].append(qubit_mode());
    sOP H = sOP(label, mi);

    site_operator<T> siteop = site_operator<T>(H, sysinf);

    auto mat = siteop.todense();
    std::cout << label << std::endl;
    std::cout << mat << std::endl;
}

template <typename T>
void test_1b( std::string label)
{
    system_modes sysinf(1);
    sysinf[0] = qubit_mode();

    sOP H = sOP(label, 0);
    linalg::matrix<T> mat;
    convert_to_dense(H, sysinf, mat);
    //site_operator<T> siteop = site_operator<T>(H, sysinf);

    //auto mat = siteop.todense();
    std::cout << label << std::endl;
    std::cout << mat << std::endl;
}

template <typename T>
void test_2b(std::string label, size_t mi)
{
    mi = mi % 3;
    system_modes sysinf(1);
    sysinf[0].append(qubit_mode());
    sysinf[0].append(qubit_mode());
    sysinf[0].append(qubit_mode());
    sOP H = sOP(label, mi);
    linalg::matrix<T> mat;
    convert_to_dense(H, sysinf, mat);
    //site_operator<T> siteop = site_operator<T>(H, sysinf);

    //auto mat = siteop.todense();
    std::cout << label << std::endl;
    std::cout << mat << std::endl;
}


template <typename T>
void test_3( std::string label)
{
    system_modes sysinf(1);
    sysinf[0].append(qubit_mode());
    sysinf[0].append(qubit_mode());
    sysinf[0].append(qubit_mode());

    sPOP H = sOP(label, 0)*sOP(label, 2);
    linalg::matrix<T> mat;
    convert_to_dense(H, sysinf, mat);
    //site_operator<T> siteop = site_operator<T>(H, sysinf);

    //auto mat = siteop.todense();
    std::cout << label << std::endl;
    std::cout << mat << std::endl;
}

int main(int argc, char* argv[])
{
    using real_type = double;
    using complex_type = linalg::complex<double>;
    using backend_type = linalg::blas_backend;
    using namespace utils;
    backend_type::initialise();

    test_3<complex_type>(std::string("s+"));
    test_3<complex_type>(std::string("s-"));
    test_3<complex_type>(std::string("sx"));
    test_3<complex_type>(std::string("sy"));
    test_3<complex_type>(std::string("sz"));


    test_3<complex_type>(std::string("|0><0|"));
    test_3<complex_type>(std::string("|0><1|"));
    test_3<complex_type>(std::string("|1><0|"));
    test_3<complex_type>(std::string("|1><1|"));
    

    //for (size_t i = 0; i < 3; ++i)
    //{
    //    test_2b<complex_type>(std::string("s+"), i);
    //    test_2b<complex_type>(std::string("s-"), i);
    //    test_2b<complex_type>(std::string("sx"), i);
    //    test_2b<complex_type>(std::string("sy"), i);
    //    test_2b<complex_type>(std::string("sz"), i); 
    //}
    
}


