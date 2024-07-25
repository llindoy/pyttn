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


#include <utils/io/input_wrapper.hpp>

#include <ttns_lib/sop/autoSOP.hpp>
#include <ttns_lib/sweeping_algorithm/environment/sum_of_product_operator_env.hpp>
#include <ttns_lib/sweeping_algorithm/sweeping_algorithm.hpp>

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

/*  
template <typename T, typename U, typename B>
void energy_est_test(const ttn<T, B>& in, const std::vector<size_t>& dims, autoSOP<U>& aSOP, bool use_id_op, bool compute_energy = false)
{
    const auto& site_operators = aSOP.site_operators();
    size_t N = site_operators.size();
    std::vector<size_t> terms_per_mode(site_operators.size());

    for(size_t i = 0; i < site_operators.size(); ++i)
    {
        terms_per_mode[i] = site_operators[i].size();
    }
    std::shared_ptr<utils::occupation_number_basis> bose_basis = std::make_shared<utils::direct_product_occupation_number_basis>(15, 1);
    std::shared_ptr<utils::occupation_number_basis> basis = std::make_shared<utils::direct_product_occupation_number_basis>(2, 1);
    
    using bose_dict = boson::default_boson_operator_dictionary<T>;
    using spin_dict = spin::default_spin_operator_dictionary<T>;

    sop_operator<T, B> H(N, dims, terms_per_mode);
    for(size_t nu = 0; nu < site_operators.size(); ++nu)
    {
        for(size_t j = 0; j < site_operators[nu].size(); ++j)
        {
            
            sPOP t = site_operators[nu][j];
            for(const auto& op : t.ops())
            {
                ASSERT(op.mode() == nu, "Invalid site operators.");
            }

            linalg::matrix<T, B> M;
            if(t.size() == 1)
            {
                std::string label = t.ops().front().op();
                if(label == "id")
                {
                    H.bind(ops::identity<T, B>{dims[nu]}, nu);
                }
                else
                {
                    if(nu == 0)
                    {
                        spin_dict::query(label)->as_dense(basis, 0, M);
                    }
                    else
                    {
                        bose_dict::query(label)->as_dense(bose_basis, 0, M);
                    }
                    H.bind(ops::dense_matrix_operator<T, B>{M}, nu);
                }
            }
            else
            {
                std::vector<std::shared_ptr<ops::primitive<T, B>>> ops;   ops.reserve(t.size());
                for(const auto& op : t.ops())
                {
                    std::string label = op.op();
                    if(label == "id")
                    {
                        ops.push_back(std::make_shared<ops::identity<T, B>>(dims[nu]));
                    }
                    else
                    {
                        if(nu == 0)
                        {
                            spin_dict::query(label)->as_dense(basis, 0, M);
                        }
                        else
                        {
                            bose_dict::query(label)->as_dense(bose_basis, 0, M);
                        }
                        ops.push_back(std::make_shared<ops::dense_matrix_operator<T, B>>(M));
                    }
                }
                H.bind(ops::sequential_product_operator<T, B>{ops}, nu);
            }
        }
    }

    ttn<T, B> A(in);
    A.normalise();

    std::cerr << "setting up hamiltonian." << std::endl;
    operator_container<T, B> hbuf;
    std::cout << "mem: " << hbuf.compute_memory_requirements(A, aSOP.operator_representation(), use_id_op) << std::endl;

    if(compute_energy)
    {
        hbuf.resize(A, aSOP.operator_representation(), use_id_op);
        aSOP.clear();

        sweeping_algorithm<T, B, trivial_update, sop_environment> sweep(A, hbuf.op());
        CALL_AND_HANDLE(sweep.prepare_environment(A, H, hbuf.op()), "Failed to prepare the hamiltonian buffer for evolution.");



        std::cout << "Energy: " << hbuf.e() << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();

        for(size_t i = 0; i < 1; ++i)
        {
            sweep(A, H, hbuf.op());
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << "Energy: " << hbuf.e() << std::endl;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);
        std::cout << "Time: " << duration.count() << std::endl;
    }

}*/

int main(int argc, char* argv[])
{
    try
    {
        using real_type = double;
        using complex_type = linalg::complex<double>;
        using backend_type = linalg::blas_backend;
        using namespace utils;
        backend_type::initialise();

        if(argc < 3)
        {
            std::cerr << argv[0] << " <input filename>" << std::endl;
            return 1;
        }

        size_t N = std::atoi(argv[1]);
        size_t nbranch = std::atoi(argv[2]);
        size_t nspf_lower = 8;
        size_t nspf = 16;

        INIT_TIMER;
        std::mt19937 rng;
        std::uniform_real_distribution<real_type> dist(0, 1);

        std::vector<real_type> g(N);
        std::vector<real_type> w(N);

        real_type delta = 1.0;
        for(size_t i = 0; i < N; ++i)
        {
            g[i] = dist(rng);
            w[i] = dist(rng);
        }

        {
            START_TIMER;
            SOP<real_type> sop(N+1);            
            //add on the impurity interaction terms
            size_t count = 0;

            sop += delta*sOP("Sx", 0);
            for(size_t i = 0; i < N; ++i)
            {
                sop += g[i]*sOP("Sz", 0)*sOP("q", i+1);
                sop += w[i]*sOP("n", i+1);
            }

            STOP_TIMER("SOP built");
              
            START_TIMER;
            STOP_TIMER("jordan_wigner");

            std::vector<size_t> dims(N+1);  dims[0] = 2; std::fill(dims.begin()+1, dims.end(), 15);
            std::vector<size_t> nmodes(N);  std::fill(nmodes.begin(), nmodes.end(), 15);



            //now we build the topology tree for 
            ntree<size_t> topology{};    topology.insert(1);
            topology().insert(dims[0]);        topology()[0].insert(dims[0]);
            topology().insert(dims[0]);

            size_t nlevels = static_cast<size_t>(std::log2(nmodes.size()));
            ntree_builder<size_t>::htucker_subtree(topology()[1], nmodes, nbranch, 
            [nspf, nspf_lower, nlevels](size_t l)
            {
                size_t ret = 0;
                if( l >= nlevels){ret = nspf_lower;}
                else if(l == 0){ret = nspf;}
                else
                {
                    real_type rmax = std::log2(nspf);
                    real_type rmin = std::log2(nspf_lower);
                    ret = static_cast<size_t>(std::pow(2.0, ((nlevels-l)*static_cast<real_type>(rmax-rmin))/nlevels+rmin));
                }
                return ret;
            }
            );
            ntree_builder<size_t>::sanitise_tree(topology);
      

            ntree_builder<size_t>::sanitise_tree(topology, false);
            std::cout << topology << std::endl;
            std::vector<size_t> ones(N+1);   std::fill(ones.begin(), ones.end(), 0);
            ttn<complex_type, backend_type> A(topology);      A.set_state(ones);

            autoSOP<real_type> aSOP;

            //aSOP.primitive(sop, A);
            //energy_est_test(A, dims, aSOP, use_id_op, compute_energy);
            //aSOP.construct_sop_tree(use_id_op);

            aSOP.compressed(sop, A);
            energy_est_test(A, dims, aSOP, true, true);
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




