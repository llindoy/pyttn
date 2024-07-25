#define TIMING

#define TTNS_REGISTER_COMPLEX_DOUBLE_OPERATOR

#ifdef CEREAL_LIBRARY_FOUND
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>
#endif

#include <common/timing_macro.hpp>
#include <ttns_lib/ttns.hpp>

#include <map>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <algorithm>

#include <utils/orthopol.hpp>
#include <utils/io/input_wrapper.hpp>

using namespace ttns;

int main(int argc, char* argv[])
{
    try
    {
        using real_type = double;
        using complex_type = ttns::complex<real_type>;
        using backend_type = linalg::blas_backend;
        using size_type = typename backend_type::size_type;
        backend_type::initialise();

        //start constructing the tree topology.  
        size_type nbranch = 2;

        //determine the partitioning of the N modes into nblocks so that they all have roughly the same numbers of states.
        size_t nblocks = 1e2;
        std::vector<size_t> nmodes(nblocks);
        std::vector<size_t> mode_dimensions(nblocks+1);
        mode_dimensions[0] = 2;

        size_t seed = 0;
        size_t nspf = 2;
        size_t nspf_lower = 2;

        size_t nspf_cap = 2;
        size_t nspf_lower_cap = 2;

        for(size_t i=0; i<nblocks; ++i)
        {
            size_t nstates = 2;
            nmodes[i] = nstates;
            mode_dimensions[i+1] = nstates;
        }

        //now we build the topology tree for 
        ntree<size_type> topology{};    topology.insert(1);
        topology().insert(mode_dimensions[0]);        topology()[0].insert(mode_dimensions[0]);
        topology().insert(mode_dimensions[0]);

        size_type nlevels = static_cast<size_type>(std::log2(nmodes.size()));
        ntree_builder<size_type>::htucker_subtree(topology()[1], nmodes, nbranch, 
        [nspf, nspf_lower, nlevels](size_type l)
        {
            size_type ret = 0;
            if( l >= nlevels){ret = nspf_lower;}
            else if(l == 0){ret = nspf;}
            else
            {
                real_type rmax = std::log2(nspf);
                real_type rmin = std::log2(nspf_lower);
                ret = static_cast<size_type>(std::pow(2.0, ((nlevels-l)*static_cast<real_type>(rmax-rmin))/nlevels+rmin));
            }
            return ret;
        }
        );
        ntree_builder<size_type>::sanitise_tree(topology);

        ntree<size_type> capacity{};    capacity.insert(1);
        capacity().insert(mode_dimensions[0]);        capacity()[0].insert(mode_dimensions[0]);
        capacity().insert(mode_dimensions[0]);

        ntree_builder<size_type>::htucker_subtree(capacity()[1], nmodes, nbranch, 
        [nspf_cap, nspf_lower_cap, nlevels](size_type l)
        {
            size_type ret = 0;
            if( l >= nlevels){ret = nspf_lower_cap;}
            else if(l == 0){ret = nspf_cap;}
            else
            {
                real_type rmax = std::log2(nspf_cap);
                real_type rmin = std::log2(nspf_lower_cap);
                ret = static_cast<size_type>(std::pow(2.0, ((nlevels-l)*static_cast<real_type>(rmax-rmin))/nlevels+rmin));
            }
            return ret;
        }
        );
        ntree_builder<size_type>::sanitise_tree(capacity);
        std::cerr << "tree topology constructed" << std::endl;

        
        std::cerr << std::setprecision(16) << std::endl;
        //now we can construct our initial ttns representation of the wavefunction

        
        ms_ttn<complex_type, backend_type> msA(topology, capacity, 2);
        
        ttn<complex_type, backend_type> A(topology, capacity);  
        A.rng().seed(seed);
        A.random();
            
          /*  
           for(auto& c : A)
           {
               if(c.is_leaf())
               {
                   linalg::matrix<complex_type, backend_type> ct(c().shape(1), c().shape(0));  ct.fill_zeros();
                   ct = 1e-1*trans(c().as_matrix());
                   ct(0, 0) += 1.0;

                   std::uniform_real_distribution<real_type> dist(0, 2.0*acos(real_type(-1.0)));
                   std::normal_distribution<real_type> length_dist(0, 1);
                   for(size_type i=1; i<c().shape(1); ++i)
                   {
                       bool vector_generated = false;
               
                       while(!vector_generated)
                       {
                           vector_generated = true;
                           //generate a random vector
                           for(size_type j=0; j<c().shape(0); ++j)
                           {
                               real_type theta = dist(rng);
                               ct(i, j) += length_dist(rng)*complex_type(cos(theta), sin(theta));
                           }

                           //now we normalise it
                           ct[i] /= sqrt(dot_product(conj(ct[i]), ct[i]));

                           //now we attempt to modified gram-schmidt this
                           //if we run into linear dependence then we need to try another random vector
                           for(size_type j=0; j < i; ++j)
                           {
                               ct[i] -= dot_product(conj(ct[j]), ct[i])*ct[j];
                           }

                           //now we compute the norm of the new vector
                           real_type norm = sqrt(real(dot_product(conj(ct[i]), ct[i])));
                           if(norm > 1e-12)
                           {
                               ct[i] /= norm;
                               vector_generated = true;
                           }
                       }
                   }
                   c().as_matrix() = trans(ct);
                   
               }
               //if its an interior node fill it with the identity matrix
               else if(!c.is_root())
               {
                   linalg::matrix<complex_type, backend_type> ct(c().shape(0), c().shape(1));  ct.fill_zeros();
                   ct = 1e-1*c().as_matrix();
                   for(size_type i=0; i<c().size(0); ++i){for(size_type j=0; j<c().size(1); ++j){ct(i, j) += (i == j ? 1.0 : 0.0);}}
                   c().as_matrix() = ct;

               }
               //and fill the root node with the matrix with 1 at position 0, 0 and 0 everywhere else
               else
               {
                   for(size_type i=0; i<c().size(0); ++i){for(size_type j=0; j<c().size(1); ++j){c()(i, j) = ((i==j) ? 1.0/std::exp((i+1.0)) : 0.0);}}
               }
           }*/


        ttn<complex_type, backend_type> B(topology, capacity);  B.set_state(std::vector<size_type>(B.nmodes(), 0));
        ttn<complex_type, backend_type> Cc(topology, capacity); Cc.set_state(std::vector<size_type>(Cc.nmodes(), 0));

        std::cout << topology() << std::endl;

        linalg::matrix<complex_type, backend_type> sx(2, 2);    sx.fill_zeros();
        sx(0, 1) = 1.0; sx(1, 0) = 1.0;            
        Op<complex_type, backend_type> op(sx, {0}, {2});


        //B.random(rng);
        /*  
           for(auto& c : B)
           {
               if(c.is_leaf())
               {
                   linalg::matrix<complex_type, backend_type> ct(c().shape(1), c().shape(0));  ct.fill_zeros();
                   ct = 1e-1*trans(c().as_matrix());
                   ct(0, 0) += 1.0;

                   std::uniform_real_distribution<real_type> dist(0, 2.0*acos(real_type(-1.0)));
                   std::normal_distribution<real_type> length_dist(0, 1);
                   for(size_type i=1; i<c().shape(1); ++i)
                   {
                       bool vector_generated = false;
               
                       while(!vector_generated)
                       {
                           vector_generated = true;
                           //generate a random vector
                           for(size_type j=0; j<c().shape(0); ++j)
                           {
                               real_type theta = dist(rng);
                               ct(i, j) += length_dist(rng)*complex_type(cos(theta), sin(theta));
                           }

                           //now we normalise it
                           ct[i] /= sqrt(dot_product(conj(ct[i]), ct[i]));

                           //now we attempt to modified gram-schmidt this
                           //if we run into linear dependence then we need to try another random vector
                           for(size_type j=0; j < i; ++j)
                           {
                               ct[i] -= dot_product(conj(ct[j]), ct[i])*ct[j];
                           }

                           //now we compute the norm of the new vector
                           real_type norm = sqrt(real(dot_product(conj(ct[i]), ct[i])));
                           if(norm > 1e-12)
                           {
                               ct[i] /= norm;
                               vector_generated = true;
                           }
                       }
                   }
                   c().as_matrix() = trans(ct);
                   
               }
               //if its an interior node fill it with the identity matrix
               else if(!c.is_root())
               {
                   linalg::matrix<complex_type, backend_type> ct(c().shape(0), c().shape(1));  ct.fill_zeros();
                   ct = 1e-1*c().as_matrix();
                   for(size_type i=0; i<c().size(0); ++i){for(size_type j=0; j<c().size(1); ++j){ct(i, j) += (i == j ? 1.0 : 0.0);}}
                   c().as_matrix() = ct;

               }
               //and fill the root node with the matrix with 1 at position 0, 0 and 0 everywhere else
               else
               {
                   for(size_type i=0; i<c().size(0); ++i){for(size_type j=0; j<c().size(1); ++j){c()(i, j) = ((i==j) ? 1.0/std::exp((i+1.0)) : 0.0);}}
               }
           }*/

        matrix_element<complex_type, backend_type> mel(A);
        //A[0]() /= std::sqrt(mel(A));
        A.orthogonalise();
        B.orthogonalise();
            //A.set_is_orthogonalised();
            //
        INIT_TIMER;

        B.apply_operator(op);
        START_TIMER;
        std::cout << mel(Cc, B) << std::endl;
        STOP_TIMER("Matrix Element Evaluation");
        B.apply_operator(op);
        std::cout << mel(Cc, B) << std::endl;
        B.apply_operator(op);
        std::cout << mel(Cc, B) << std::endl;
        B.apply_operator(op);
        std::cout << mel(Cc, B) << std::endl;
        B.apply_operator(op);
        std::cout << mel(Cc, B) << std::endl;
        B.apply_operator(op);
        std::cout << mel(Cc, B) << std::endl;


        //orthogonaliser<complex_type, backend_type> ortho(A);
        //ortho(A);

        A.normalise();
        B.normalise();

        auto C = A;
        auto D = B;

        //auto l1 = A.at(8).index();
        //auto l2 = A.at(4).index();
        //std::cerr << "this" << std::endl;
        //for(auto& i : l1){std::cerr << i << " ";}
        //std::cerr << std::endl;
        //std::cerr << "that" << std::endl;
        //for(auto& i : l2){std::cerr << i << " ";}
        //std::cerr << std::endl;
        ////std::cout << A << std::endl;
        //
        //std::uniform_int_distribution<size_t> dist(1e-8, A.ntensors()-1);
        //for(size_t i = 0; i < 10000; ++i)
        //{
        //    std::cerr << i << std::endl;
        //    size_t new_orth_centre = dist(rng);
        //    A.set_orthogonality_centre(new_orth_centre);

        //    std::cerr << "current orthogonality centre: " << A.orthogonality_centre() << ": request" << new_orth_centre << std::endl;
        //    ASSERT(A.orthogonality_centre() == new_orth_centre, "Didn't move to new orthogonality centre.");
        //}


        std::cout << mel(A, true) << std::endl;
        std::cout << mel(B, true) << std::endl;
        std::cout << "ab overlap" << mel(A, B) << std::endl;

        //std::cerr << "maximum bond entropy: " << A.compute_maximum_bond_entropy() << std::endl;;
        //std::cerr << "maximum bond entropy: " << B.compute_maximum_bond_entropy() << std::endl;;
        //A.output_bond_dimensions(std::cout) << std::endl;
        A.truncate(1e-4, 16);
        B.truncate(1e-4, 16);
        A.normalise();
        B.normalise();

        //std::cerr << "maximum bond entropy: " << A.compute_maximum_bond_entropy() << std::endl;;
        //std::cerr << "maximum bond entropy: " << B.compute_maximum_bond_entropy() << std::endl;;
        //A.output_bond_dimensions(std::cout) << std::endl;

        std::cout << mel(A, true) << std::endl;
        std::cout << mel(B, true) << std::endl;
        std::cout << "ac overlap" << mel(A, C) << std::endl;
        std::cout << "bd overlap" << mel(B, D) << std::endl;
        
        /*  
        {
            linalg::tensor<double, 3> Am(2,3,4 ,[](size_t i , size_t j, size_t k){return i*12+j*4+k;});
            std::cout << Am << std::endl;
            linalg::tensor<double, 3> Bm = linalg::trans(Am, {1, 2, 0});
            std::cout << Bm << std::endl;
        }

        {
            linalg::tensor<double, 5> Am(3,3,3,3,3);
            linalg::tensor<double, 4> Bm(3,3,3,3);

            for(size_t i = 0; i < Am.size(); ++i)
            {
                Am.buffer()[i] = i;
            }

            for(size_t i = 0; i < Bm.size(); ++i)
            {
                Bm.buffer()[i] = i;
            }
            std::cout << Am  << std::endl;
            std::cout << Bm  << std::endl;

            auto expr = linalg::tensordot(Am, Bm, std::array<int,3>{{0, 3, 1}}, std::array<int, 3>{{3, 2, 1}});
            decltype(expr)::result_type Cm = expr;
            std::cout << Cm << std::endl;
        }

        size_type nd = 2;
        size_type Di = nd*nd*nd*nd*nd*nd;
        linalg::matrix<double> M(Di,Di, [Di](size_t i, size_t j){return i*Di+j;});

        Op<double> op(M, {0, 1, 2, 3, 4, 5}, {nd,nd,nd,nd, nd, nd});

        auto mpo = op.as_mpo();
        //auto expr = linalg::tensordot(mpo[0], mpo[1], std::array<int,1>{{3}}, std::array<int, 1>{{0}});
        //decltype(expr)::result_type Mres = expr;
        //decltype(expr)::result_type Mres2 = linalg::trans(Mres, {0, 1, 3, 2, 4, 5});
        //std::cout << Mres2 << std::endl;*/
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

}




