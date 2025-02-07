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

        INIT_TIMER;
        std::mt19937 rng;
        std::uniform_real_distribution<real_type> dist(0, 1);

        size_t Nmodes = 24;


        size_t ng1 = 5;
        std::vector<size_t> g1(ng1);
        linalg::vector<double> ai(ng1, [&rng, &dist](size_t ){return dist(rng);});
        linalg::vector<double> bi(ng1, [&rng, &dist](size_t ){return dist(rng);});

        std::vector<std::vector<size_t>> g2(8);
        g2[0] = {1, 2, 3, 4, 5};
        g2[1] = {11, 19};
        g2[2] = {0};
        g2[3] = {8, 9};
        g2[4] = {6, 10, 7, 23};
        g2[5] = {12, 16, 18, 13};
        g2[6] = {15, 17, 14, 20};
        g2[7] = {21, 22};

        std::vector<linalg::matrix<double>> aij(g2.size());
        std::vector<linalg::matrix<double>> bij(g2.size());
        for(size_t i = 0; i < g2.size(); ++i)
        {
            aij[i] = linalg::matrix<double>(g2.size(), g2.size(), [&rng, &dist](size_t, size_t){return dist(rng);});
            bij[i] = linalg::matrix<double>(g2.size(), g2.size(), [&rng, &dist](size_t, size_t){return dist(rng);});
        }

        size_t ng3 = 1;
        std::vector<size_t> g3(ng3);
        linalg::vector<double> ci(ng3, [&rng, &dist](size_t ){return dist(rng);});

        std::vector<std::vector<size_t>> g4i(4);
        std::vector<std::vector<size_t>> g4j(4);
        g4i[0] = {1, 2, 3, 4, 5};
        g4j[0] = {0};
        
        g4i[1] = {6, 10, 7, 23};
        g4j[1] = {8, 9};

        g4i[2] = {12, 16, 18, 13};
        g4j[2] = {11, 19};

        g4i[2] = {15, 17, 14, 20};
        g4j[2] = {21, 22};

        std::vector<linalg::matrix<double>> cij(g4i.size());
        for(size_t i = 0; i < g4i.size(); ++i)
        {
            cij[i] = linalg::matrix<double>(g4i.size(), g4j.size(), [&rng, &dist](size_t, size_t){return dist(rng);});
        }
          

        size_t nterms = 0;
        {
            START_TIMER;
            SOP<real_type> sop(Nmodes + 1);
            for(size_t i = 0; i < Nmodes; ++i)
            {
                sop += dist(rng)*sOP("H0", i+1);
                ++nterms;
            }
            sop += dist(rng)*sOP("sz", 0);

            for(size_t i = 0; i < g1.size(); ++i)
            {
                sop += ai[i]*sOP("|0><0|", 0)*sOP("Q", g1[i]+1);
                sop += bi[i]*sOP("|1><1|", 0)*sOP("Q", g1[i]+1);
                ++nterms;
            }

            for(size_t r = 0; r < g2.size(); ++r)
            {
                for(size_t i = 0; i < g2[r].size(); ++i)
                {
                    for(size_t j=i; j < g2[r].size(); ++j)
                    {
                        sop += aij[r](i, j)*sOP("|0><0|", 0)*sOP("Q", g2[r][i]+1)*sOP("Q", g2[r][j]+1);
                        sop += bij[r](i, j)*sOP("|1><1|", 0)*sOP("Q", g2[r][i]+1)*sOP("Q", g2[r][j]+1);
                        ++nterms;
                    }
                }
            }

            for(size_t i = 0; i < g3.size(); ++i)
            {
                sop += ci[i]*sOP("sx", 0)*sOP("Q", g3[i]+1);
                ++nterms;
            }

            for(size_t r = 0; r < g4i.size(); ++r)
            {
                for(size_t i = 0; i < g4i[r].size(); ++i)
                {
                    for(size_t j=i; j < g4j[r].size(); ++j)
                    {
                        sop += cij[r](i, j)*sOP("|0><0|", 0)*sOP("Q", g4i[r][i]+1)*sOP("Q", g4j[r][j]+1);
                        ++nterms;
                    }
                }
            }
            std::cout << nterms << std::endl;
            STOP_TIMER("SOP built");
              
            START_TIMER;
            STOP_TIMER("jordan_wigner");
            //std::cout << sop << std::endl;
            //START_TIMER;
            //compressedSOP<real_type> csop(sop);
            //STOP_TIMER("sop compression");
            //std::cout << csop << std::endl;

            //auto sopnew = csop.sop();
            //std::cout << sopnew << std::endl;
            //std::cout << sop << std::endl;

            std::vector<size_t> dims(Nmodes+1);  std::fill(dims.begin(), dims.end(), 2);
            ntree<size_t> topology;     topology.insert(1);
            //add electronic degrees of freedom
            topology().insert(1);   topology()[0].insert(1);
            topology().insert(1);

            size_t N1 = 1;  
            size_t N2 = 2;
            size_t N3 = 3;
            size_t N4 = 4;
            size_t N5 = 5;

            std::vector<size_t> m(24);
            for(size_t i=0; i < m.size(); ++i){m[i] = 1;}

            topology()[1].insert(N1);

            topology()[1][0].insert(N2);
            topology()[1][0][0].insert(m[0]);   topology()[1][0][0][0].insert(m[0]);
            topology()[1][0][0].insert(m[1]);   topology()[1][0][0][1].insert(m[1]);


            topology()[1][0].insert(N2);
            topology()[1][0][1].insert(m[2]);   topology()[1][0][1][0].insert(m[2]);
            topology()[1][0][1].insert(m[3]);   topology()[1][0][1][1].insert(m[3]);
            topology()[1][0][1].insert(m[4]);   topology()[1][0][1][2].insert(m[4]);


            topology()[1].insert(N1);
            topology()[1][1].insert(N3);
            topology()[1][1][0].insert(N4);
            topology()[1][1][0][0].insert(m[5]);    topology()[1][1][0][0][0].insert(m[5]);
            topology()[1][1][0][0].insert(m[6]);    topology()[1][1][0][0][1].insert(m[6]);
            topology()[1][1][0][0].insert(m[7]);    topology()[1][1][0][0][2].insert(m[7]);
            
            topology()[1][1][0].insert(N4);
            topology()[1][1][0][1].insert(m[8]);    topology()[1][1][0][1][0].insert(m[8]);
            topology()[1][1][0][1].insert(m[9]);    topology()[1][1][0][1][1].insert(m[9]);
            topology()[1][1][0][1].insert(m[10]);   topology()[1][1][0][1][2].insert(m[10]);

            topology()[1][1][0].insert(N4);
            topology()[1][1][0][2].insert(m[11]);   topology()[1][1][0][2][0].insert(m[11]);
            topology()[1][1][0][2].insert(m[12]);   topology()[1][1][0][2][1].insert(m[12]);
            topology()[1][1][0][2].insert(m[13]);   topology()[1][1][0][2][2].insert(m[13]);

            topology()[1][1].insert(N3);
            topology()[1][1][1].insert(N5);
            topology()[1][1][1][0].insert(m[14]);   topology()[1][1][1][0][0].insert(m[14]);
            topology()[1][1][1][0].insert(m[15]);   topology()[1][1][1][0][1].insert(m[15]);

            topology()[1][1][1].insert(N5);
            topology()[1][1][1][1].insert(m[16]);   topology()[1][1][1][1][0].insert(m[16]);
            topology()[1][1][1][1].insert(m[17]);   topology()[1][1][1][1][1].insert(m[17]);
            topology()[1][1][1][1].insert(m[18]);   topology()[1][1][1][1][2].insert(m[18]);
            topology()[1][1][1][1].insert(m[19]);   topology()[1][1][1][1][3].insert(m[19]);

            topology()[1][1][1].insert(N5);
            topology()[1][1][1][2].insert(m[20]);   topology()[1][1][1][2][0].insert(m[20]);
            topology()[1][1][1][2].insert(m[21]);   topology()[1][1][1][2][1].insert(m[21]);
            topology()[1][1][1][2].insert(m[22]);   topology()[1][1][1][2][2].insert(m[22]);
            topology()[1][1][1][2].insert(m[23]);   topology()[1][1][1][2][3].insert(m[23]);

            std::cout << topology << std::endl;
            ttn<complex_type, backend_type> A(topology);     

            return 0;
        }
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

}




