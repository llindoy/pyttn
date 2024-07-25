#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>
#include <complex>

#include <ttns_lib/ttn/tree/ntree_builder.hpp>
#include <ttns_lib/ttn/ttn.hpp>

ttns::ntree<size_t> generate_topology(size_t nsys, const std::vector<size_t>& nmodes, size_t nspf, size_t nspfl, size_t nbranch)
{
    //now we build the topology tree for 
    ttns::ntree<size_t> topology{};    topology.insert(1);
    topology().insert(nsys);        topology()[0].insert(nsys);
    topology().insert(nsys);
    
    size_t nlevels = static_cast<size_t>(std::log2(nmodes.size()));
    ttns::ntree_builder<size_t>::htucker_subtree(topology()[1], nmodes, nbranch, 
    [nspf, nspfl, nlevels](size_t l)
    {
        size_t ret = 0;
        if( l >= nlevels){ret = nspfl;}
        else if(l == 0){ret = nspf;}
        else
        {
            double rmax = std::log2(nspf);
            double rmin = std::log2(nspfl);
            ret = static_cast<size_t>(std::pow(2.0, ((nlevels-l)*static_cast<double>(rmax-rmin))/nlevels+rmin));
        }
        return ret;
    }
    );
    ttns::ntree_builder<size_t>::sanitise_tree(topology);
    return topology;
}

/* 
template <typename T> 
bool test_ttn_topology(const typename ttns::ttn<T>::node_type& A, const ttns::ntree_node<size_t>& node)
{
    if(A().hrank() != topology()){return false;}
    if(node.size() != A().size()){return false;}
    for(size_t i = 0; i < A().size(); ++i)
    {
        if(A.sha
        if(!A.is_leaf())
        {
            if(!test_ttn_topology(A[i], node[i])){return false;}
        }
    }
}


TEST_CASE("ttn", "[ttn]")
{
    using namespace ttns;
    using complex_type = std::complex<double>;

    SECTION("A TTN object can be constructed and sizes queried")
    {
        //build the topology object
        std::vector<size_t> modedims = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
        ntree<size_t> topology = generate_topology(2, modedims, 16, 8, 2);
        ttn<complex_type> A(topology);

        //now we iterate over the tree and ensure that the dimensions are all correct
        REQUIRE(!A.empty());
        REQUIRE(A.size() == topology.size() - (modedims.size()+1));
        REQUIRE(A.nleaves() == modedims.size() + 1);

        //test that the tensors of the network are the correct size
        

        SECTION("A ttn can be created then resized");
        {
            ttn<complex_type> B;

            REQUIRE(B.empty());
            B.resize(topology);
            REQUIRE(!B.empty());
            REQUIRE(B.size() == topology.size() - (modedims.size()+1));
            REQUIRE(B.nleaves() == modedims.size() + 1);

            std::vector<size_t> modedimsB = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
            ntree<size_t> topologyB = generate_topology(2, modedimsB, 16, 8, 3);
            B.resize(topologyB);
        }

        SECTION("A created ttn can be cleared")
        {
            A.clear();
            REQUIRE(A.empty());
            REQUIRE(A.size() == 0);
            REQUIRE(A.nleaves() == 0);
        }

        SECTION("A ttn can be copy constructed from another ttn.")
        {
            ttn<complex_type> C(A);

            //check that the result is not empty and has the correct size
            REQUIRE(!C.empty());
            REQUIRE(C.size() == A.size());
            REQUIRE(C.nleaves() == A.nleaves());

            //and ensure that each of its tensors is the correct size
            for(auto z : common::zip(A, C))
            {
                auto& a = std::get<0>(z);
                auto& c = std::get<1>(z);
                REQUIRE(a().shape() == c().shape());
            }
        }
    }

    SECTION("A TTN object can be initialised and represent quantum states.")
    {
    }
}*/


TEST_CASE("ttn", "[ttn]")
{

}
