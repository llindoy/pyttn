#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>
#include <complex>

#include <utils/bipartite_graph.hpp>



//a structure for storing a bipartite graph that allows for information to be stored at each edge
/*
class bipartite_matching
    bipartite_matching() : _nU(0), _nV(0) {}
    template <typename T, typename V>
    bipartite_matching(const bipartite_graph<T, V>& o)

    template <typename T, typename V>
    size_t compute_maximal_matching(const bipartite_graph<T, V>& o)

    std::vector<std::pair<size_t, size_t>> edges() const

    template <typename T, typename Val>
    std::pair<std::vector<size_t>, std::vector<size_t>> minimum_vertex_cover(const bipartite_graph<T, Val>& o) const
*/


TEST_CASE("bipartite_matching", "[utils]")
{
    using namespace utils;

    /*  Test the bipartite matching and minimum vertex cover of the graph
     *  0  
     *  | \
     *  0  1
     *  which should give
     *  bpm     minimum vertex covert
     *  0       0 <-
     *  | 
     *  0   1   0  1
     */
    SECTION("Test Graph 1")
    {
        bipartite_graph<size_t, size_t> bpg;
        bpg.add_U(0);
        bpg.add_V(0);
        bpg.add_V(1);

        bpg.add_edge(0, 0);
        bpg.add_edge(0, 1);

        utils::bipartite_matching bpm(bpg);
        auto m = bpm.edges();

        //check that the 
        REQUIRE(m.size() == 1);
        REQUIRE(std::get<0>(m[0]) == 0);
        REQUIRE(std::get<1>(m[0]) == 0);

        //compute the minimum vertex cover of the edges - this allows us to construct the optimised spf operators.  
        //Namely for any node in the minimum vertex cover of the mean-field operators.  We can construct a optimised
        //SPF operator by combining all of the nodes present in the corresponding 
        auto vc = bpm.minimum_vertex_cover(bpg);
        
        auto& _U = std::get<0>(vc); auto& _V = std::get<1>(vc);

        //there is a single U mode in the minimum vertex cover and it is node 0
        REQUIRE(_U.size() == 1);
        REQUIRE(_U[0] == 0);

        //there are no V nodes in the minimum vertex cover
        REQUIRE(_V.size() == 0);
    }


    /*  Test the bipartite matching and minimum vertex cover of the graph
     *  0  1
     *  | /
     *  0
     *  which should give
     *  bpm     minimum vertex covert
     *  0  1    0  1
     *  |       
     *  0       0 <-
     */
    SECTION("Test Graph 2")
    {
        bipartite_graph<size_t, size_t> bpg;
        bpg.add_U(0);
        bpg.add_U(1);
        bpg.add_V(0);

        bpg.add_edge(0, 0);
        bpg.add_edge(1, 0);

        utils::bipartite_matching bpm(bpg);
        auto m = bpm.edges();

        //check that the 
        REQUIRE(m.size() == 1);
        REQUIRE(std::get<0>(m[0]) == 0);
        REQUIRE(std::get<1>(m[0]) == 0);

        //compute the minimum vertex cover of the edges - this allows us to construct the optimised spf operators.  
        //Namely for any node in the minimum vertex cover of the mean-field operators.  We can construct a optimised
        //SPF operator by combining all of the nodes present in the corresponding 
        auto vc = bpm.minimum_vertex_cover(bpg);
        
        auto& _U = std::get<0>(vc); auto& _V = std::get<1>(vc);

        //there are no U nodes in the minimum vertex cover
        REQUIRE(_U.size() == 0);

        //there is a single V mode in the minimum vertex cover and it is node 0
        REQUIRE(_V.size() == 1);
        REQUIRE(_V[0] == 0);
    }

    /*  Test the bipartite matching and minimum vertex cover of the graph
     */
    SECTION("Test Graph 1")
    {
        bipartite_graph<size_t, size_t> bpg;
        bpg.add_U(0);
        bpg.add_U(1);
        bpg.add_U(2);
        bpg.add_U(3);
        bpg.add_U(4);
        bpg.add_U(5);
        bpg.add_U(6);
        bpg.add_U(7);
        bpg.add_U(8);
        bpg.add_U(9);
        bpg.add_U(10);
        bpg.add_U(11);
        bpg.add_V(0);
        bpg.add_V(1);
        bpg.add_V(2);
        bpg.add_V(3);

        bpg.add_edge(0, 0);
        bpg.add_edge(0, 1);
        bpg.add_edge(0, 2);
        bpg.add_edge(0, 3);

        bpg.add_edge(1, 0);
        bpg.add_edge(1, 1);
        bpg.add_edge(1, 2);
        bpg.add_edge(1, 3);

        bpg.add_edge(2, 0);
        bpg.add_edge(2, 1);
        bpg.add_edge(2, 2);
        bpg.add_edge(2, 3);

        for(size_t i = 3; i< 11; ++i)
        {
            bpg.add_edge(i, 2);
            bpg.add_edge(i, 3);
        }

        utils::bipartite_matching bpm(bpg);
        auto m = bpm.edges();

        for(auto _e : m)
        {
            std::cout << std::get<0>(_e) << "--" << std::get<1>(_e) << std::endl;
        }

        //check that the 
        REQUIRE(m.size() == 4);
        REQUIRE(std::get<0>(m[0]) == 0);
        REQUIRE(std::get<1>(m[0]) == 0);
        REQUIRE(std::get<0>(m[1]) == 1);
        REQUIRE(std::get<1>(m[1]) == 1);
        REQUIRE(std::get<0>(m[2]) == 2);
        REQUIRE(std::get<1>(m[2]) == 2);
        REQUIRE(std::get<0>(m[3]) == 3);
        REQUIRE(std::get<1>(m[3]) == 3);

        //compute the minimum vertex cover of the edges - this allows us to construct the optimised spf operators.  
        //Namely for any node in the minimum vertex cover of the mean-field operators.  We can construct a optimised
        //SPF operator by combining all of the nodes present in the corresponding 
        auto vc = bpm.minimum_vertex_cover(bpg);
        
        auto& _U = std::get<0>(vc); auto& _V = std::get<1>(vc);
        for(auto u : _U)
        {
            std::cout << u << " ";
        }std::cout << std::endl;

        for(auto v : _V)
        {
            std::cout << v << " ";
        }std::cout << std::endl;

        //there is a single U mode in the minimum vertex cover and it is node 0
        REQUIRE(_U.size() == 0);

        //there are no V nodes in the minimum vertex cover
        REQUIRE(_V.size() == 4);
        REQUIRE(_V[0] == 0);
        REQUIRE(_V[1] == 1);
        REQUIRE(_V[2] == 2);
        REQUIRE(_V[3] == 3);
    }
}
