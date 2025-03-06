#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>
#include <complex>

#include <utils/bipartite_graph.hpp>



/*  
class bipartite_graph
    static void generate_connected_subgraphs(const bipartite_graph<T, Val>& G, std::list<bipartite_graph<T, Val>>& SG)

    bipartite_graph() : _nedge(0) {}
    bipartite_graph(size_t n, size_t m) : _nedge(0)

    bipartite_graph(const bipartite_graph& o) = default;
    bipartite_graph(bipartite_graph&& o) = default;

    bipartite_graph& operator=(const bipartite_graph& o) = default;
    bipartite_graph& operator=(bipartite_graph&& o) = default;
    
    void clear()

    void resize(size_t n, size_t m)

    size_t add_U(const T& u)
    size_t add_V(const T& v)

    bool add_edge(size_t u, size_t v)
    bool add_edge(size_t u, size_t v, const Val& d)

    bool add_edge_v(const T& u, const T& v)
    bool add_edge_v(const T& u, const T& v, const Val& d)

    bool U_contains(const T& u) const
    bool V_contains(const T& v) const

    size_t U_ind(const T& u)
    size_t V_ind(const T& v)


    const T& U(size_t i) const
    T& U(size_t i)

    const T& V(size_t i) const
    T& V(size_t i)

    const std::list<size_t>& U_edges(size_t i) const
    const std::list<size_t>& V_edges(size_t i) const

    const std::list<Val>& U_edges_data(size_t i) const
    const std::list<Val>& V_edges_data(size_t i) const

    size_t N() const{return _U.size();}
    size_t M() const{return _V.size();}
    size_t nedge() const{return _nedge;}
*/


TEST_CASE("bipartite_graph", "[utils]")
{
    using namespace utils;
    //TODO: Write tests for the bipartite graph class
}
