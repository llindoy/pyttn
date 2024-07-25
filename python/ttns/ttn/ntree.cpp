#include "ntree.hpp"

void initialise_ntree(py::module& m)
{
    init_ntree_node<size_t>(m);
    init_ntree<size_t>(m);
    init_ntree_builder<size_t>(m);
}
