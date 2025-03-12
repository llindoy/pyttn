#ifndef TTNS_DATASTRUCTURES_NODE_DATA_TRAITS_HPP
#define TTNS_DATASTRUCTURES_NODE_DATA_TRAITS_HPP

#include <type_traits>
#include "ntree.hpp"
#include <common/exception_handling.hpp>

namespace ttns
{

namespace node_data_traits
{
    template <typename T>
    struct default_initialisation_traits 
    {
        using is_applicable = std::true_type;
        template <typename ... Args>
        void operator()(T& /* n */, Args&& ... /* args */){}
    };

    template <typename T, typename ... Args>
    struct assignment_traits
    {
        using is_applicable = std::false_type;
    };

    template <typename T, typename ... Args>
    struct resize_traits
    {
        using is_applicable = std::false_type;
    };


    template <typename T, typename ... Args>
    struct reallocate_traits
    {
        using is_applicable = std::false_type;
    };

    template <typename T, typename ... Args>
    struct size_comparison_traits
    {
        using is_applicable = std::false_type;
    };

    template <typename T>
    struct clear_traits 
    {
        using is_applicable = std::true_type;
        void operator()(T& /* o */){}
    };
}   //namespace  node_data_traits

}   //namespace ttns

#endif  //TTNS_DATASTRUCTURES_NODE_DATA_TRAITS_HPP//


