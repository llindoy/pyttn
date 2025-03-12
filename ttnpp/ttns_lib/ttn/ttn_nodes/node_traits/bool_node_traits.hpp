#ifndef TTNS_BOOL_NODE_TRAITS_HPP
#define TTNS_BOOL_NODE_TRAITS_HPP

namespace ttns
{

namespace node_data_traits
{
    //bool nodes
    template <>
    struct assignment_traits<bool, bool>
    {
        using is_applicable = std::true_type;

        inline void operator()(bool& o,  const bool& i){o=i;}
    };

    //resize traits for tensor and matrix objects
    template <typename ... Args>
    struct resize_traits<bool, Args...>
    {
        using is_applicable = std::true_type;
        inline void operator()(bool& /* o */, const Args&... /* args */){}
    };

    template <typename ... Args>
    struct reallocate_traits<bool, Args...>
    {
        using is_applicable = std::true_type;
        inline void operator()(bool& /* o */, const Args&... /* args */){}
    };

    //size comparison traits for the tensor and matrix objects
    template <typename ... Args>
    struct size_comparison_traits<bool, Args...>
    {
        using is_applicable = std::true_type;

        inline bool operator()(const bool& /* o */, const Args&... /* i */){return true;}
    };


    /*
     * size_t nodes
     */
    template <>
    struct default_initialisation_traits<size_t>
    {
        using is_applicable = std::true_type;
        template <typename ... Args>
        void operator()(size_t& n , Args&& ... /* args */){ n = 0;}
    };

    template <>
    struct assignment_traits<size_t, size_t>
    {
        using is_applicable = std::true_type;

        inline void operator()(size_t& o,  const size_t& i){o=i;}
    };

    template <typename ... Args>
    struct resize_traits<size_t, Args...>
    {
        using is_applicable = std::true_type;
        inline void operator()(size_t& o, const Args&... /* args */){o = 0;}
    };

    template <typename ... Args>
    struct reallocate_traits<size_t, Args...>
    {
        using is_applicable = std::true_type;
        inline void operator()(size_t& o, const Args&... /* args */){o = 0;}
    };

    template <typename ... Args>
    struct size_comparison_traits<size_t, Args...>
    {
        using is_applicable = std::true_type;

        inline size_t operator()(const size_t& /* o */, const Args&... /* i */){return true;}
    };
}



}

#endif  //TTNS_BOOL_NODE_TRAITS_HPP//

