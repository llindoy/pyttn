#ifndef TTNS_MULTI_SET_TTN_NODE_TRAITS_HPP
#define TTNS_MULTI_SET_TTN_NODE_TRAITS_HPP

#include <linalg/linalg.hpp>

#include <type_traits>

namespace ttns
{

namespace node_data_traits
{
    //assignment traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct assignment_traits<multiset_node_data<T, backend1>, multiset_node_data<U, backend2> >
    {
        using hdata = multiset_node_data<T, backend1>;
        using hdata2 = multiset_node_data<U, backend2>;
        using is_applicable = std::is_convertible<U, T>;

        inline typename std::enable_if<std::is_convertible<U, T>::value, void>::type operator()(hdata& o,  const hdata2& i)
        {
            CALL_AND_RETHROW(o.resize(i.size()));
            for(size_t ind = 0; ind <i.size(); ++ind)
            {
                CALL_AND_RETHROW(o[ind] = i[ind]);
            }
        }
    };

    //assignment traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct assignment_traits<multiset_node_data<T, backend1>, ttn_node_data<U, backend2> >
    {
        using hdata = multiset_node_data<T, backend1>;
        using hdata2 = ttn_node_data<U, backend2>;
        using is_applicable = std::is_convertible<U, T>;

        inline typename std::enable_if<std::is_convertible<U, T>::value, void>::type operator()(hdata& o,  const hdata2& i)
        {
            CALL_AND_RETHROW(o.resize(1));
            CALL_AND_RETHROW(o[0] = i);
        }
    };


    //assignment traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct assignment_traits<multiset_node_data<T, backend1>, linalg::matrix<U, backend2>>
    {
        using hdata = multiset_node_data<T, backend1>;
        using mat = linalg::matrix<U, backend2>;

        using is_applicable = std::is_convertible<U, T>;

        inline typename std::enable_if<std::is_convertible<U, T>::value, void>::type operator()(hdata& o,  const mat& i)
        {
            CALL_AND_RETHROW(o.resize(1));
            CALL_AND_RETHROW(o[0] = i);
        }
    };


    //resize traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct resize_traits<multiset_node_data<T, backend1>, multiset_node_data<U, backend2> >
    {
        template <typename V, typename backend> using hdata = multiset_node_data<V, backend>;

        using is_applicable = std::true_type;

        inline void operator()(hdata<T, backend1>& o,  const hdata<U, backend2>& i)
        {
            CALL_AND_RETHROW(o.resize(i.size()));
            for(size_t ind = 0; ind < i.size(); ++ind)
            {
                CALL_AND_RETHROW(o[ind].resize(i[ind].hrank(), i[ind].dims()));
            }
        }
    };


    //resize traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct resize_traits<multiset_node_data<T, backend1>, ttn_node_data<U, backend2> >
    {
        template <typename V, typename backend> using hdata = multiset_node_data<V, backend>;
        template <typename V, typename backend> using hdata2 = ttn_node_data<V, backend>;

        using is_applicable = std::true_type;

        inline void operator()(hdata<T, backend1>& o,  const hdata2<U, backend2>& i)
        {
            CALL_AND_RETHROW(o.resize(1));
            CALL_AND_RETHROW(o[0].resize(i.hrank(), i.dims()));
        }
    };

    //resize traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct reallocate_traits<multiset_node_data<T, backend1>, multiset_node_data<U, backend2> >
    {
        template <typename V, typename backend> using hdata = multiset_node_data<V, backend>;

        using is_applicable = std::true_type;

        inline void operator()(hdata<T, backend1>& o,  const hdata<U, backend2>& i)
        {
            CALL_AND_RETHROW(o.reallocate(i.size()));
            for(size_t ind = 0; ind < i.size(); ++ind)
            {
                CALL_AND_RETHROW(o[ind].reallocate(i[ind].hrank(), i[ind].dims()));
            }
        }
    };


    //reallocate traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct reallocate_traits<multiset_node_data<T, backend1>, ttn_node_data<U, backend2> >
    {
        template <typename V, typename backend> using hdata = multiset_node_data<V, backend>;
        template <typename V, typename backend> using hdata2 = ttn_node_data<V, backend>;

        using is_applicable = std::true_type;

        inline void operator()(hdata<T, backend1>& o,  const hdata2<U, backend2>& i)
        {
            CALL_AND_RETHROW(o.reallocate(1));
            CALL_AND_RETHROW(o[0].reallocate(i.hrank(), i.dims()));
        }
    };

    //size comparison traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct size_comparison_traits<multiset_node_data<T, backend1>, multiset_node_data<U, backend2> >
    {
        template <typename V, typename backend> using hdata = multiset_node_data<V, backend>;

        using is_applicable = std::true_type;

        inline bool operator()(hdata<T, backend1>& o,  const hdata<U, backend2>& i)
        {
            if(o.size() == i.size())
            {
                for(size_t ind = 0; ind < o.size(); ++ind)
                {
                    if(o[ind].nmodes() != i[ind].nmodes()){return false;}
                    else if(! ((o[ind].shape() == i[ind].shape()) && (o[ind].dims() == i[ind].dims()))){return false;}
                }
                return true;
            }
            else
            {
                return false;
            }
        }
    };


    //size_comparison traits for the httensor node data object
    template <typename T, typename U, typename backend1, typename backend2>
    struct size_comparison_traits<multiset_node_data<T, backend1>, ttn_node_data<U, backend2> >
    {
        template <typename V, typename backend> using hdata = multiset_node_data<V, backend>;
        template <typename V, typename backend> using hdata2 = ttn_node_data<V, backend>;

        using is_applicable = std::true_type;

        inline bool operator()(hdata<T, backend1>& o,  const hdata2<U, backend2>& i)
        {
            if(o.size() == 1)
            {
                if(o[0].nmodes() != i.nmodes()){return false;}
                else
                {
                    return ((o[0].shape() == i.shape()) && (o[0].dims() == i.dims()));
                }
            }
            else
            {
                return false;
            }
        }
    };

    template <typename T, typename U, typename backend1, typename backend2>
    struct size_comparison_traits<multiset_node_data<T, backend1>, linalg::matrix<U, backend2> >
    {
        using hdata = multiset_node_data<T, backend1>;
        using mat = linalg::matrix<U, backend2>;

        using is_applicable = std::true_type;

        inline bool operator()(const hdata& o,  const mat& i)
        {
            if(o.size() == 1)
            {
                return (o[0].shape() == i.shape());
            }
        }
    };

    //clear traits for the httensor node data object
    template <typename T, typename backend>
    struct clear_traits<multiset_node_data<T, backend> > 
    {
        void operator()(multiset_node_data<T, backend>& t){CALL_AND_RETHROW(t.clear());}
    };
}

}

#endif  //TTNS_MULTI_SET_TENSOR_NODE_TRAITS_HPP//

