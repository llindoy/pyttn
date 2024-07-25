#ifndef TTNS_TENSOR_NODE_TRAITS_HPP
#define TTNS_TENSOR_NODE_TRAITS_HPP

namespace ttns
{

namespace node_data_traits
{
    //assignment traits for the tensor and matrix objects
    template <typename T, typename U, size_t D, typename backend1, typename backend2>
    struct assignment_traits<linalg::tensor<T, D, backend1>, linalg::tensor<U, D, backend2> >
    {
        template <typename V, typename bck> using tens = linalg::tensor<V, D, bck>;

        using is_applicable = std::is_convertible<U, T>;

        inline void operator()(tens<T, backend1>& o,  const tens<U, backend2>& i){CALL_AND_RETHROW(o = i);}
    };

    template <typename T, typename U, typename backend1, typename backend2>
    struct assignment_traits<linalg::matrix<T, backend1>, linalg::matrix<U, backend2> >
    {
        template <typename V, typename bck> using mat = linalg::matrix<V, bck>;

        using is_applicable = std::is_convertible<U, T>;

        inline void operator()(mat<T, backend1>& o,  const mat<U, backend2>& i){CALL_AND_RETHROW(o = i);}
    };

    template <typename T, typename U, typename backend1, typename backend2>
    struct assignment_traits<linalg::matrix<T, backend1>, ttn_node_data<U, backend2> >
    {
        using mat = linalg::matrix<T, backend1>;
        using hdata = ttn_node_data<U, backend2>;

        using is_applicable = std::is_convertible<U, T>;

        inline void operator()(mat& o,  const hdata& i){CALL_AND_RETHROW(o = i.as_matrix());}
    };


    //resize traits for tensor and matrix objects
    template <typename T, typename U, size_t D, typename backend1, typename backend2>
    struct resize_traits<linalg::tensor<T, D, backend1>, linalg::tensor<U, D, backend2>>
    {
        template <typename V, typename backend> using tens = linalg::tensor<V, D, backend>;

        using is_applicable = std::true_type;
        inline void operator()(tens<T, backend1>& o,  const tens<U, backend2>& i){CALL_AND_RETHROW(o.resize(i.shape()));}
    };

    template <typename T, typename U, typename backend1, typename backend2>
    struct resize_traits<linalg::matrix<T, backend1>, linalg::matrix<U, backend2> >
    {
        template <typename V, typename backend> using mat = linalg::matrix<V, backend>;

        using is_applicable = std::true_type;
        inline void operator()(mat<T, backend1>& o,  const mat<U, backend2>& i){CALL_AND_RETHROW(o.resize(i.shape()));}
    };

    template <typename T, typename U, typename backend1, typename backend2>
    struct resize_traits<linalg::matrix<T, backend1>, ttn_node_data<U, backend2> >
    {
        using mat = linalg::matrix<T, backend1>;
        using hdata = ttn_node_data<U, backend2>;

        using is_applicable = std::true_type;
        inline void operator()(mat& o,  const hdata& i){CALL_AND_RETHROW(o.resize(i.shape()));}
    };

    //reallocate traits for tensor and matrix objects
    template <typename T, typename U, size_t D, typename backend1, typename backend2>
    struct reallocate_traits<linalg::tensor<T, D, backend1>, linalg::tensor<U, D, backend2>>
    {
        template <typename V, typename backend> using tens = linalg::tensor<V, D, backend>;

        using is_applicable = std::true_type;
        inline void operator()(tens<T, backend1>& o,  const tens<U, backend2>& i){CALL_AND_RETHROW(o.reallocate(i.capacity()));}
    };

    template <typename T, typename U, typename backend1, typename backend2>
    struct reallocate_traits<linalg::matrix<T, backend1>, linalg::matrix<U, backend2> >
    {
        template <typename V, typename backend> using mat = linalg::matrix<V, backend>;

        using is_applicable = std::true_type;
        inline void operator()(mat<T, backend1>& o,  const mat<U, backend2>& i){CALL_AND_RETHROW(o.reallocate(i.capacity()));}
    };

    template <typename T, typename U, typename backend1, typename backend2>
    struct reallocate_traits<linalg::matrix<T, backend1>, ttn_node_data<U, backend2> >
    {
        using mat = linalg::matrix<T, backend1>;
        using hdata = ttn_node_data<U, backend2>;

        using is_applicable = std::true_type;
        inline void operator()(mat& o,  const hdata& i){CALL_AND_RETHROW(o.reallocate(i.capacity()));}
    };

    //size comparison traits for the tensor and matrix objects
    template <typename T, typename U, size_t D, typename backend1, typename backend2>
    struct size_comparison_traits<linalg::tensor<T, D, backend1>, linalg::tensor<U, D, backend2> >
    {
        template <typename V, typename backend> using tens = linalg::tensor<V, D, backend>;

        using is_applicable = std::true_type;

        inline bool operator()(const tens<T, backend1>& o, const tens<U, backend2>& i){return o.shape() == i.shape();}
    };
    
    template <typename T, typename U, typename backend1, typename backend2>
    struct size_comparison_traits<linalg::matrix<T, backend1>, linalg::matrix<U, backend2> >
    {
        template <typename V, typename backend> using mat = linalg::matrix<V, backend>;

        using is_applicable = std::true_type;

        inline bool operator()(const mat<T, backend1>& o,  const mat<U, backend2>& i)
        {
            return (o.shape() == i.shape());
        }
    };

    template <typename T, typename U, typename backend1, typename backend2>
    struct size_comparison_traits<linalg::matrix<T, backend1>, ttn_node_data<U, backend2> >
    {
        using mat = linalg::matrix<T, backend1>;
        using hdata = ttn_node_data<U, backend2>;

        using is_applicable = std::true_type;

        inline bool operator()(const mat& o,  const hdata& i)
        {
            return (o.shape() == i.shape());
        }
    };


    //clear traits for the ttn node data object
    template <typename T, size_t D, typename backend>
    struct clear_traits<tensor<T, D, backend> > 
    {
        void operator()(tensor<T, D, backend>& t){CALL_AND_RETHROW(t.clear());}
    };
}

}

#endif  //TTNS_TENSOR_NODE_TRAITS_HPP//

