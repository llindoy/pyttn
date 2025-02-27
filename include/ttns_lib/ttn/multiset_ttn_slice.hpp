#ifndef MULTISET_TTN_SLICE_HPP
#define MULTISET_TTN_SLICE_HPP

#include <common/tmp_funcs.hpp>
#include <common/exception_handling.hpp>

namespace ttns
{



//class for wrapping a single set index of a multiset_ttn_node object
template <typename T, typename backend, bool CONST>
struct multiset_slice_node_storage_type;

template <typename T, typename backend>
struct multiset_slice_node_storage_type<T, backend, true>
{
public:
    using size_type = typename backend::size_type;
    using node_type = typename ms_ttn<T, backend>::node_type;
    using type = const node_type&;

    multiset_slice_node_storage_type(type obj) : m_obj(obj){}

    type obj() {return m_obj;}
    type obj() const {return m_obj;}

protected:
    using node_data = const ttn_node_data<T, backend>&;
    using const_node_data = const ttn_node_data<T, backend>&;

    node_data get_node_data_slice(size_type i1) const {return m_obj(i1);}
    node_data get_node_data_slice(size_type i1) {return m_obj(i1);}
protected:
    type m_obj;
};

template <typename T, typename backend>
struct multiset_slice_node_storage_type<T, backend, false>
{
public:
    using size_type = typename backend::size_type;
    using node_type = typename ms_ttn<T, backend>::node_type;
    using type = node_type&;
    multiset_slice_node_storage_type(type obj) : m_obj(obj){}

    type obj(){return m_obj;}
    const node_type& obj() const {return m_obj;}

protected:
    using node_data = ttn_node_data<T, backend>&;
    using const_node_data = const ttn_node_data<T, backend>&;

    const_node_data get_node_data_slice(size_type i1) const {return m_obj(i1);}
    node_data get_node_data_slice(size_type i1) {return m_obj(i1);}
protected:
    type m_obj;
};


template <typename T, typename backend, bool CONST>
class multiset_ttn_node_slice : multiset_slice_node_storage_type<T, backend, CONST>
{
public:
    using size_type = typename backend::size_type;
    using base_type = multiset_slice_node_storage_type<T, backend, CONST>;
    using obj_type = typename base_type::type;

    using node_type = multiset_ttn_node_slice<T, backend, CONST>;
    using node_data = typename base_type::node_data;
    using const_node_data = typename base_type::const_node_data;

    using base_type::obj;

public:
    multiset_ttn_node_slice() = delete;
    multiset_ttn_node_slice(obj_type _obj, size_type slice_index) : base_type(_obj), m_slice_index(slice_index){}
    multiset_ttn_node_slice(const multiset_ttn_node_slice& o) = default;
    multiset_ttn_node_slice(multiset_ttn_node_slice&& o) = default;

    size_type nset() const{return 1;}

    size_type& slice_index(){return m_slice_index;}
    const size_type& slice_index() const{return m_slice_index;}
    size_type max_dim(size_type n) const{return base_type::m_obj.max_dim(n);}


    bool is_leaf() const
    {
        return base_type::m_obj.is_leaf();
    }

    size_type leaf_index() const
    {
        return base_type::m_obj.is_leaf();
    }

    size_type maxhrank(bool use_capacity = false) const{return base_type::m_obj.maxhrank(use_capacity);}
    size_type maxsize() const{return base_type::m_obj.maxsize();}
    size_type nmodes() const {return base_type::m_obj.nmodes();}

    const_node_data dataview(size_type ) const { return base_type::get_node_data_slice(m_slice_index);}
    node_data dataview(size_type ) {return base_type::get_node_data_slice(m_slice_index);}

    const_node_data operator()(size_type i1) const {ASSERT(i1 == 0, "Invalid set index"); return base_type::get_node_data_slice(m_slice_index);}
    node_data operator()(size_type i1) {ASSERT(i1 == 0, "Invalid set index"); return base_type::get_node_data_slice(m_slice_index);}

    const_node_data operator()() const { return base_type::get_node_data_slice(m_slice_index);}
    node_data operator()() {return base_type::get_node_data_slice(m_slice_index);}
protected:
    size_type m_slice_index;

};



//class for wrapping a single set index of a multiset_ttn object
template <typename T, typename backend, bool CONST>
struct multiset_slice_storage_type;

template <typename T, typename backend>
struct multiset_slice_storage_type<T, backend, true>
{
public:
    using size_type = typename backend::size_type;
    using type = const ms_ttn<T, backend>&;
    multiset_slice_storage_type(type obj) : m_obj(obj){}

    using slice_ref = multiset_ttn_node_slice<T, backend, true>;
    using const_slice_ref = multiset_ttn_node_slice<T, backend, true>;

    using slice_data = const ttn_node_data<T, backend>&;
    using const_slice_data = const ttn_node_data<T, backend>&;

    template <typename U, typename be>
    void set_slice(const ttn<U, be>& o, size_type sind)
    {
        RAISE_EXCEPTION("Cannot assign const multiset storage slice.");
    }

    template <typename U, typename be, bool C2>
    void set_slice(const multiset_ttn_slice<U, be, C2>& o, size_type sind, size_type sind2)
    {
        RAISE_EXCEPTION("Cannot assign const multiset storage slice.");
    }

    type obj() {return m_obj;}
    type obj() const {return m_obj;}

protected:
    type m_obj;
};

template <typename T, typename backend>
struct multiset_slice_storage_type<T, backend, false>
{
public:
    using size_type = typename backend::size_type;
    using type = ms_ttn<T, backend>&;

    using slice_ref = multiset_ttn_node_slice<T, backend, false>;
    using const_slice_ref = multiset_ttn_node_slice<T, backend, true>;

    multiset_slice_storage_type(type obj) : m_obj(obj){}

    using slice_data = ttn_node_data<T, backend>&;
    using const_slice_data = const ttn_node_data<T, backend>&;

    template <typename U, typename be>
    void set_slice(size_type sind, const ttn<U, be>& o)
    {
        CALL_AND_RETHROW(m_obj.set_slice(sind, o));
    }

    template <typename U, typename be, bool C2>
    void set_slice(size_type sind, const multiset_ttn_slice<U, be, C2>& o)
    {
        CALL_AND_RETHROW(m_obj.set_slice(sind, o));
    }

    type obj(){return m_obj;}
    const ms_ttn<T, backend>& obj() const {return m_obj;}
  

protected:
    type m_obj;
};


template <typename T, typename backend, bool CONST>
class multiset_ttn_slice : multiset_slice_storage_type<T, backend, CONST>
{
public:
    using size_type = typename backend::size_type;
    using base_type = multiset_slice_storage_type<T, backend, CONST>;
    using obj_type = typename base_type::type;

    using slice_ref = typename base_type::slice_ref;
    using const_slice_ref = typename base_type::const_slice_ref;

    using base_type::obj;

public:
    multiset_ttn_slice() = delete;
    multiset_ttn_slice(obj_type _obj, size_type slice_index) : base_type(_obj), m_slice_index(slice_index){}
    multiset_ttn_slice(const multiset_ttn_slice& o) = default;
    multiset_ttn_slice(multiset_ttn_slice&& o) = default;

    multiset_ttn_slice& operator=(const multiset_ttn_slice& o)
    {
        base_type::set_slice(m_slice_index, o);
        return *this;
    }

    template <typename U, typename be, bool enabled = CONST>
    multiset_ttn_slice& operator=(const ttn<U, be>& o)
    {
        base_type::set_slice(m_slice_index, o);
        return *this;
    }

    template <typename U, typename be, bool const2>
    typename std::enable_if<not std::is_same<be, backend>::value or not std::is_same<U, T>::value or const2 == CONST, multiset_ttn_slice&>::type operator=(const multiset_ttn_slice<U, be, const2>& o)
    {
        base_type::set_slice(m_slice_index, o);
        return *this;
    }

    size_type nset() const{return 1;}
    size_type size() const{return base_type::m_obj.size();}
    size_type nmodes() const {return base_type::m_obj.nmodes();}

    size_type& slice_index(){return m_slice_index;}
    const size_type& slice_index() const{return m_slice_index;}

    template <typename U>
    inline slice_ref operator[](U&& ind){return slice_ref(base_type::m_obj[ind], m_slice_index);}

    template <typename U>
    inline const_slice_ref operator[](U&& ind) const{return const_slice_ref(base_type::m_obj[ind], m_slice_index);}

    bool is_orthogonalised() const{return base_type::m_obj.is_orthogonalised();}
    bool has_orthogonality_centre() const{return base_type::m_obj.has_orthogonality_centre();}
    size_type orthogonality_centre() const{return base_type::m_obj.orthogonality_centre();}

    template <typename ancestor_index>
    void ancestor_indexing(const size_type& ind, ancestor_index& inds) const
    {
        CALL_AND_RETHROW(base_type::m_obj.ancestor_indexing(ind, inds));
    }

    template <typename mode_type, typename ancestor_index>
    void ancestor_indexing_leaf(const mode_type& li, ancestor_index& inds) const
    {
        CALL_AND_RETHROW(base_type::m_obj.ancestor_indexing_leaf(li, inds));
    }
protected:
    size_type m_slice_index;

};

template <typename U, typename T, typename backend, bool CONST, typename = typename std::enable_if<is_tree<T>::value, void>::type>
static inline bool 
has_same_structure(const T& t, const multiset_ttn_slice<U, backend, CONST>& u)
{
    return has_same_structure(t, u.obj());
}

template <typename U, typename T, typename backend, bool CONST, typename = typename std::enable_if<is_tree<T>::value, void>::type>
static inline bool 
has_same_structure(const multiset_ttn_slice<U, backend, CONST>& u, const T& t)
{
    return has_same_structure(t, u.obj());
}

template <typename U, typename T, typename backend, typename be, bool CONST, bool CONST2>
static inline bool 
has_same_structure(const multiset_ttn_slice<U, backend, CONST>& u, const multiset_ttn_slice<T, be, CONST2>& t)
{
    return has_same_structure(t.obj(), u.obj());
}
}   //namespace ttns

#endif


