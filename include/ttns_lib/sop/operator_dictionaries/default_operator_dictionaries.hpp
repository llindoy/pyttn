#ifndef TTNS_DEFAULT_OPERATOR_DICTIONARIES_HPP
#define TTNS_DEFAULT_OPERATOR_DICTIONARIES_HPP

#include "fermionic_operator.hpp"
#include "bosonic_operator.hpp"
#include "spin_operator.hpp"
#include "pauli_operator.hpp"
#include <unordered_map>

#include "../system_information.hpp"

namespace ttns
{
namespace fermion
{

template <typename T>
class default_fermion_operator_dictionary
{
public:
    using op_type = std::shared_ptr<single_site_operator<T>>;
    using dict_type = std::unordered_map<std::string, op_type>;

    static bool in_dict(const std::string& key)
    {
        return (s_dict.find(key) != s_dict.end());
    }

    static op_type query(const std::string& key)
    {
        auto it = s_dict.find(key);
        ASSERT(it != s_dict.end(), "Failed to query default fermion operator.  Operator not recognised.");
        return (*it).second;
    }

protected:
    static dict_type s_dict;
};

template <typename T> 
typename default_fermion_operator_dictionary<T>::dict_type default_fermion_operator_dictionary<T>::s_dict = 
{
    //bind the creation operators
    {std::string("cdag"), std::make_shared<creation<T>>()}, {std::string("cd"), std::make_shared<creation<T>>()},
    {std::string("adag"), std::make_shared<creation<T>>()}, {std::string("ad"), std::make_shared<creation<T>>()},
    {std::string("fdag"), std::make_shared<creation<T>>()}, {std::string("fd"), std::make_shared<creation<T>>()},
    //bind the annihilation<T> operators
    {std::string("c"), std::make_shared<annihilation<T>>()},
    {std::string("a"), std::make_shared<annihilation<T>>()},
    {std::string("f"), std::make_shared<annihilation<T>>()},
    //bind the number operators
    {std::string("n"), std::make_shared<number<T>>()},
    {std::string("cdagc"), std::make_shared<number<T>>()}, {std::string("cdc"), std::make_shared<number<T>>()},
    {std::string("adaga"), std::make_shared<number<T>>()}, {std::string("ada"), std::make_shared<number<T>>()},
    {std::string("fdagf"), std::make_shared<number<T>>()}, {std::string("fdf"), std::make_shared<number<T>>()},
    //bind the vacancy operators
    {std::string("v"), std::make_shared<vacancy<T>>()},
    {std::string("jw"), std::make_shared<jordan_wigner<T>>()}
};

}   //namespace fermion


namespace boson
{

template <typename T>
class default_boson_operator_dictionary
{
public:
    using op_type = std::shared_ptr<single_site_operator<T>>;
    using dict_type = std::unordered_map<std::string, op_type>;

    static bool in_dict(const std::string& key)
    {
        return (s_dict.find(key) != s_dict.end());
    }

    static op_type query(const std::string& key)
    {
        auto it = s_dict.find(key);
        ASSERT(it != s_dict.end(), "Failed to query default bosonic operator.  Operator not recognised.");
        return (*it).second;
    }

protected:
    static dict_type s_dict;
};

template <typename T> 
typename default_boson_operator_dictionary<T>::dict_type default_boson_operator_dictionary<T>::s_dict = 
{
    //bind the creation operators
    {std::string("cdag"), std::make_shared<creation<T>>()}, {std::string("cd"), std::make_shared<creation<T>>()},
    {std::string("adag"), std::make_shared<creation<T>>()}, {std::string("ad"), std::make_shared<creation<T>>()},
    {std::string("bdag"), std::make_shared<creation<T>>()}, {std::string("bd"), std::make_shared<creation<T>>()},
    //bind the annihilation<T> operators
    {std::string("c"), std::make_shared<annihilation<T>>()},
    {std::string("a"), std::make_shared<annihilation<T>>()},
    {std::string("b"), std::make_shared<annihilation<T>>()},
    //bind the number operators
    {std::string("n"), std::make_shared<number<T>>()},
    {std::string("cdagc"), std::make_shared<number<T>>()}, {std::string("cdc"), std::make_shared<number<T>>()},
    {std::string("adaga"), std::make_shared<number<T>>()}, {std::string("ada"), std::make_shared<number<T>>()},
    {std::string("bdagb"), std::make_shared<number<T>>()}, {std::string("bdb"), std::make_shared<number<T>>()},
    //bind the position operators
    {std::string("x"), std::make_shared<position<T>>()},
    {std::string("q"), std::make_shared<position<T>>()},
    //bind the moment operators
    {std::string("p"), std::make_shared<momentum<T>>()},
};

}   //namespace boson


namespace spin
{

template <typename T>
class default_spin_operator_dictionary
{
public:
    using op_type = std::shared_ptr<single_site_operator<T>>;
    using dict_type = std::unordered_map<std::string, op_type>;

    static bool in_dict(const std::string& key)
    {
        return (s_dict.find(key) != s_dict.end());
    }

    static op_type query(const std::string& key)
    {
        auto it = s_dict.find(key);
        ASSERT(it != s_dict.end(), "Failed to query default spin operator.  Operator not recognised.");
        return (*it).second;
    }

protected:
    static dict_type s_dict;
};

template <typename T> 
typename default_spin_operator_dictionary<T>::dict_type default_spin_operator_dictionary<T>::s_dict = 
{
    //bind the spin raising operators
    {std::string("s+"), std::make_shared<S_p<T>>()}, {std::string("sp"), std::make_shared<S_p<T>>()},
    //bind the spin lower operators
    {std::string("s-"), std::make_shared<S_m<T>>()}, {std::string("sm"), std::make_shared<S_m<T>>()},
    //bind the spin x operators
    {std::string("sx"), std::make_shared<S_x<T>>()},
    {std::string("x"), std::make_shared<S_x<T>>()},
    //bind the spin y operators
    {std::string("sy"), std::make_shared<S_y<T>>()},
    {std::string("y"), std::make_shared<S_y<T>>()},
    //bind the spin z operators
    {std::string("sz"), std::make_shared<S_z<T>>()},
    {std::string("z"), std::make_shared<S_z<T>>()}
};

}   //namespace spin

namespace pauli
{

template <typename T>
class default_pauli_operator_dictionary
{
public:
    using op_type = std::shared_ptr<single_site_operator<T>>;
    using dict_type = std::unordered_map<std::string, op_type>;

    static bool in_dict(const std::string& key)
    {
        return (s_dict.find(key) != s_dict.end());
    }

    static op_type query(const std::string& key)
    {
        auto it = s_dict.find(key);
        ASSERT(it != s_dict.end(), "Failed to query default pauli operator.  Operator not recognised.");
        return (*it).second;
    }

protected:
    static dict_type s_dict;
};


template <typename T> 
typename default_pauli_operator_dictionary<T>::dict_type default_pauli_operator_dictionary<T>::s_dict = 
{
    //bind the pauli raising operators
    {std::string("s+"), std::make_shared<sigma_p<T>>()}, {std::string("sp"), std::make_shared<sigma_p<T>>()},
    {std::string("sigma+"), std::make_shared<sigma_p<T>>()}, {std::string("sigmap"), std::make_shared<sigma_p<T>>()},
    //bind the pauli lower operators
    {std::string("sigma-"), std::make_shared<sigma_m<T>>()}, {std::string("sigmam"), std::make_shared<sigma_m<T>>()},
    //bind the pauli x operators
    {std::string("sigmax"), std::make_shared<sigma_x<T>>()},
    {std::string("x"), std::make_shared<sigma_x<T>>()},
    {std::string("sx"), std::make_shared<sigma_x<T>>()},
    //bind the pauli y operators
    {std::string("sigmay"), std::make_shared<sigma_y<T>>()},
    {std::string("y"), std::make_shared<sigma_y<T>>()},
    {std::string("sy"), std::make_shared<sigma_y<T>>()},
    //bind the pauli z operators
    {std::string("sigmaz"), std::make_shared<sigma_z<T>>()},
    {std::string("z"), std::make_shared<sigma_z<T>>()},
    {std::string("sz"), std::make_shared<sigma_z<T>>()}
};

}   //namespace pauli


template <typename T>
std::shared_ptr<single_site_operator<T>> query_default_operator_dictionary(const mode_type& type, const std::string& label)
{
    using fermi_dict = fermion::default_fermion_operator_dictionary<T>;
    using bose_dict = boson::default_boson_operator_dictionary<T>;
    using pauli_dict = pauli::default_pauli_operator_dictionary<T>;
    using spin_dict = spin::default_spin_operator_dictionary<T>;
    switch(type)
    {   
        case mode_type::FERMION_MODE:
            CALL_AND_RETHROW(return fermi_dict::query(label));

        case mode_type::BOSON_MODE:
            CALL_AND_RETHROW(return bose_dict::query(label));

        case mode_type::QUBIT_MODE:
            CALL_AND_RETHROW(return pauli_dict::query(label));
            break;

        case mode_type::SPIN_MODE:
            CALL_AND_RETHROW(return spin_dict::query(label));

        case mode_type::GENERIC_MODE:
            RAISE_EXCEPTION("No default mode dictionary for generic mode types.");
    };
    return std::shared_ptr<single_site_operator<T>>(nullptr);
}

}   //namespace ttns

#endif
