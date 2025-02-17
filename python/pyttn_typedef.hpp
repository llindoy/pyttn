#ifndef PYTTN_TYPE_DEFINITIONS_HPP
#define PYTTN_TYPE_DEFINITIONS_HPP

#include <string>
#include "linalg/linalg.hpp"

using pyttn_real_type = double;

template <typename T>
class pyttn_type_label;

template <>
class pyttn_type_label<pyttn_real_type>
{
public:
    static std::string label(){return std::string("R");}
};

template <>
class pyttn_type_label<linalg::complex<pyttn_real_type>>
{
public:
    static std::string label(){return std::string("C");}
};



#endif  //PYTTN_TYPE_DEFINITIONS_HPP