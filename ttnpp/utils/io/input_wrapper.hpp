#ifndef UTILS_IO_INPUT_WRAPPER_HPP_
#define UTILS_IO_INPUT_WRAPPER_HPP_

#include <iostream>
#include <sstream>
#include <regex>
#include <string>
#include <algorithm>

#include <common/exception_handling.hpp>
#include <linalg/linalg.hpp>

namespace utils
{
namespace io
{



static std::string& remove_whitespace_and_to_lower(std::string& str)
{
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){return std::tolower(c);});
    str.erase(std::remove_if(str.begin(), str.end(), [](unsigned char c){return (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f');}), str.end());
    return str;
}
}
}



#ifdef USE_RAPIDJSON

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>

namespace utils
{
namespace io
{


template <typename INTERFACE_TYPE>
class factory;

template <typename Impl>
class input_wrapper;


struct rapidjson_tag{};

namespace rapidjson_loader
{
template <typename type, typename enabled = void>
class loader;

template <typename T> 
class complex_from_string
{
public:
    using complex_type = linalg::complex<T>;
    static bool is_valid(const std::string& str)
    {
        std::regex x_regex("^(-)?([0-9]+([.][0-9]*)?|[.][0-9]+)$");   //for matching a purely real number
        std::regex iy_regex("^(-)?([0-9]+([.][0-9]*)?|[.][0-9]+)[ij]$");  //for matching a purely imaginary number of the form yi
        std::regex x_yi_regex("^((-)?([0-9]+([.][0-9]*)?|[.][0-9]+))(([-+])?([0-9]+([.][0-9]*)?|[.][0-9]+)[ij])$");   //for 
        std::regex yi_x_regex("^((-)?([0-9]+([.][0-9]*)?|[.][0-9]+)[ij])(([-+])?([0-9]+([.][0-9]*)?|[.][0-9]+))$");   //for 
        std::smatch match;
        return  (std::regex_match(str, match, x_regex) || std::regex_match(str, match, iy_regex) || std::regex_match(str, match, x_yi_regex) || std::regex_match(str, match, yi_x_regex));
    }
    static complex_type get(const std::string& str)
    {
        T x = 0, y = 0;
        std::regex x_regex("^(-)?([0-9]+([.][0-9]*)?|[.][0-9]+)$");   //for matching a purely real number
        std::regex iy_regex("^(-)?([0-9]+([.][0-9]*)?|[.][0-9]+)[ij]$");  //for matching a purely imaginary number of the form yi
        std::regex x_yi_regex("^((-)?([0-9]+([.][0-9]*)?|[.][0-9]+))(([-+])?([0-9]+([.][0-9]*)?|[.][0-9]+)[ij])$");   //for 
        std::regex yi_x_regex("^((-)?([0-9]+([.][0-9]*)?|[.][0-9]+)[ij])(([-+])?([0-9]+([.][0-9]*)?|[.][0-9]+))$");   //for 
        std::smatch match;
        if(std::regex_match(str, match, x_regex)){x = std::stod(match[0].str());}
        else if(std::regex_match(str, match, iy_regex)){y = (match[1].matched ? -1.0 : 1.0)*std::stod(match[2].str());}
        else if(std::regex_match(str, match, x_yi_regex))
        {
            x = std::stod(match[1].str());
            y = (match[6].matched && match[6].str() == "-" ? -1.0 : 1.0) * std::stod(match[7].str());
        }
        else if(std::regex_match(str, match, yi_x_regex))
        {
            x = (match[6].matched && match[6].str() == "-" ? -1.0 : 1.0) * std::stod(match[7].str());
            y = (match[2].matched && match[2].str() == "-" ? -1.0 : 1.0) * std::stod(match[3].str());
        }
        else{RAISE_EXCEPTION("Unable to obtaind complex value from string.  Invalid string pattern.");}
        
        return complex_type(x, y);
    }
};  //complex_from_string


template <typename T>
class parse_complex
{
public:
    static_assert(std::is_floating_point<T>::value, "Require a floating point real type when parsing complex numbers.");
    using real_type = T;
    using complex_type = linalg::complex<T>;
public:
    static bool is_complex(const rapidjson::Value& val)
    {
        if(val.IsNumber()){return true;}
        else if(val.IsObject())
        {
            if(val.HasMember("real") && val.HasMember("imag"))
            {
                if(val["real"].IsNumber() && val["imag"].IsNumber()){return true;}
            }
        }
        else if(val.IsString())
        {
            std::string s(val.GetString());
            remove_whitespace_and_to_lower(s);
            return complex_from_string<T>::is_valid(s);
        }
        return false;
    }

    static complex_type get_complex(const rapidjson::Value& val)
    {
        try
        {
            complex_type z;
            if(val.IsNumber())
            {
                real_type x = val.GetDouble();
                z = complex_type(x, 0);
            }
            else if(val.IsObject())
            {
                if(val.HasMember("real") && val.HasMember("imag"))
                {
                    if(val["real"].IsNumber() && val["imag"].IsNumber())
                    {
                        real_type x = val["real"].GetDouble();
                        real_type y = val["imag"].GetDouble();
                        z = complex_type(x, y);
                    }
                    else{RAISE_EXCEPTION("The real and imaginary parts of the object are not both numbers.");}
                }
                else{RAISE_EXCEPTION("The object did not contain both real and imaginary components.");}
            }
            else if(val.IsString())
            {
                std::string s(val.GetString());
                remove_whitespace_and_to_lower(s);
                z = complex_from_string<T>::get(s);
            }
            return z;
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to get complex number from rapidjson value object.");
        }
    }

};  //parse_complex



template <typename T>
class loader<linalg::complex<T>, typename std::enable_if<std::is_floating_point<T>::value, void>::type >
{
public:
    using complex_type = linalg::complex<T>;
public:
    template <typename obj>
    static bool is_type(const obj& val){return parse_complex<T>::is_complex(val);}
    template <typename obj>
    static complex_type load(const obj& val){CALL_AND_RETHROW(return parse_complex<T>::get_complex(val));}
    template <typename obj>
    static void load(const obj& val, complex_type& v){CALL_AND_RETHROW(v = parse_complex<T>::get_complex(val));}
};

template <typename T>
class loader<T, typename std::enable_if<std::is_floating_point<T>::value, void>::type >
{
public:
    using real_type = T;
public:
    template <typename obj>
    static bool is_type(const obj& val){return val.IsNumber();}
    template <typename obj>
    static real_type load(const obj& val){ASSERT(val.IsNumber(), "Failed to load real_type the input val object is not a number."); return val.GetDouble();}
    template <typename obj>
    static void load(const obj& val, real_type& v){ASSERT(val.IsNumber(), "Failed to load real_type the input val object is not a number."); v = val.GetDouble();}
};

template <>
class loader<std::string>
{
public:
    template <typename obj>
    static bool is_type(const obj& val){return val.IsString();}
    template <typename obj>
    static std::string load(const obj& val){ASSERT(val.IsString(), "Failed to load string the input val object is not a number."); return val.GetString();}
    template <typename obj>
    static void load(const obj& val, std::string& v){ASSERT(val.IsString(), "Failed to load string the input val object is not a number."); v = val.GetString();}
};

template <>
class loader<bool, void>
{
public:
    using int_type = bool;
public:
    template <typename obj>
    static bool is_type(const obj& val){return val.IsBool();}
    template <typename obj>
    static int_type load(const obj& val){ASSERT(val.IsBool(), "Failed to load int_type the input val object is not a number."); return val.GetBool();}
    template <typename obj>
    static void load(const obj& val, int_type& v){ASSERT(val.IsBool(), "Failed to load int_type the input val object is not a number."); v = val.GetBool();}
};

template <typename T>
class loader<T, typename std::enable_if<std::is_integral<T>::value, void>::type >
{
public:
    using int_type = T;
public:
    template <typename obj>
    static bool is_type(const obj& val){return val.IsInt64();}
    template <typename obj>
    static int_type load(const obj& val){ASSERT(val.IsInt64(), "Failed to load int_type the input val object is not a number."); return val.GetInt64();}
    template <typename obj>
    static void load(const obj& val, int_type& v){ASSERT(val.IsInt64(), "Failed to load int_type the input val object is not a number."); v = val.GetInt64();}
};

template <typename T>
class loader<std::vector<T>>
{
protected:
    using load_elements = loader<T>;
public:
    template <typename obj>
    static bool is_type(const obj& val)
    {
        if(val.IsArray())
        {
            for(size_t i=0; i < val.Size(); ++i)
            {
                if(!load_elements::is_type(val[i])){return false;}
            }
            return true;
        }
        else{return load_elements::is_type(val);}
    }

    template <typename obj>
    static std::vector<T> load(const obj& val)
    {
        std::vector<T> ret;
        CALL_AND_RETHROW(load(val, ret));
        return ret;
    }    


    template <typename obj>
    static void load(const obj& val, std::vector<T>& ret)
    {
        try
        {
            ASSERT(is_type(val), "Invalid rapidjson object.");
            if(val.IsArray())
            {
                ret.resize(val.Size());
                for(size_t i=0; i < ret.size(); ++i)
                {
                    CALL_AND_HANDLE(load_elements::load(val[i], ret[i]), "Failed to load element of vector."); 
                }
            }
            else
            {
                ret.resize(1);
                CALL_AND_HANDLE(load_elements::load(val, ret[0]), "Failed to load element of vector."); 
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to load vector from rapidjson value.");
        }
    }
};


template <typename T>
class loader<linalg::vector<T>>
{
protected:
    using load_elements = loader<T>;
public:
    template <typename obj>
    static bool is_type(const obj& val)
    {
        if(val.IsArray())
        {
            for(size_t i=0; i < val.Size(); ++i)
            {
                if(!load_elements::is_type(val[i])){return false;}
            }
            return true;
        }
        else{return load_elements::is_type(val);}
    }

    template <typename obj>
    static linalg::vector<T> load(const obj& val)
    {
        linalg::vector<T> ret;
        CALL_AND_RETHROW(load(val, ret));
        return ret;
    }    


    template <typename obj>
    static void load(const obj& val, linalg::vector<T>& ret)
    {
        try
        {
            ASSERT(is_type(val), "Invalid rapidjson object.");
            if(val.IsArray())
            {
                ret.resize(val.Size());
                for(size_t i=0; i < ret.size(); ++i)
                {
                    CALL_AND_HANDLE(load_elements::load(val[i], ret(i)), "Failed to load element of vector."); 
                }
            }
            else
            {
                ret.resize(1);
                CALL_AND_HANDLE(load_elements::load(val, ret(0)), "Failed to load element of vector."); 
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to load vector from rapidjson value.");
        }
    }
};


template <typename T>
class loader<linalg::diagonal_matrix<T>>
{
protected:
    using load_elements = loader<T>;
public:
    template <typename obj>
    static bool is_type(const obj& val)
    {
        if(val.IsArray())
        {
            for(size_t i=0; i < val.Size(); ++i)
            {
                if(!load_elements::is_type(val[i])){return false;}
            }
            return true;
        }
        else{return false;}
    }

    template <typename obj>
    static linalg::diagonal_matrix<T> load(const obj& val)
    {
        linalg::diagonal_matrix<T> ret;
        CALL_AND_RETHROW(load(val, ret));
        return ret;
    }    

    template <typename obj>
    static void load(const obj& val, linalg::diagonal_matrix<T>& ret)
    {
        try
        {
            ASSERT(is_type(val), "Invalid rapidjson object.");
            ret.resize(val.Size());
            for(size_t i=0; i < ret.size(); ++i)
            {
                CALL_AND_HANDLE(load_elements::load(val[i], ret(i, i)), "Failed to load element of vector."); 
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to load diagonal matrix from rapidjson value.");
        }
    }
};


template <typename T>
class loader<linalg::matrix<T>>
{
protected:
    using load_elements = loader<T>;

public:
    template <typename obj>
    static bool is_type(const obj& val)
    {
        if(val.IsArray())
        {
            size_t n = val.Size();
            size_t m;
            for(size_t i=0; i < n; ++i)
            {
                if(val[i].IsArray())
                {
                    if(i > 0)
                    {
                        if(val[i].Size() != m){return false;}
                    }
                    else{ m  = val[i].Size();}
                    for(size_t j = 0; j < m; ++j)
                    {
                        if(!load_elements::is_type(val[i][j])){return false;}
                    }
                }
                else{return false;}
            }
            return true;
        }
        else{return false;}
    }

    template <typename obj>
    static linalg::matrix<T> load(const obj& val)
    {
        linalg::matrix<T> ret;
        CALL_AND_RETHROW(load(val, ret));
        return ret;
    }    

    template <typename obj>
    static void load(const obj& val, linalg::matrix<T>& ret)
    {
        try
        {
            ASSERT(is_type(val), "Invalid rapidjson object.");
            
            size_t n = val.Size();
            size_t m = 0;
            if(n != 0){m = val[0].Size();}
            ret.resize(n, m);
            for(size_t i=0; i < n; ++i)
            {
                for(size_t j=0; j < m; ++j)
                {
                    CALL_AND_HANDLE(load_elements::load(val[i][j], ret(i, j)), "Failed to load element of matrix."); 
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to load matrix from rapidjson value.");
        }
    }
};

template <typename T, size_t D>
class loader<linalg::tensor<T, D>>
{
protected:
    using load_elements = loader<T>;


protected:
    template <typename obj>
    static bool is_type_slice(const obj& val, size_t m, size_t index)
    {
        if(val.IsArray())
        {
            size_t n = val.Size();
            if(n != m){return false;}

            for(size_t i=0; i < n; ++i)
            {
                if(val[i].IsArray())
                {
                    return is_type_slice(val[i], val[0].Size(), index+1);
                }
                else
                {
                    if(index+1 == D)
                    {
                        return load_elements::is_type(val[i]);
                    }
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    template <typename obj>
    static void load_slice(const obj& val, linalg::tensor<T, D>& ret)
    {
        size_t n = val.Size();
        for(size_t i=0; i < n; ++i)
        {
            load_slice(val[i], ret[i]);
        }
    }

    template <typename obj, size_t Dp>
    static void load_slice(const obj& val, linalg::tensor_slice<linalg::tensor<T, D>, T, Dp> ret)
    {
        size_t n = val.Size();
        for(size_t i=0; i < n; ++i)
        {
            load_slice(val[i], ret[i]);
        }
    }
    template <typename obj>
    static void load_slice(const obj& val, linalg::tensor_slice<linalg::tensor<T, D>, T, 1> ret)
    {
        size_t n = val.Size();
        for(size_t i=0; i < n; ++i)
        {
            CALL_AND_HANDLE(load_elements::load(val[i], ret[i]), "Failed to load element of matrix."); 
        }
    }

    template <typename obj>
    static void get_size(const obj& val, std::array<size_t, D>& size, size_t ind)
    {
        if(ind != D)
        {
            size[ind] = val.Size();
            get_size(val[0], size, ind+1);
        }

    }

public:
    template <typename obj>
    static bool is_type(const obj& val)
    {
        if(val.IsArray())
        {
            return is_type_slice(val, val.Size(), 0);
        }
        else{return false;}
    }

    template <typename obj>
    static linalg::tensor<T, D> load(const obj& val)
    {
        linalg::tensor<T, D> ret;
        CALL_AND_RETHROW(load(val, ret));
        return ret;
    }    

    template <typename obj>
    static void load(const obj& val, linalg::tensor<T, D>& ret)
    {
        try
        {
            ASSERT(is_type(val), "Invalid rapidjson object.");
            std::array<size_t, D> size; get_size(val, size, 0);
            ret.resize(size);
            load_slice(val, ret);
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to load matrix from rapidjson value.");
        }
    }
};


template <typename T>
class loader<linalg::csr_matrix<T>>
{
protected:
    using load_elements = loader<T>;
    using coo_type = typename linalg::csr_matrix<T>::coo_type;
    using index_type = typename linalg::csr_matrix<T>::index_type;
public:
    template <typename obj>
    static bool is_type(const obj& val)
    {
        if(!val.IsObject())
        {
            return false;
        }
        if(val.HasMember("coo"))
        {
            if(val["coo"].IsArray())
            {
                for(size_t i=0; i < val["coo"].Size(); ++i)
                {
                    if(val["coo"][i].IsArray())
                    {
                        if(val["coo"][i].Size() == 3)
                        {
                            if(!val["coo"][i][0].IsInt() || !val["coo"][i][1].IsInt() || !load_elements::is_type(val["coo"][i][2])){return false;}
                        }
                        else{return false;}
                    }
                    else{return false;}
                }
                if(val.HasMember("nrows"))
                {
                    if(!val["nrows"].IsInt()){return false;}
                    if(val.HasMember("ncols"))
                    {
                        if(!val["ncols"].IsInt()){return false;}
                    }
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }
        else if(val.HasMember("rowptr") && val.HasMember("colind") && val.HasMember("data")) 
        {
            if(val["rowptr"].IsArray()){for(size_t i = 0; i < val["rowptr"].Size(); ++i){if(!val["rowptr"][i].IsInt()){return false;}}}
            if(val["colind"].IsArray()){for(size_t i = 0; i < val["colind"].Size(); ++i){if(!val["colind"][i].IsInt()){return false;}}}
            if(val["buffer"].IsArray()){for(size_t i = 0; i < val["buffer"].Size(); ++i){if(!load_elements::is_type(val["buffer"][i])){return false;}}}
            if(val.HasMember("nrows"))
            {
                if(!val["nrows"].IsInt()){return false;}
            }

            if(val.HasMember("ncols"))
            {
                if(!val["ncols"].IsInt()){return false;}
            }
    
            if(val["colind"].Size() != val["buffer"].Size()){return false;}
        }
        else{return false;}
        return false;
    }

    template <typename obj>
    static linalg::vector<T> load(const obj& val)
    {
        linalg::csr_matrix<T> ret;
        CALL_AND_RETHROW(load(val, ret));
        return ret;
    }    

    template <typename obj>
    static void load(const obj& val, linalg::csr_matrix<T>& ret)
    {
        try
        {
            ASSERT(is_type(val), "Invalid rapidjson object.");
            
            if(val.HasMember("coo"))
            {
                index_type max_row = 0;
                index_type max_col = 0;
                coo_type coo(val["coo"].Size());

                
                for(size_t i = 0; i < val["coo"].Size(); ++i)
                {
                    index_type row, col;
                    T data;
                    row = val["coo"][i][0].GetInt64();
                    col = val["coo"][i][1].GetInt64();
                    ASSERT(row >= 0 && col >= 0, "Row and Column Indices must be positive.");
                    CALL_AND_HANDLE(load_elements::load(val["coo"][i][2], data), "Failed to load data.");
                    coo[i] = std::make_tuple(row, col, data);
                    if(row > max_row){max_row = row;}
                    if(col > max_col){max_col = col;}
                }

                size_t nrows = val["nrows"].GetInt64();
                size_t ncols = nrows;
                if(val.HasMember("ncols"))
                {
                    ncols = val["ncols"].GetInt64();
                }
                ASSERT(static_cast<size_t>(max_row) < nrows && static_cast<size_t>(max_col) < ncols , "An element is out of bounds");
                ret.init(coo, nrows, ncols);
            }
            else if(val.HasMember("rowptr") && val.HasMember("colind") && val.HasMember("data")) 
            {
                size_t nrows = val["rowptr"].Size()-1;
                int64_t _nnz = val["rowptr"][nrows].GetInt64();

                size_t size = nrows;
                if(val.HasMember("nrows"))
                {
                    size = val["nrows"].GetInt64();
                }

                size_t ncols = size;
                if(val.HasMember("ncols"))
                {
                    ncols = val["ncols"].GetInt64();
                }

                ASSERT(val["buffer"].Size() == _nnz, "Invalid sized buffer.");
                size_t nnz = _nnz;

                ret.resize(nnz, size, ncols);
    
                auto* buffer = ret.buffer();
                auto* colind = ret.colind();
                auto* rowptr = ret.rowptr();

                //read in the rowptr objects that have been specified
                for(size_t i = 0; i < nrows+1; ++i)
                {
                    index_type rp = val["rowptr"][i].GetInt64();
                    ASSERT(rp >= 0, "Failed to read in csr matrix a rowptr is negative.");
                    rowptr[i] = rp;
                }
                //and pad with the remaining values
                for(size_t i = nrows+1; i < size+1; ++i){rowptr[i] = nnz;}

                for(size_t i = 0; i < nnz; ++i)
                {
                    index_type col = val["colind"][i].GetInt64();
                    ASSERT(col < static_cast<index_type>(ncols) && col >= 0, "Failed to read in csr matrix a column is out of bounds.");
                    colind[i] = col;

                    T d;
                    CALL_AND_HANDLE(load_elements::load(val["buffer"][i], d), "Failed to load buffer.");
                    buffer[i] = d;
                }
            }
        }
        catch(const std::exception& ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to load csr matrix from rapidjson value.");
        }
    }
};

/*
 *  A class for loading a shared ptr that has had a self registering factory created for it
 */
template <typename T>
class loader<std::shared_ptr<T> >
{
public:
    using ftype = factory<T>;
public:
    template <typename obj>
    static bool is_type(const obj& val)
    {   
        return ftype::is_loadable(val);
    }

    template <typename obj>
    static std::shared_ptr<T> load(const obj& val)
    {
        CALL_AND_RETHROW(return ftype::create(val));
    }
    template <typename obj>
    static void load(const obj& val, std::shared_ptr<T>& v)
    {
        CALL_AND_RETHROW(v = ftype::create(val));
    }
};

}

template <>
class input_wrapper<rapidjson_tag>
{
public:
    using input_object = rapidjson::Value;
    using input_base = rapidjson::Document;

public:
    template <typename U, typename V>
    static void dfs_replace_space_and_capitals_in_key(U& value, V& allocator)
    {
        if(value.IsObject())
        {
            for(rapidjson::Value::MemberIterator itr = value.MemberBegin(); itr != value.MemberEnd(); ++itr)
            {
                std::string s(itr->name.GetString());
                remove_whitespace_and_to_lower(s);
                itr->name.SetString(s.c_str(), allocator);
                dfs_replace_space_and_capitals_in_key(itr->value, allocator);
            }
        }
        else if(value.IsArray())
        {
            for(rapidjson::Value::ValueIterator itr = value.Begin(); itr != value.End(); ++itr)
            {
                dfs_replace_space_and_capitals_in_key(*itr, allocator);
            }
        }
    }

    static input_base& parse_stream(input_base& base, std::istream& stream)
    {
        rapidjson::IStreamWrapper isw{stream};
        base.ParseStream(isw);

        if(base.HasParseError())
        {
            std::ostringstream oss;

            oss << "Failed to parse input stream. Failed with Error: " << base.GetParseError() << " at offset: " 
                << base.GetErrorOffset() << ".  Ensure that the input file is a correctly formatted json object.";

            throw std::runtime_error(oss.str());
        }
        
        //now iterate over the 
        dfs_replace_space_and_capitals_in_key(base, base.GetAllocator());
        return base;
    }


    static input_base& parse(input_base& base, const std::string& str)
    {
        base.Parse(str.c_str());
    
        if(base.HasParseError())
        {
            std::ostringstream oss;

            oss << "Failed to parse string. Failed with Error: " << base.GetParseError() << " at offset: " 
                << base.GetErrorOffset() << ".  Ensure that the input string is a correctly formatted json object.";

            throw std::runtime_error(oss.str());
        }
        
        //now iterate over the 
        dfs_replace_space_and_capitals_in_key(base, base.GetAllocator());
        return base;
    }

    template <typename type, typename T, typename Iobj>
    static void load(const Iobj& obj, T&& name, type& v)
    {
        ASSERT(is_type<type>(obj, std::forward<T>(name)), "Variable not found.");
        CALL_AND_RETHROW(rapidjson_loader::loader<type>::load(obj[std::forward<T>(name)], v));
    }
    template <typename type, typename T, typename Iobj>
    static type load(const Iobj& obj, T&& name)
    {
        ASSERT(is_type<type>(obj, std::forward<T>(name)), "Variable not found.");
        CALL_AND_RETHROW(return rapidjson_loader::loader<type>::load(obj[std::forward<T>(name)]));
    }


    template <typename type, typename T, typename Iobj>
    static bool load_optional(const Iobj& obj, T&& name, type& v)
    {
        if(has_member(obj, std::forward<T>(name)))
        {
            CALL_AND_RETHROW(rapidjson_loader::loader<type>::load(obj[std::forward<T>(name)], v));
            return true;
        }
        return false;
    }

    template <typename T, typename Iobj>
    static const input_object& get(const Iobj& obj, T&& name)
    {
        ASSERT(has_member(obj, std::forward<T>(name)) , "Variable not found.");
        return obj[std::forward<T>(name)];
    }
    template <typename Iobj>
    static const input_object& get(const Iobj& obj, size_t name)
    {
        ASSERT(obj.IsArray() , "Variable not found.");
        ASSERT(name < obj.Size() , "Variable not found.");
        return obj[name];
    }

    template <typename T, typename Iobj>
    static size_t size(const Iobj& obj, T&& name)
    {
        ASSERT(is_array(obj, std::forward<T>(name)) , "Variable not found.");
        return obj[std::forward<T>(name)].Size();
    }


    template <typename Iobj, typename T>
    static bool has_member(const Iobj& obj, T&& name)
    {
        return obj.HasMember(std::forward<T>(name));
    }

    template <typename Iobj>
    static bool has_member(const Iobj& obj, size_t name)
    {
        if(!obj.IsArray())
        {
            return false;
        }
        else
        {
            return obj.Size() > name;
        }
    }

    template <typename type, typename T, typename Iobj> 
    static bool is_type(const Iobj& obj, T&& name)
    {
        ASSERT(has_member(obj, std::forward<T>(name)), "Variable not found.");
        return rapidjson_loader::loader<type>::is_type(obj[std::forward<T>(name)]);
    }

    template <typename Iobj, typename T> 
    static bool is_object(const Iobj& obj, T&& name)
    {
        ASSERT(has_member(obj, std::forward<T>(name)), "Variable not found.");
        return obj[std::forward<T>(name)].IsObject();
    }

    template <typename Iobj> 
    static bool is_object(const Iobj& obj)
    {
        return obj.IsObject();
    }


    template <typename Iobj, typename T> 
    static bool is_array(const Iobj& obj, T&& name)
    {
        ASSERT(has_member(obj, std::forward<T>(name)), "Variable not found.");
        return obj[std::forward<T>(name)].IsArray();
    }
};

}
}
using IOWRAPPER = utils::io::input_wrapper<io::rapidjson_tag>;

#endif //USE_RAPIDJSON//

#endif


