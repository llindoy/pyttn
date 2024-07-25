#ifndef COMMON_EXCEPTION_HANDLING_MACROS_HPP_
#define COMMON_EXCEPTION_HANDLING_MACROS_HPP_

#include <stdexcept>
#include <iostream>

namespace common
{
    struct invalid_value : public std::exception
    {
        std::string m_msg;
        invalid_value(const char* msg) : m_msg(msg) {}
        invalid_value(const std::string& msg) : m_msg(msg) {}
        virtual const char* what() const throw(){ return m_msg.c_str();}
    };
}


#ifndef LINALG_EXCEPTION_HANDLING_MACROS_HPP
#define DEBUG

#define STRINGIFY_MACRO(x) #x
#define TOSTRING_MACRO(x) STRINGIFY_MACRO(x)
#define AT_MACRO(message) __FILE__ ":" TOSTRING_MACRO(__LINE__) ":" message

#if defined(NOEXCEPT)
#define RAISE_EXCEPTION_MESSSTR(error_string, str)                  
#define RAISE_EXCEPTION_STR(str)                  
#define RAISE_EXCEPTION(error_string)                  
#define RAISE_NUMERIC_MESSSTR(error_string, str)                  
#define RAISE_NUMERIC_STR(str)                  
#define RAISE_NUMERIC(error_string)                                   

#define CALL_AND_HANDLE(expr, error_string)                             \
    expr;                                        

#define CALL_AND_RETHROW(expr)                                          \
    expr;                                        

#else
#define RAISE_EXCEPTION_MESSSTR(error_string, str)                      \
    throw std::runtime_error(std::string(AT_MACRO(error_string))        \
        + std::string(str));

#define RAISE_EXCEPTION_STR(str)                                        \
    throw std::runtime_error(std::string(AT_MACRO(""))                  \
        + std::string(str));

#define RAISE_EXCEPTION(error_string)                                   \
    throw std::runtime_error(AT_MACRO(error_string));

#define RAISE_NUMERIC_MESSSTR(error_string, str)                      \
    throw common::invalid_value(std::string(AT_MACRO(error_string))   \
        + std::string(str));

#define RAISE_NUMERIC_STR(str)                                        \
    throw common::invalid_value(std::string(AT_MACRO("Invalid value encountered when "))                  \
        + std::string(str));

#define RAISE_NUMERIC(error_string)                                     \
    throw common::invalid_value(AT_MACRO("Invalid value encountered when " error_string));

#define CALL_AND_HANDLE(expr, error_string)                                     \
    try{ expr; }                                                                \
    catch(const common::invalid_value& ex)                                      \
    {                                                                           \
        std::cerr << ex.what() << std::endl;                                    \
        throw common::invalid_value(AT_MACRO("Invalid value: " error_string));   \
    }                                                                           \
    catch(const std::exception& ex)                                             \
    {                                                                           \
        std::cerr << ex.what() << std::endl;                                    \
        throw std::runtime_error(AT_MACRO(error_string));                       \
    }                                                                           

#define CALL_AND_RETHROW(expr)                                          \
    try{ expr; }                                                        \
    catch(...){throw;}                                                  
#endif


#if !defined(DEBUG) || defined(NOEXCEPT)
#define ASSERT(cond, error_string)                          

#define ASSERT_NUMERIC(cond, error_string)

#else
#define ASSERT(cond, error_string)                                      \
    if(!(cond))                                                         \
    {                                                                   \
        throw std::runtime_error(AT_MACRO("Assertion failed "           \
                TOSTRING_MACRO(cond) ": " error_string));               \
    }   

#define ASSERT_NUMERIC(cond, error_string)                              \
    if(!(cond))                                                         \
    {                                                                   \
        throw common::invalid_value(AT_MACRO("Assertion failed "        \
                TOSTRING_MACRO(cond) ": " error_string));               \
    }   
#endif

#endif
#endif  //COMMON_EXCEPTION_HANDLING_MACROS_HPP_//
