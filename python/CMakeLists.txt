cmake_minimum_required (VERSION 3.12)
project(_pyttn)

include(CheckIPOSupported)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

#compiler flags
option(USE_CUDA "Compile with cuda" OFF)
option(USE_OPENMP "Compile with openmp" OFF)
option(USE_GEMM3M "Compile with openmp" OFF)
option(EVAL_BUT_DONT_APPLY "Compile with eval but don't apply option set" OFF)

set(PYTHON_BINDINGS_SRC pythonBinding.cpp 
          utils/orthopol.cpp utils/discretisation.cpp
          linalg/sparseMatrix.cpp linalg/tensor.cpp 
          ttns/observables/matrix_element.cpp 
          ttns/operators/siteOperators.cpp ttns/operators/product_operator.cpp ttns/operators/sop_operator.cpp
          ttns/sop/sSOP.cpp ttns/sop/SOP.cpp ttns/sop/system_information.cpp ttns/sop/operator_dictionary.cpp ttns/sop/models/models.cpp ttns/sop/liouville_space.cpp
          ttns/ttn/ntree.cpp ttns/ttn/ttn.cpp ttns/ttn/ms_ttn.cpp
          ttns/algorithms/tdvp.cpp ttns/algorithms/dmrg.cpp
          )


if(COMPILER_SET)
else()
    if(MSVC)
        message(FATAL_ERROR "Request MSVC compiler.  I don't know how this works so exiting.")
    else()
        set(CMAKE_CXX_FLAGS "-O3 -g -Wall -Wextra -Wshadow -pedantic -Wold-style-cast")
        option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" ON)
        if(WITH_WERROR)
            #set(CMAKE_CXX_FLAGS "-Werror ${CMAKE_CXX_FLAGS}")
        endif(WITH_WERROR)

        #additional compilation flags for GNU compilers
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcx-fortran-rules -march=native -mtune=native")
            if(USE_OPENMP)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
            endif()
        #additional compilation flags for inteal compilers
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=3 -O3 -xHost -restrict")
            message(STATUS "using intel compiler.")
            if(USE_OPENMP)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
            endif()

        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -qopt-report=3 ")
            if(USE_OPENMP)
                message(STATUS "Using Openmp")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
            endif()

        elseif ( ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang") )
            find_package(LLVM REQUIRED)
            find_package(Clang REQUIRED)

            set(LINALG_LIBS ${LINALG_LIBS} ${LLVM_LIBRARIES} ${CLANG_LIBS} ${CLANG_CLANG_LIBS})
            

            #if we are using clang then find threads
            set(CMAKE_THREAD_LIBS_INIT "-lpthread")
            set(CMAKE_HAVE_THREADS_LIBRARY 1)
            set(CMAKE_USE_WIN32_THREADS_INIT 0)
            set(CMAKE_USE_PTHREADS_INIT 1)
            set(THREADS_PREFER_PTHREAD_FLAG ON)

            set(CMAKE_CXX_FLAGS "-stdlib=libc++ ${CMAKE_CXX_FLAGS}")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ ")


            if(USE_OPENMP)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libomp")
            endif()
        endif()
        

        if(CMAKE_VERSION VERSION_LESS 3.1)
            set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
        else()
          if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD STREQUAL "98")
            set(CMAKE_CXX_STANDARD 17)
          endif()

          if(CMAKE_CXX_STANDARD GREATER 17)
            cmake_minimum_required(VERSION 3.8)
          endif()

          set(CMAKE_CXX_STANDARD_REQUIRED ON)
        endif()
    endif()
endif()

set(CMAKE_BUILD_PARALLEL_LEVEL 4)
set(PYBIND11_ROOT_DIR ${CMAKE_SOURCE_DIR}/external/Pybind11)
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
    SOURCE_DIR ${PYBIND11_ROOT_DIR}
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
add_definitions(-DPYBIND11)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "${PYTHON_BINDINGS_SRC}")
pybind11_add_module(_pyttn ${PYTHON_BINDINGS_SRC})
target_include_directories(_pyttn PUBLIC ${TTNS_INCLUDE_DIRECTORIES})
target_link_libraries(_pyttn PUBLIC pybind11::module ${LINALG_LIBS} ${LIBS}) 

set_target_properties(_pyttn PROPERTIES
                            VISIBILITY_INLINES_HIDDEN ON)
check_ipo_supported(RESULT ipo)
if(ipo)
set_property(TARGET _pyttn PROPERTY VISIBILITY_INLINES_HIDDEN TRUE)
endif()
if(EVAL_BUT_DONT_APPLY)
  target_compile_definitions(_pyttn PUBLIC ALLOW_EVAL_BUT_DONT_APPLY=0)
endif()

install(
    TARGETS _pyttn
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)




