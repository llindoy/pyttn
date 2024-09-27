cmake_minimum_required (VERSION 3.12)
project(ttns_lib)


set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/pyttn)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{CMAKE_TOOLCHAIN_FILE})
    set(CMAKE_TOOLCHAIN_FILE $ENV{CMAKE_TOOLCHAIN_FILE})
endif()

add_library(ttns_lib INTERFACE)
target_include_directories(ttns_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)


get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

#compiler flags
option(USE_CUDA "Compile with cuda" OFF)
option(USE_OPENMP "Compile with openmp" ON)
option(USE_GEMM3M "Compile with openmp" OFF)
option(BUILD_TESTS "Build Tests" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python Bindings" OFF)
option(BUILD_SRC "Build C++ src" ON)

if(USE_OPENMP)
    add_compile_definitions(USE_OPENMP)
endif()

if(USE_GEMM3M)
    add_compile_definitions(USE_GEMM3M)
endif()

set(TTNS_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/include CACHE INTERNAL "")

#BLAS AND LAPACK VENDORS
if(DEFINED BLA_VENDOR)
    if(BLA_VENDOR MATCHES "Intel")
        find_package(LAPACK)
        find_package(BLAS)
        if(BLAS_FOUND)
            set(MKL_INCLUDE_DIRS ${MKLROOT}/include)
            #FIND_PATH(MKL_INCLUDE_DIR "mkl_cblas.h" PATHS ${MKLROOT}/include)
            message(STATUS ${MKL_INCLUDE_DIRS})
            add_compile_definitions(USE_MKL)
            add_compile_definitions(BLAS_HEADER_INCLUDED)
            add_compile_definitions(NOT_ALLOWS_RETURN_TYPE )
            add_compile_definitions(BLAS_NO_TRAILING_UNDERSCORE )
            if(DLA_VENDOR MATCHES "64ilp")
                add_compile_definitons(BLAS_64_BIT)
            endif()
            set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} CACHE INTERNAL "")
            include_directories(SYSTEM ${MKL_INCLUDE_DIRS})
            set(USE_SPARSE_BLAS OFF)
        else()
            set(USE_DEFAULT_FIND_BLAS ON)
        endif()
    elseif(BLA_VENDOR MATCHES "OpenBLAS")   
        add_compile_definitions(USE_OPENBLAS)
        set(USE_DEFAULT_FIND_BLAS ON)
    elseif(BLA_VENDOR MATCHES "FLAME")
        #here we allow for us to handle the multithreaded case.  This is not dealt with by the standard FindBLAS and FindLapack files.  
        #this has been setup to work with the aocl versions of blis and flame
        if(USE_OPENMP)
            #first we attempt to find the standard FLAME LIBRARY
            set(BLA_VENDOR "FLAME")
            include(FindBLAS)
            if(NOT BLAS_FOUND)
                message(FATAL_ERROR "BLAS NOT FOUND")
            endif()
            check_blas_libraries(
              BLAS_LIBRARIES
              BLAS
              sgemm
              ""
              "blis-mt"
              ""
              ""
              ""
              )
            message(STATUS ${BLAS_LIBRARIES})
            set(BLA_VENDOR "FLAME")
            find_package(LAPACK REQUIRED)
            set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} CACHE INTERNAL "")
            set(USE_DEFAULT_FIND_BLAS OFF)
        else()
            add_compile_definitions(USE_FLAME)
            find_package(BLAS REQUIRED)
            find_package(LAPACK REQUIRED)
            message(STATUS ${BLAS_LIBRARIES})
            set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} CACHE INTERNAL "")
            set(USE_DEFAULT_FIND_BLAS OFF)
        endif()
    else()
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
        message(STATUS ${BLAS_LIBRARIES})
        set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} CACHE INTERNAL "")
        set(USE_DEFAULT_FIND_BLAS OFF)
    endif()
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    message(STATUS ${BLAS_LIBRARIES})
    set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} CACHE INTERNAL "")
    set(USE_DEFAULT_FIND_BLAS OFF)
endif()


if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

if(BUILD_SRC)
    add_subdirectory(src)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
 


