cmake_minimum_required(VERSION 3.12)
project(operator_test)

include(CheckIPOSupported)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)


option(USE_CUDA "Compile with cuda" OFF)
option(USE_OPENMP "Compile with openmp" OFF)
message(STATUS "using openmp" ${USE_OPENMP})
option(USE_GEMM3M "Compile with openmp" OFF)
option(BUILD_SIAM "build siam" OFF)

if(USE_OPENMP)
    add_compile_definitions(USE_OPENMP)
endif()

if(USE_GEMM3M)
    add_compile_definitions(USE_GEMM3M)
endif()


if(COMPILER_SET)
else()
    if(MSVC)
        message(FATAL_ERROR "Request MSVC compiler.  I don't know how this works so exiting.")
    else()
        set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -Wshadow -pedantic -Wold-style-cast")
        option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" ON)
        if(WITH_WERROR)
            #set(CMAKE_CXX_FLAGS "-Werror ${CMAKE_CXX_FLAGS}")
        endif(WITH_WERROR)

        #additional compilation flags for GNU compilers
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcx-fortran-rules -march=native -mtune=native -flto=auto")
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
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libomp -flto=thin")
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
message(STATUS ${CMAKE_CXX_FLAGS})
add_executable(operator_test.x operator_gen.cpp)



#pybind11_add_module(tree_test pythonBinding.cpp)


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
            set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
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
            set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
            set(USE_DEFAULT_FIND_BLAS OFF)
        else()
            add_compile_definitions(USE_FLAME)
            find_package(BLAS REQUIRED)
            find_package(LAPACK REQUIRED)
            message(STATUS ${BLAS_LIBRARIES})
            set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
            set(USE_DEFAULT_FIND_BLAS OFF)
        endif()
    else()
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
        message(STATUS ${BLAS_LIBRARIES})
        set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        set(USE_DEFAULT_FIND_BLAS OFF)
    endif()
else()
    set(USE_DEFAULT_FIND_BLAS ON)
endif()

#if we aren't dealing with one of the vendors we treat specially then we will just go ahead and use the default
#findBLAS and findLAPACK calls
if(USE_DEFAULT_FIND_BLAS)
    find_package(BLAS)
    if(NOT BLAS_FOUND)
        set(BLAS_LIBRARIES "-lblas")
    endif()

    find_package(LAPACK)
    if(NOT LAPACK_FOUND)
        set(LAPACK_LIBRARIES "-llapack -lm")
    endif()
    set(LINALG_LIBS ${LINALG_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

target_include_directories(operator_test.x PUBLIC ${TTNS_INCLUDE_DIRECTORIES})
target_link_libraries(operator_test.x PRIVATE ${LINALG_LIBS} ${LIBS}) 

check_ipo_supported(RESULT ipo)
if(ipo)
set_property(TARGET operator_test.x PROPERTY VISIBILITY_INLINES_HIDDEN TRUE)
endif()

install(TARGETS operator_test.x
        DESTINATION ${CMAKE_SOURCE_DIR}/bin)