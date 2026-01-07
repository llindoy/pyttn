cmake_minimum_required(VERSION 3.12)
project(operator_test)

include(CheckIPOSupported)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)


option(BUILD_SIAM "build siam" OFF)

if(USE_GEMM3M)
    add_compile_definitions(USE_GEMM3M)
endif()


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
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcx-fortran-rules -flto=auto")
        #additional compilation flags for inteal compilers
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=3 -O3 -xHost -restrict")
            message(STATUS "using intel compiler.")
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -qopt-report=3 ")

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

            set(CMAKE_CXX_FLAGS "-stdlib=libc++ -flto=thin ${CMAKE_CXX_FLAGS} ")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ ")

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
add_executable(operator_test.x operator_test.cpp)


target_include_directories(operator_test.x PUBLIC ${TTNS_INCLUDE_DIRECTORIES})
target_link_libraries(operator_test.x PRIVATE ${LINALG_LIBS} ${LIBS} ${OpenMP_LIBS}) 

check_ipo_supported(RESULT ipo)
if(ipo)
set_property(TARGET operator_test.x PROPERTY VISIBILITY_INLINES_HIDDEN TRUE)
endif()

install(TARGETS operator_test.x
        DESTINATION ${CMAKE_SOURCE_DIR}/bin)