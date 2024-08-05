cmake_minimum_required (VERSION 3.12)
project(mlmctdh LANGUAGES C CXX)

if(MSVC)
    message(FATAL_ERROR "Request MSVC compiler.  I don't know how this works so exiting.")
else()
    set(CMAKE_CXX_FLAGS "-O3 -Wall -g -Wextra -Wshadow -pedantic -Wold-style-cast ${CMAKE_CXX_FLAGS}")
    option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" ON)
    if(WITH_WERROR)
        #set(CMAKE_CXX_FLAGS "-Werror ${CMAKE_CXX_FLAGS}")
    endif(WITH_WERROR)

    #additional compilation flags for GNU compilers
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -fwhole-program -fcx-fortran-rules")
        if(USE_OPENMP)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        endif()
    #additional compilation flags for inteal compilers
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=3 -ipo -O3 -xHost -restrict")
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
set(COMPILER_SET TRUE INTERNAL BOOL "")


get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
add_executable(siam_test siam.cpp)
add_executable(pyrazine_test pyrazine_operator.cpp)
add_executable(spin_boson spin_boson.cpp)
add_executable(spin_boson_multiset spin_boson_multiset.cpp)


target_include_directories(siam_test PRIVATE ${TTNS_INCLUDE_DIRECTORIES})
target_link_libraries(siam_test PRIVATE ${LINALG_LIBS} ${LIBS}) 

install(TARGETS siam_test
        DESTINATION ${CMAKE_SOURCE_DIR}/bin)

target_include_directories(spin_boson PRIVATE ${TTNS_INCLUDE_DIRECTORIES})
target_link_libraries(spin_boson PRIVATE ${LINALG_LIBS} ${LIBS}) 

install(TARGETS spin_boson
  DESTINATION ${CMAKE_SOURCE_DIR}/bin)

target_include_directories(spin_boson_multiset PRIVATE ${TTNS_INCLUDE_DIRECTORIES})
target_link_libraries(spin_boson_multiset PRIVATE ${LINALG_LIBS} ${LIBS}) 

install(TARGETS spin_boson_multiset
  DESTINATION ${CMAKE_SOURCE_DIR}/bin)

target_include_directories(pyrazine_test PRIVATE ${TTNS_INCLUDE_DIRECTORIES})
target_link_libraries(pyrazine_test PRIVATE ${LINALG_LIBS} ${LIBS}) 

install(TARGETS pyrazine_test
        DESTINATION ${CMAKE_SOURCE_DIR}/bin)
