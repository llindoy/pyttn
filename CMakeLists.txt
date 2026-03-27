cmake_minimum_required(VERSION 3.18)
project(ttnpp_project LANGUAGES CXX)

# -------------------- Install Location -----------
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/pyttn)


# -------------------- Options --------------------
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(USE_CUDA "Enable CUDA support" ON)
option(USE_OPENMP "Enable OpenMP support" ON)
option(BUILD_REAL_TTN "Build real-valued TTN objects" OFF)
option(DEBUG "Enable debug messages" OFF)
option(USE_SPDLOG "Enable spdlog" ON)
option(USE_GEMM3M "Enable GEMM3M optimization" OFF)

# -------------------- Compiler standards --------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------- Dependencies --------------------
include(FetchContent)
set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)


# =========================================================
# pybind11
# =========================================================

set(TTNPP_EXTERNAL_INCLUDE_DIRS "")
if(BUILD_PYTHON_BINDINGS)
    set(PYBIND11_ROOT_DIR ${CMAKE_SOURCE_DIR}/external/Pybind11)

    if(NOT EXISTS ${PYBIND11_ROOT_DIR})
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
            SOURCE_DIR ${PYBIND11_ROOT_DIR}
        )

    else()
        message(STATUS "${PYBIND11_ROOT_DIR}")
        FetchContent_Declare(
            pybind11 SOURCE_DIR ${PYBIND11_ROOT_DIR}
        )
    endif()

    FetchContent_GetProperties(pybind11)
    if(NOT pybind11_POPULATED)
        FetchContent_Populate(pybind11)
        add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    endif()
    add_definitions(-DPYBIND11)
    # Extract include dirs
    get_target_property(PYBIND11_INCLUDE_DIRS pybind11::headers INTERFACE_INCLUDE_DIRECTORIES)

endif()


# =========================================================
# cereal (header-only)
# =========================================================
set (BUILD_DOC OFF CACHE INTERNAL "Turn off cereal docs")
set (BUILD_SANDBOX OFF CACHE INTERNAL "Turn off cereal sandbox")
set (SKIP_PERFORMANCE_COMPARISON ON CACHE INTERNAL "Turn on skipping cereal/boost performance comparison")
set(CEREAL_ROOT_DIR ${CMAKE_SOURCE_DIR}/external/cereal)
if(NOT EXISTS ${CEREAL_ROOT_DIR})
    FetchContent_Declare(
        cereal
        GIT_REPOSITORY "https://github.com/USCiLab/cereal.git"
        GIT_TAG v1.3.2
        SOURCE_DIR ${CEREAL_ROOT_DIR}
        PATCH_COMMAND git apply ${CMAKE_SOURCE_DIR}/patches/patch_cereal.patch
        UPDATE_DISCONNECTED 1
    )
else()
    message(STATUS "${CEREAL_ROOT_DIR}")
    FetchContent_Declare(
        cereal SOURCE_DIR ${CEREAL_ROOT_DIR}
    )
endif()
FetchContent_GetProperties(cereal)
if(NOT cereal_POPULATED)
    FetchContent_Populate(cereal)
    add_subdirectory(${cereal_SOURCE_DIR} ${cereal_BINARY_DIR})
endif()
add_definitions(-DCEREAL_LIBRARY_FOUND)
set(cereal_INCLUDE_DIR ${cereal_SOURCE_DIR}/include)
message(STATUS "${cereal_INCLUDE_DIR}")


# cereal is header-only — its include dir is:
set(CEREAL_INCLUDE_DIRS
    $<BUILD_INTERFACE:${cereal_SOURCE_DIR}/include>
)


# =========================================================
# spdlog
# =========================================================
if(USE_SPDLOG)

    if(EXISTS ${EXTERNAL_DIR}/spdlog/CMakeLists.txt)
        message(STATUS "Using local spdlog from external/")
        add_subdirectory(${EXTERNAL_DIR}/spdlog)
    else()
        message(STATUS "Fetching spdlog from GitHub")
        FetchContent_Declare(
            spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG v1.x
        )
        FetchContent_MakeAvailable(spdlog)
    endif()

    add_definitions(-DSPDLOG_LIBRARY_FOUND)

    get_target_property(SPDLOG_INCLUDE_DIRS spdlog::spdlog INTERFACE_INCLUDE_DIRECTORIES)

endif()

# -------------------- OpenMP --------------------
if(USE_OPENMP)
    find_package(OpenMP)
endif()

# -------------------- BLAS/LAPACK --------------------
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

# -------------------- External module path --------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# -------------------- CUDA --------------------
if(USE_CUDA)
    # Set default CUDA architectures if not defined
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 CACHE STRING "CUDA architectures to build for" FORCE)
        message(STATUS "CMAKE_CUDA_ARCHITECTURES not set. Defaulting to: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()

    # Enable CUDA language
    enable_language(CUDA)

    # Find CUDA toolkit
    find_package(CUDAToolkit REQUIRED)

    # Find cuTENSOR using custom FindCUTENSOR.cmake
    find_package(CUTENSOR REQUIRED)
endif()

# -------------------- Subdirectories --------------------
add_subdirectory(ttnpp)
#add_subdirectory(cuda)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

if(BUILD_SRC)
    if(USE_CUDA)
        add_subdirectory(cuda_test)
    else()
        add_subdirectory(src)
    endif()
endif()