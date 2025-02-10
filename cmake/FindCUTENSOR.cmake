
#import the cuda tool kit library
find_package(CUDAToolkit REQUIRED)

#now attempt to find cutensor files
find_path(CUTENSOR_INCLUDE_PATH cutensor.h
    HINTS ${CUTENSOR_ROOT_DIR} $ENV{CUTENSOR_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CUDA_TOOLKIT_ROOT_DIR} $ENV{INCLUDE_PATH} ${CUTENSOR_INCLUDE} 
    NO_DEFAULT_PATH
    DOC "Path to cutensor includes"

)

message(STATUS ${CUTENSOR_INCLUDE_PATH} ${CUDA_TOOLKIT_ROOT_DIR})

find_library(CUTENSOR_LIBRARY NAMES libcutensor.so
    HINTS ${CUTENSOR_ROOT_DIR} $ENV{CUTENSOR_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CUDA_TOOLKIT_ROOT_DIR} $ENV{LD_LIBRARY_PATH} ${CUTENSOR_LIBRARY_PATH} 
    PATH_SUFFIXES lib lib/x64  cuda/lib cuda/lib64 lib/x64 
    lib64/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} lib64/${CUDAToolkit_VERSION_MAJOR} lib/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} lib/${CUDAToolkit_VERSION_MAJOR} lib64  lib
    libcutensor/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} libcutensor/${CUDAToolkit_VERSION_MAJOR} 
    NO_DEFAULT_PATH
    DOC "Path to cutensor library"
)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUTENSOR DEFAULT_MSG CUTENSOR_LIBRARY CUTENSOR_INCLUDE_PATH)

if (NOT TARGET CUDA::cutensor)

  add_library(CUDA::cutensor INTERFACE IMPORTED)

  set_property(TARGET CUDA::cutensor PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_INCLUDE_PATH}")

  set_property(TARGET CUDA::cutensor PROPERTY
    INTERFACE_LINK_LIBRARIES "${CUTENSOR_LIBRARY}")

endif (NOT TARGET CUDA::cutensor)