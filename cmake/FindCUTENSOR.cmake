
#import the cuda tool kit library
find_package(CUDAToolkit REQUIRED)

#now attempt to find cutensor files
find_path(CUTENSOR_INCLUDE_PATH cutensor.h
    HINTS ${CUTENSOR_ROOT_DIR}/include $ENV{CUTENSOR_ROOT_DIR}/include ${CUDA_TOOLKIT_ROOT_DIR}/include $ENV{CUDA_TOOLKIT_ROOT_DIR}/include $ENV{INCLUDE_PATH} ${CUTENSOR_INCLUDE} usr/local/cutensor usr usr/local /usr/include
    NO_DEFAULT_PATH
    DOC "Path to cutensor includes"

)

message(STATUS ${CUTENSOR_INCLUDE_PATH} ${CUDA_TOOLKIT_ROOT_DIR})

find_library(CUTENSOR_LIBRARY NAMES libcutensor.so
    HINTS ${CUTENSOR_ROOT_DIR}lib64/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} ${CUTENSOR_ROOT_DIR}/lib64/${CUDAToolkit_VERSION_MAJOR}
    ${CUTENSOR_ROOT_DIR}lib/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} ${CUTENSOR_ROOT_DIR}/lib/${CUDAToolkit_VERSION_MAJOR}
    $ENV{CUTENSOR_ROOT_DIR}lib64/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} $ENV{CUTENSOR_ROOT_DIR}/lib64/${CUDAToolkit_VERSION_MAJOR}
    $ENV{CUTENSOR_ROOT_DIR}lib/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} $ENV{CUTENSOR_ROOT_DIR}/lib/${CUDAToolkit_VERSION_MAJOR}
    ${CUDA_TOOLKIT_ROOT_DIR}lib64/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} ${CUDA_TOOLKIT_ROOT_DIR}/lib64/${CUDAToolkit_VERSION_MAJOR}
    ${CUDA_TOOLKIT_ROOT_DIR}lib/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} ${CUDA_TOOLKIT_ROOT_DIR}/lib/${CUDAToolkit_VERSION_MAJOR}
    $ENV{CUDA_TOOLKIT_ROOT_DIR}lib64/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} $ENV{CUDA_TOOLKIT_ROOT_DIR}/lib64/${CUDAToolkit_VERSION_MAJOR}
    $ENV{CUDA_TOOLKIT_ROOT_DIR}lib/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} $ENV{CUDA_TOOLKIT_ROOT_DIR}/lib/${CUDAToolkit_VERSION_MAJOR}
    $ENV{LD_LIBRARY_PATH} ${CUTENSOR_LIBRARY_PATH}  lib lib/x64  cuda/lib cuda/lib64 lib/x64 /usr/local/cutensor /usr/lib /usr/lib/x86_64-linux-gnu
    lib64/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} lib64/${CUDAToolkit_VERSION_MAJOR} lib/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} lib/${CUDAToolkit_VERSION_MAJOR} lib64  lib
    libcutensor/${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} libcutensor/${CUDAToolkit_VERSION_MAJOR} 
    NO_DEFAULT_PATH
    DOC "Path to cutensor library"
)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  CUTENSOR 
  "Failed to find cuTENSOR Library. If this occurs please specify the CUTENSOR_ROOT_DIR environment variable."
  CUTENSOR_LIBRARY CUTENSOR_INCLUDE_PATH)

if (NOT TARGET CUDA::cutensor)

  add_library(CUDA::cutensor INTERFACE IMPORTED)

  set_property(TARGET CUDA::cutensor PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_INCLUDE_PATH}")

  set_property(TARGET CUDA::cutensor PROPERTY
    INTERFACE_LINK_LIBRARIES "${CUTENSOR_LIBRARY}")

endif (NOT TARGET CUDA::cutensor)