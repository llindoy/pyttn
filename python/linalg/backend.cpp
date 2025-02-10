#include "backend.hpp"

#include <linalg/linalg.hpp>
#include <sstream>


void initialise_blas_backend(py::module& m)
{
    using namespace linalg;

    using size_type = blas_backend::size_type;

    //expose the ttn node class.  This is our core tensor network object.
    py::class_<blas_backend>(m, "backend")
        .def_static("initialise", [](size_type nthreads, bool batch_par){blas_backend::initialise(nthreads, batch_par);}, py::arg("nthreads")=1, py::arg("batch_par")=false, R"mydelim(
            Initialise blas backend passing user defined arguments

            :param nthreads: The number of threads to use for linear algebra operations (Default: 1)
            :type nthreads: int, optional
            :param batch_par: Whether or not to parallelise batched gemm operationrs (Default: false)
            :type bath_par: bool, optional
            )mydelim")
        .def_static("destroy", &blas_backend::destroy, R"mydelim(
            Clear the blas_backend object.   Free any resources allocated.
            )mydelim")
        .def_static("label", &blas_backend::label, R"mydelim(
            :returns: A string representing a blas backend
            :rtype: str
            )mydelim");
}

#ifdef PYTTN_BUILD_CUDA
void initialise_cuda_backend(py::module& m)
{
    using namespace linalg;

    using size_type = cuda_backend::size_type;

    py::class_<cuda_environment>(m, "environment")
        .def(py::init(), "Default construct an empty cuda environment.")
        .def(py::init<int, int>(), R"mydelim(
            Construct a cuda environment specifying the device id and number of streams

            :Parameters:    - **device_id** (int) - The cuda device index
                            - **nstreams** (int) - The number of cuda streams to use
            )mydelim")        
        .def("init", &cuda_environment::init, py::arg(), py::arg("nstreams")=1, R"mydelim(
            Construct a cuda environment specifying the device id and number of streams

            :Parameters:    - **device_id** (int) - The cuda device index
                            - **nstreams** (int, optional) - The number of cuda streams to use (Default: 1)
            )mydelim")        
        .def("destroy", &cuda_environment::destroy, R"mydelim(
            Destroys the cuda environment object deallocating any internal memory.
            )mydelim")    
         .def("is_initialised", &cuda_environment::is_initialised, R"mydelim(
            :returns: Whether or not the cuda_environment object has been successfully initialised.
            "rtype: bool
            )mydelim")    
        .def_static("number_of_devices", &cuda_environment::number_of_devices, R"mydelim(
            :returns: The number of cuda devices available on the system
            :rtype: int
        )mydelim")
        .def("list_devices", 
            [](const cuda_environment& o)
            {
                std::ostringstream oss;
                o.list_devices(oss);
                return oss.str();
            }, R"mydelim(
            :returns: A string of the cuda_environmen properties
            :rtype: str
            )mydelim")
        .def("__str__", 
            [](const cuda_environment& o)
            {
                std::ostringstream oss;
                oss << o;
                return oss.str();
            }, R"mydelim(
            :returns: A string of the cuda_environmen properties
            :rtype: str
            )mydelim");

    //expose the ttn node class.  This is our core tensor network object.
    py::class_<cuda_backend>(m, "backend")
        .def_static("environment", &cuda_backend::environment, py::return_value_policy::reference, R"mydelim(
            Access the cuda environment parameters bound to the backend object.
            )mydelim")
        .def_static("initialise", [](size_type device_id, size_type nstreams){cuda_backend::initialise(device_id, nstreams);}, py::arg("device_id")=0, py::arg("nstreams")=1, R"mydelim(
            Initialise cuda backend passing a user defined environment object.

            :param device_id: The device id used for the cuda backend (Default: 0)
            :type device_id: int, optional
            :param nstreams: The maximum number of streams to use (Default: 1)
            :type nstreams: int, optional
            )mydelim")
        .def_static("destroy", &cuda_backend::destroy, R"mydelim(
            Clear the cuda_backend object.   Free any resources allocated.
            )mydelim")
        .def_static("device_properties", 
            []()
            {
                std::ostringstream oss;
                cuda_backend::device_properties(oss);
                return oss.str();
            }, R"mydelim(
            :returns: A string of the cuda device properties
            :rtype: str
            )mydelim")        
        .def_static("label", &cuda_backend::label, R"mydelim(
            :returns: A string representing a blas backend
            :rtype: str
            )mydelim");
}
#endif