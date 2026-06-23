/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2025 NPL Management Limited
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICEN_ZN6linalglsERSoRKNS_16cuda_environmentESE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

#include <linalg/linalg.cuh>
#include "../../linalg/backend.hpp"
#include <sstream>


void initialise_cuda_backend(py::module &m)
{
    using namespace linalg;

    using size_type = typename linalg::traits<cuda_backend>::size_type;

    py::class_<cuda_environment>(m, "environment")
        .def(py::init(), "Default construct an empty cuda environment.")
        .def(py::init<int, int>(), R"mydelim(
            Construct a cuda environment specifying the device id and number of streams

            :param device_id: The cuda device index
            :type device_id: int
            :param nstreams: The number of cuda streams to use
            :type nstreams: int
            )mydelim")
        .def("init", &cuda_environment::init, py::arg(), py::arg("nstreams") = 1, R"mydelim(
            Construct a cuda environment specifying the device id and number of streams

            :param device_id: The cuda device index
            :type device_id: int
            :param nstreams: The number of cuda streams to use
            :type nstreams: int, Optional
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
        .def("list_devices", []()
             {
                std::ostringstream oss;
                cuda_environment::list_devices(oss);
                return oss.str(); }, R"mydelim(
            :returns: A string of the cuda_environmen properties
            :rtype: str
            )mydelim")
        .def("__str__", [](const cuda_environment &o)
             {
                std::ostringstream oss;
                oss << o;
                return oss.str(); }, R"mydelim(
            :returns: A string of the cuda_environmen properties
            :rtype: str
            )mydelim");

    // expose the ttn node class.  This is our core tensor network object.
    py::class_<cuda_backend>(m, "backend")
        .def_static("environment", &cuda_backend::environment, py::return_value_policy::reference, R"mydelim(
            Access the cuda environment parameters bound to the backend object.
            )mydelim")
        .def_static("initialise", [](size_type device_id, size_type nstreams)
                    { cuda_backend::initialise(device_id, nstreams); }, py::arg("device_id") = 0, py::arg("nstreams") = 1, R"mydelim(
            Initialise cuda backend passing a user defined environment object.

            :param device_id: The device id used for the cuda backend (Default: 0)
            :type device_id: int, optional
            :param nstreams: The maximum number of streams to use (Default: 1)
            :type nstreams: int, optional
            )mydelim")
        .def_static("destroy", &cuda_backend::destroy, R"mydelim(
            Clear the cuda_backend object.   Free any resources allocated.
            )mydelim")
        .def_static("device_properties", []()
                    {
                std::ostringstream oss;
                cuda_backend::device_properties(oss);
                return oss.str(); }, R"mydelim(
            :returns: A string of the cuda device properties
            :rtype: str
            )mydelim");
}

