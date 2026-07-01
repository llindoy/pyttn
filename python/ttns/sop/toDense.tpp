/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2025 NPL Management Limited
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

#ifndef PYTHON_BINDING_TODENSE_TPP
#define PYTHON_BINDING_TODENSE_TPP

#include "toDense.hpp"

namespace py = pybind11;

template <typename T, typename U>
void init_convert_to_dense(py::module &m)
{
    using namespace ttns;
    m.def("convert_to_dense", [](const sOP &op, const system_modes &sysinf)
          {
        linalg::matrix<U> mat;
        CALL_AND_HANDLE(convert_to_dense(op, sysinf, mat), "Failed to convert sOP to dense matrix.");
        return mat; }, R"mydelim(
        Create a dense matrix representation of an sOP.

        :param op: The string site operator object
        :type op: sOP

        :returns: The dense matrix representation of the operator
        :rtype: Matrix
      )mydelim");

    m.def("convert_to_dense",
          [](const sOP &op,
            const system_modes &sysinf,
            const operator_dictionary<U, linalg::blas_backend> &opdict)
          {
              linalg::matrix<U> mat;
              CALL_AND_HANDLE(
                  convert_to_dense(op, sysinf, opdict, mat),
                  "Failed to convert sOP to dense matrix with operator dictionary.");
              return mat;
          },
          R"mydelim(
    Create a dense matrix representation of an sOP using a custom operator dictionary.

    :param op: The string site operator object
    :type op: sOP
    :param sysinf: System mode information
    :type sysinf: system_modes
    :param opdict: Operator dictionary defining site operators
    :type opdict: operator_dictionary

    :returns: The dense matrix representation of the operator
    :rtype: Matrix
    )mydelim");


    m.def("convert_to_dense", [](const sPOP &op, const system_modes &sysinf)
          {
        linalg::matrix<U> mat;
        CALL_AND_HANDLE(convert_to_dense(op, sysinf, mat), "Failed to convert sPOP to dense matrix.");
        return mat; }, R"mydelim(
        Create a dense matrix representation of an sPOP.

        :param op: The string product operator object
        :type op: sPOP

        :returns: The dense matrix representation of the operator
        :rtype: Matrix
      )mydelim");

    m.def("convert_to_dense",
          [](const sPOP &op,
            const system_modes &sysinf,
            const operator_dictionary<U, linalg::blas_backend> &opdict)
          {
              linalg::matrix<U> mat;
              CALL_AND_HANDLE(
                  convert_to_dense(op, sysinf, opdict, mat),
                  "Failed to convert sPOP to dense matrix with operator dictionary.");
              return mat;
          },
          R"mydelim(
    Create a dense matrix representation of an sPOP using a custom operator dictionary.

    :param op: The string product operator object
    :type op: sPOP
    :param sysinf: System mode information
    :type sysinf: system_modes
    :param opdict: Operator dictionary defining site operators
    :type opdict: operator_dictionary

    :returns: The dense matrix representation of the operator
    :rtype: Matrix
    )mydelim");

    m.def("convert_to_dense", [](const sNBO<T> &op, const system_modes &sysinf)
          {
        linalg::matrix<U> mat;
        CALL_AND_HANDLE(convert_to_dense(op, sysinf, mat), "Failed to convert sNBO to dense matrix.");
        return mat; }, R"mydelim(
        Create a dense matrix representation of an sNBO.

        :param op: The string operator object
        :type op: sNBO

        :returns: The dense matrix representation of the operator
        :rtype: Matrix
      )mydelim");

    m.def("convert_to_dense",
          [](const sNBO<T> &op,
            const system_modes &sysinf,
            const operator_dictionary<U, linalg::blas_backend> &opdict)
          {
              linalg::matrix<U> mat;
              CALL_AND_HANDLE(
                  convert_to_dense(op, sysinf, opdict, mat),
                  "Failed to convert sNBO to dense matrix with operator dictionary.");
              return mat;
          },
          R"mydelim(
    Create a dense matrix representation of an sNBO using a custom operator dictionary.

    :param op: The string n-body operator object
    :type op: sNBO
    :param sysinf: System mode information
    :type sysinf: system_modes
    :param opdict: Operator dictionary defining site operators
    :type opdict: operator_dictionary

    :returns: The dense matrix representation of the operator
    :rtype: Matrix
    )mydelim");

    m.def("convert_to_dense", [](const sSOP<T> &op, const system_modes &sysinf)
          {
        linalg::matrix<U> mat;
        CALL_AND_HANDLE(convert_to_dense(op, sysinf, mat), "Failed to convert sSOP to dense matrix.");
        return mat; }, R"mydelim(
        Create a dense matrix representation of an sSOP.

        :param op: The string operator object
        :type op: sSOP

        :returns: The dense matrix representation of the operator
        :rtype: Matrix
      )mydelim");

    m.def("convert_to_dense",
          [](const sSOP<T> &op,
            const system_modes &sysinf,
            const operator_dictionary<U, linalg::blas_backend> &opdict)
          {
              linalg::matrix<U> mat;
              CALL_AND_HANDLE(
                  convert_to_dense(op, sysinf, opdict, mat),
                  "Failed to convert sSOP to dense matrix with operator dictionary.");
              return mat;
          },
          R"mydelim(
    Create a dense matrix representation of an sSOP using a custom operator dictionary.

    :param op: The string sum-of-product operator object
    :type op: sSOP
    :param sysinf: System mode information
    :type sysinf: system_modes
    :param opdict: Operator dictionary defining site operators
    :type opdict: operator_dictionary

    :returns: The dense matrix representation of the operator
    :rtype: Matrix
    )mydelim");


    m.def("convert_to_dense", [](const SOP<T> &op, const system_modes &sysinf)
          {
        linalg::matrix<U> mat;
        CALL_AND_HANDLE(convert_to_dense(op, sysinf, mat), "Failed to convert SOP to dense matrix.");
        return mat; }, R"mydelim(
        Create a dense matrix representation of an SOP.

        :param op: The string operator object
        :type op: SOP

        :returns: The dense matrix representation of the operator
        :rtype: Matrix
      )mydelim");

    m.def("convert_to_dense",
          [](const SOP<T> &op,
            const system_modes &sysinf,
            const operator_dictionary<U, linalg::blas_backend> &opdict)
          {
              linalg::matrix<U> mat;
              CALL_AND_HANDLE(
                  convert_to_dense(op, sysinf, opdict, mat),
                  "Failed to convert SOP to dense matrix with operator dictionary.");
              return mat;
          },
          R"mydelim(
    Create a dense matrix representation of an SOP using a custom operator dictionary.

    :param op: The compact SOP operator object
    :type op: SOP
    :param sysinf: System mode information
    :type sysinf: system_modes
    :param opdict: Operator dictionary defining site operators
    :type opdict: operator_dictionary

    :returns: The dense matrix representation of the operator
    :rtype: Matrix
    )mydelim");
}

#endif
