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

#ifndef PYTTN_COMMON_TIMING_MACRO_HPP_
#define PYTTN_COMMON_TIMING_MACRO_HPP_

#include <chrono>

#ifdef TIMING
#define INIT_TIMER auto start = std::chrono::high_resolution_clock::now();
#define START_TIMER start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name) std::cerr << "RUNTIME of " << name << ": " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << " us " << std::endl;
#else
#define INIT_TIMER
#define START_TIMER
#define STOP_TIMER(name)
#endif

#ifdef TIMING_T
#define INIT_TIMER_T auto start_T = std::chrono::high_resolution_clock::now();
#define START_TIMER_T start_T = std::chrono::high_resolution_clock::now();
#define STOP_TIMER_T(name) std::cerr << "RUNTIME of " << name << ": " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_T).count() << " us " << std::endl;
#else
#define INIT_TIMER_T
#define START_TIMER_T
#define STOP_TIMER_T(name)
#endif

#endif // PYTTN_COMMON_TIMING_MACRO_HPP_//
