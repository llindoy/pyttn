#ifndef COMMON_TIMING_MACRO_HPP
#define COMMON_TIMING_MACRO_HPP

#include <chrono>


#ifdef TIMING
#define INIT_TIMER auto start = std::chrono::high_resolution_clock::now();
#define START_TIMER  start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name)  std::cerr << "RUNTIME of " << name << ": " << \
    std::chrono::duration_cast<std::chrono::microseconds>( \
            std::chrono::high_resolution_clock::now()-start \
    ).count() << " us " << std::endl; 
#else
#define INIT_TIMER
#define START_TIMER
#define STOP_TIMER(name)
#endif

#ifdef TIMING_T
#define INIT_TIMER_T auto start_T = std::chrono::high_resolution_clock::now();
#define START_TIMER_T  start_T = std::chrono::high_resolution_clock::now();
#define STOP_TIMER_T(name)  std::cerr << "RUNTIME of " << name << ": " << \
    std::chrono::duration_cast<std::chrono::microseconds>( \
            std::chrono::high_resolution_clock::now()-start_T \
    ).count() << " us " << std::endl; 
#else
#define INIT_TIMER_T
#define START_TIMER_T
#define STOP_TIMER_T(name)
#endif

#endif  //COMMON_TIMING_MACRO_HPP_//

