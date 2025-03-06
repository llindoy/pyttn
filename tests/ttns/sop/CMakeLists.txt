file(GLOB SRC *.cpp)
target_sources(tests PRIVATE ${SRC})
add_subdirectory(operator_dictionaries)
