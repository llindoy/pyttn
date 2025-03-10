file(GLOB SRC *.cpp)
target_sources(tests PRIVATE ${SRC})
add_subdirectory(operators)
add_subdirectory(tree)
add_subdirectory(sop)

