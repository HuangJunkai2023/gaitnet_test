cmake_minimum_required(VERSION 3.5.1)
project(mysim)
# cmake_policy(SET CMP0148 OLD)
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

set(OpenGL_GL_PREFERENCE LEGACY)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

find_package(PythonLibs 3.6 EXACT REQUIRED)
endif(NOT CMAKE_BUILD_TYPE)

link_libraries(stdc++fs)

add_compile_options(-fPIC)
# add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)

add_compile_options(-DHAVE_CSTDDEF)

add_definitions(-DMASS_ROOT_DIR="..")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake")

# Set Python paths to conda environment
# set(PYTHON_EXECUTABLE "$ENV{HOME}/miniconda3/envs/gaitnet/bin/python")
# set(PYTHON_INCLUDE_DIR "$ENV{HOME}/miniconda3/envs/gaitnet/include/python3.6m")
# set(PYTHON_LIBRARY "$ENV{HOME}/miniconda3/envs/gaitnet/lib/libpython3.6m.so")

find_package(PythonLibs 3.6 EXACT REQUIRED)
find_package(PythonInterp 3.6 EXACT REQUIRED)
add_subdirectory( libs )
add_subdirectory( sim )
add_subdirectory( python )
if(NOT SERVER_BUILD)
add_subdirectory( viewer )
endif()
