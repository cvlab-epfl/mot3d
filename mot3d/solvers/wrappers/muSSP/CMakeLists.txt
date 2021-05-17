cmake_minimum_required(VERSION 2.9)
project("wrapper")

set(muSSP_DIR "../../../../muSSP-master/muSSP")
set(PY_LIB_NAME "wrapper")

set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -O2 -Wp,-w")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

# Force release build by default, speed is of the essence
if ( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release )
endif()

if (APPLE)
  set(CMAKE_MACOSX_RPATH ON )
endif()

if ( NOT PYTHON_MANUAL )
  find_package(PythonInterp 3)
  find_package(PythonLibs 3 REQUIRED)
  find_package(Boost COMPONENTS python${PYTHON_VERSION_MAJOR})
else()
  message("Manually add python, boost-python and boost-numpy..")

  # for python
  #set(PYTHON "/usr/local/Cellar/python@3.9/3.9.1_6/Frameworks/Python.framework/Versions/3.9")
  #set(PYTHON_LIBRARIES "${PYTHON}/lib/libpython3.9.dylib")
  #set(PYTHON_INCLUDE_DIRS "${PYTHON}/include/python3.9")

  # for libboost_python3 and libboost_numpy3
  #set(Boost_LIB_python "/usr/local/Cellar/boost-python3/1.75.0/lib/libboost_python39.dylib")
  ##set(Boost_LIB_numpy "/usr/local/Cellar/boost-python3/1.75.0/lib/libboost_numpy39.dylib")
  #set(Boost_LIBRARIES ${Boost_LIB_python})
  #set(Boost_INCLUDE_DIRS "/usr/local/Cellar/boost/1.75.0_1/include")

  message("Linked libraries: ${PYTHON_LIBRARIES} ${Boost_LIBRARIES}")
  message("Includes: ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS}")
endif()

if (${Boost_FOUND})
   message(STATUS "Found Boost.Python.")
else (${Boost_FOUND})
   message(FATAL_ERROR "Could not locate Boost.Python.")
endif (${Boost_FOUND})

include_directories(. ../ ${muSSP_DIR} ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

aux_source_directory(${muSSP_DIR} SOURCE_FILES)
add_library(${PY_LIB_NAME} SHARED wrapper.cpp ${SOURCE_FILES})
set_target_properties(${PY_LIB_NAME} PROPERTIES PREFIX "")
if(WIN32)
  # Ensure the output binary is called core.pyd on Windows (not core.dll)
  set_target_properties(${PY_LIB_NAME} PROPERTIES SUFFIX ".pyd")
endif(WIN32)
if(APPLE)
    set_target_properties(${PY_LIB_NAME} PROPERTIES SUFFIX ".so")
endif(APPLE)
message("Linked libraries: ${PYTHON_LIBRARIES} ${Boost_LIBRARIES}")
target_link_libraries(${PY_LIB_NAME} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})