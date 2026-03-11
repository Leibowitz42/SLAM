find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif()

cmake_minimum_required(VERSION 3.10)
project(ORB_SLAM3)

# Set the build type if not specified
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

# Print the build type
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    add_definitions(-DCOMPILEDWITHC14)
    message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

# Disable warnings
add_definitions(-w)

# Set the module path
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# ONNX Runtime: detect include and lib (search also in $HOME/libs/onnxruntime)
set(ONNXRUNTIME_SEARCH_PATHS
    /usr/include /usr/local/include /opt/onnxruntime/include
    $ENV{HOME}/libs/onnxruntime/onnxruntime-linux-aarch64-1.20.0/include)
find_path(ONNXRUNTIME_INCLUDE_DIR
          NAMES onnxruntime_cxx_api.h
          PATHS ${ONNXRUNTIME_SEARCH_PATHS}
          NO_DEFAULT_PATH)

set(ONNXRUNTIME_LIB_SEARCH_PATHS
    /usr/lib /usr/local/lib /opt/onnxruntime/lib
    $ENV{HOME}/libs/onnxruntime/onnxruntime-linux-aarch64-1.20.0/lib)
find_library(ONNXRUNTIME_LIB
             NAMES onnxruntime libonnxruntime.so
             PATHS ${ONNXRUNTIME_LIB_SEARCH_PATHS}
             NO_DEFAULT_PATH)

if(ONNXRUNTIME_LIB)
    get_filename_component(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIB} DIRECTORY)
else()
    message(FATAL_ERROR "onnxruntime library not found. Searched paths: ${ONNXRUNTIME_LIB_SEARCH_PATHS}")
endif()

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "onnxruntime_cxx_api.h not found. Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime or install ONNX Runtime dev package and re-run CMake. Searched paths include: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

include_directories(${ONNXRUNTIME_INCLUDE_DIR})
if(ONNXRUNTIME_LIB_DIR)
    link_directories(${ONNXRUNTIME_LIB_DIR})
endif()

# Option to enable CUDA execution provider linkage
option(ONNXRUNTIME_USE_CUDA "Enable linking against ONNX Runtime CUDA provider" OFF)

# TensorRT (for YOLO .engine inference)
find_path(TENSORRT_INCLUDE_DIR NvInferRuntime.h
    PATHS /usr/include/aarch64-linux-gnu /usr/include /usr/local/include
    NO_DEFAULT_PATH)
find_library(TENSORRT_LIBRARY_INFER nvinfer
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib
    NO_DEFAULT_PATH)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY_INFER)
    message(FATAL_ERROR "TensorRT (NvInferRuntime.h, nvinfer) not found. Install libnvinfer-dev.")
endif()
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")

# CUDA (required by TensorRT runtime)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
    PATHS /usr/local/cuda/include /usr/local/cuda-12.6/include $ENV{CUDA_PATH}/include
    NO_DEFAULT_PATH)
find_library(CUDA_CUDART_LIBRARY cudart
    PATHS /usr/local/cuda/lib64 /usr/local/cuda-12.6/targets/aarch64-linux/lib /usr/lib/aarch64-linux-gnu
    NO_DEFAULT_PATH)
if(NOT CUDA_INCLUDE_DIR)
    message(FATAL_ERROR "CUDA include (cuda_runtime_api.h) not found.")
endif()
if(NOT CUDA_CUDART_LIBRARY)
    message(FATAL_ERROR "CUDA library (cudart) not found.")
endif()
get_filename_component(CUDA_LIB_DIR ${CUDA_CUDART_LIBRARY} DIRECTORY)

# Find OpenCV
find_package(OpenCV 4.2)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()

# Print OpenCV version
MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

# Find Eigen3, Pangolin, realsense2
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
# find_package(realsense2)

# Include directories
include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/CameraModels
        ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIR}
)

# Set the library output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Add the ORB_SLAM3 library
add_library(${PROJECT_NAME} SHARED
        src/System.cc
        src/Tracking.cc
        src/LocalMapping.cc
        src/LoopClosing.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/KeyFrame.cc
        src/Atlas.cc
        src/Map.cc
        src/MapDrawer.cc
        src/Optimizer.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Viewer.cc
        src/ImuTypes.cc
        src/G2oTypes.cc
        src/CameraModels/Pinhole.cpp
        src/CameraModels/KannalaBrandt8.cpp
        src/OptimizableTypes.cpp
        src/MLPnPsolver.cpp
        src/GeometricTools.cc
        src/TwoViewReconstruction.cc
        src/Config.cc
        src/Settings.cc
        src/YoloDetect.cpp
        src/yolov8_seg_onnx.cpp
        src/yolov8_seg_trt.cpp
        src/yolov8_utils.cpp
        include/System.h
        include/Tracking.h
        include/LocalMapping.h
        include/LoopClosing.h
        include/ORBextractor.h
        include/ORBmatcher.h
        include/FrameDrawer.h
        include/Converter.h
        include/MapPoint.h
        include/KeyFrame.h
        include/Atlas.h
        include/Map.h
        include/MapDrawer.h
        include/Optimizer.h
        include/Frame.h
        include/KeyFrameDatabase.h
        include/Sim3Solver.h
        include/Viewer.h
        include/ImuTypes.h
        include/G2oTypes.h
        include/CameraModels/GeometricCamera.h
        include/CameraModels/Pinhole.h
        include/CameraModels/KannalaBrandt8.h
        include/OptimizableTypes.h
        include/MLPnPsolver.h
        include/GeometricTools.h
        include/TwoViewReconstruction.h
        include/SerializationUtils.h
        include/Config.h
        include/Settings.h
        include/YoloDetect.h
        include/yolov8_seg_onnx.h
        include/yolov8_seg_trt.h
        include/yolov8_utils.h
        )

        if(NOT ONNXRUNTIME_USE_CUDA)
            target_compile_definitions(${PROJECT_NAME} PRIVATE DISABLE_ORT_CUDA)
        endif()

# Add the g2o library
add_subdirectory(Thirdparty/g2o)

# Link libraries
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        ${ONNXRUNTIME_LIB}
        ${TENSORRT_LIBRARY_INFER}
        ${CUDA_CUDART_LIBRARY}
        boost_serialization
        crypto
        )

# If RealSense SDK is found, add the library and compile examples
if(realsense2_FOUND)
    include_directories(${PROJECT_NAME}
            ${realsense_INCLUDE_DIR}
            )
    target_link_libraries(${PROJECT_NAME}
            ${realsense2_LIBRARY}
            )
endif()

# Build examples

# Set the runtime output directory for RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

# Add the rgbd_tum example
#TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE "${ONNXRUNTIME_DIR}/include")
#TARGET_LINK_LIBRARIES(${PROJECT_NAME} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

# If RealSense SDK is found, add the rgbd_realsense_D435i example
if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i
            Examples/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
endif()

