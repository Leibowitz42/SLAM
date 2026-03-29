cmake_minimum_required(VERSION 4.0)
project(ORB_SLAM3 LANGUAGES CXX C CUDA)

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

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # Jetson Orin Nano (ARM64)
    set(ONNXRUNTIME_INCLUDE_DIR /home/mickey/libs/onnxruntime/onnxruntime-linux-aarch64-1.20.0/include)
    set(ONNXRUNTIME_LIB_DIR /home/mickey/libs/onnxruntime/onnxruntime-linux-aarch64-1.20.0/lib)
else()
    # Desktop PC (x86_64)
    set(ONNXRUNTIME_INCLUDE_DIR /home/waitangwen/libs/onnxruntime/include)
    set(ONNXRUNTIME_LIB_DIR /home/waitangwen/libs/onnxruntime/lib)
endif()

include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})

# Find CUDA
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Set CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_STANDARD 14)

# Find OpenCV
find_package(OpenCV 4.2)
if(NOT OpenCV_FOUND)
   message(FATAL_ERROR "OpenCV > 4.2 not found.")
endif()

# Optional TensorRT Support
find_path(TENSORRT_INCLUDE_DIR NvInferRuntime.h PATHS /usr/include/aarch64-linux-gnu /usr/include /usr/local/include)
find_library(TENSORRT_LIBRARY_INFER nvinfer PATHS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY_INFER)
    message(STATUS "TensorRT found (${TENSORRT_INCLUDE_DIR}), enabling native TRT backend.")
    add_definitions(-DUSE_TENSORRT)
    include_directories(${TENSORRT_INCLUDE_DIR})
    set(TRT_SOURCES src/yolov8_seg_trt.cpp src/gpu_preprocess.cu)
    set(TRT_LIBS ${TENSORRT_LIBRARY_INFER})
else()
    message(STATUS "TensorRT not found, using ONNX Runtime only.")
    set(TRT_SOURCES "")
    set(TRT_LIBS "")
endif()

# Print OpenCV version
MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

# Find CUDA
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit Found")
endif()

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
        ${CUDAToolkit_INCLUDE_DIRS}
        /usr/local/cuda/include
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
        ${TRT_SOURCES}
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
        include/yolov8_utils.h
        )

# Add the g2o library
add_subdirectory(Thirdparty/g2o)

# Link libraries
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
	onnxruntime
        ${TRT_LIBS}
        CUDA::cudart
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

# trt disabled on PC

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

# If RealSense SDK is found, add the rgbd_realsense_D435i example
if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i
            Examples/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
endif()

