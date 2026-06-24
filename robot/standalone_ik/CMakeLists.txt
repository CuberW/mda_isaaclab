cmake_minimum_required(VERSION 3.16)
project(standalone_ik LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT DEFINED KUAVO_ROOT)
  set(KUAVO_ROOT "${PROJECT_ROOT}/third_party/kuavo-ros-opensource")
endif()
set(HUMANOID_WHEEL_ROOT "${KUAVO_ROOT}/src/humanoid-wheel-control/humanoid_wheel_interface")
set(OCS2_ROOT "${KUAVO_ROOT}/src/ocs2")
set(OCS2_CORE_ROOT "${OCS2_ROOT}/ocs2_core")
set(OCS2_PINOCCHIO_ROOT "${OCS2_ROOT}/ocs2_pinocchio/ocs2_pinocchio_interface")
set(OCS2_ROBOTIC_TOOLS_ROOT "${OCS2_ROOT}/ocs2_robotic_tools")
set(OCS2_THIRDPARTY_ROOT "${OCS2_ROOT}/ocs2_thirdparty")

if(NOT EXISTS "${HUMANOID_WHEEL_ROOT}/src/motion_planner/InverseKinematics.cpp")
  message(FATAL_ERROR "Invalid KUAVO_ROOT: ${KUAVO_ROOT}")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(PINOCCHIO REQUIRED pinocchio)
pkg_check_modules(CONSOLE_BRIDGE REQUIRED console_bridge)
pkg_check_modules(URDFDOM_HEADERS REQUIRED urdfdom_headers)

find_library(URDFDOM_MODEL_LIB NAMES urdfdom_model REQUIRED)
find_library(URDFDOM_WORLD_LIB NAMES urdfdom_world REQUIRED)
find_library(URDFDOM_SENSOR_LIB NAMES urdfdom_sensor REQUIRED)
find_library(URDFDOM_MODEL_STATE_LIB NAMES urdfdom_model_state REQUIRED)
find_library(TINYXML2_LIB NAMES tinyxml2 tinyxml REQUIRED)

add_library(standalone_ik SHARED
  "${CMAKE_CURRENT_LIST_DIR}/ik_wrapper.cpp"
  "${HUMANOID_WHEEL_ROOT}/src/FactoryFunctions.cpp"
  "${HUMANOID_WHEEL_ROOT}/src/motion_planner/InverseKinematics.cpp"
  "${OCS2_PINOCCHIO_ROOT}/src/PinocchioInterface.cpp"
  "${OCS2_PINOCCHIO_ROOT}/src/PinocchioInterfaceCppAd.cpp"
  "${OCS2_PINOCCHIO_ROOT}/src/urdf.cpp"
)

target_include_directories(standalone_ik
  PRIVATE
    "${HUMANOID_WHEEL_ROOT}/include"
    "${OCS2_CORE_ROOT}/include"
    "${OCS2_PINOCCHIO_ROOT}/include"
    "${OCS2_ROBOTIC_TOOLS_ROOT}/include"
    "${OCS2_THIRDPARTY_ROOT}/include"
    ${EIGEN3_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${PINOCCHIO_INCLUDE_DIRS}
    ${CONSOLE_BRIDGE_INCLUDE_DIRS}
    ${URDFDOM_HEADERS_INCLUDE_DIRS}
)

target_compile_definitions(standalone_ik
  PRIVATE
    PINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR
    PINOCCHIO_URDFDOM_USE_STD_SHARED_PTR
)

target_compile_options(standalone_ik
  PRIVATE
    ${PINOCCHIO_CFLAGS_OTHER}
    -Wno-ignored-attributes
    -Wno-invalid-partial-specialization
)

target_link_directories(standalone_ik
  PRIVATE
    ${PINOCCHIO_LIBRARY_DIRS}
    ${CONSOLE_BRIDGE_LIBRARY_DIRS}
)

target_link_libraries(standalone_ik
  PRIVATE
    Eigen3::Eigen
    ${PINOCCHIO_LIBRARIES}
    ${CONSOLE_BRIDGE_LIBRARIES}
    ${URDFDOM_MODEL_LIB}
    ${URDFDOM_WORLD_LIB}
    ${URDFDOM_SENSOR_LIB}
    ${URDFDOM_MODEL_STATE_LIB}
    ${TINYXML2_LIB}
)
