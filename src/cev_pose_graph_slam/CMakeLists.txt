cmake_minimum_required(VERSION 3.14)
project(cev_pose_graph_slam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Build only the ROS-free core + simulation test (no ROS install needed):
#   cmake -S src/cev_pose_graph_slam -B /tmp/pgo -DPGO_CORE_ONLY=ON && cmake --build /tmp/pgo
option(PGO_CORE_ONLY "Build only the ROS-free core library and the simulation test" OFF)

find_package(Eigen3 REQUIRED)

# ICP library vendored in the lidar odometry submodule. Its headers live in include/icp/ and the
# cevicp target does not export them, so we add the include path ourselves.
set(CEV_ICP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cev_lidar_odometry_ros2/icp)
if(NOT TARGET cevicp)
  set(LIB_ONLY ON CACHE BOOL "Build the icp library only" FORCE)
  add_subdirectory(${CEV_ICP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cevicp EXCLUDE_FROM_ALL)
endif()
set_target_properties(cevicp PROPERTIES POSITION_INDEPENDENT_CODE ON)
# cevicp only uses ${EIGEN3_INCLUDE_DIRS}, which newer Eigen packages (conda/pixi) leave empty.
target_link_libraries(cevicp PUBLIC Eigen3::Eigen)
# icp/lib/icp/driver.cpp uses assert() without <cassert> (only worked via older Eigen headers).
target_compile_options(cevicp PRIVATE -include cassert)

add_library(pgo_core STATIC
  src/sliding_window_pgo.cpp
  src/icp_scan_matcher.cpp)
target_include_directories(pgo_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CEV_ICP_DIR}/include)
target_link_libraries(pgo_core PUBLIC cevicp Eigen3::Eigen)
set_target_properties(pgo_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(pgo_sim_test test/sim_test.cpp)
target_link_libraries(pgo_sim_test pgo_core)
enable_testing()
add_test(NAME pgo_sim_test COMMAND pgo_sim_test)

if(PGO_CORE_ONLY)
  return()
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(pgo_node src/SLAM_backend_math.cpp)
target_link_libraries(pgo_node pgo_core)
ament_target_dependencies(pgo_node
  rclcpp nav_msgs geometry_msgs sensor_msgs tf2 tf2_ros)

install(TARGETS pgo_node pgo_sim_test
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

ament_package()
