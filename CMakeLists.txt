cmake_minimum_required(VERSION 3.20)
project(mlx_node LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MLX comes prebuilt from the pip wheel (mlx-metal on macOS, mlx[cpu] on Linux).
# scripts/build.js passes MLX_DIR=<site-packages>/mlx/share/cmake/MLX.
find_package(MLX CONFIG REQUIRED)

add_definitions(-DNAPI_VERSION=8 -DNODE_ADDON_API_DISABLE_DEPRECATED -DNAPI_DISABLE_CPP_EXCEPTIONS)
execute_process(
  COMMAND node -p "require('node-addon-api').include"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE NODE_ADDON_API_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

add_library(${PROJECT_NAME} SHARED src/addon.cpp src/qwen35.cpp src/lora.cpp ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_JS_INC} ${NODE_ADDON_API_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE mlx ${CMAKE_JS_LIB})

# libmlx (and mlx.metallib on macOS) are copied next to the .node file, so the
# addon finds them without Python or a fixed path.
if(APPLE)
  set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_RPATH "@loader_path")
else()
  set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_RPATH "$ORIGIN")
endif()
file(GLOB MLX_RUNTIME_FILES "${MLX_LIBRARY_DIRS}/libmlx*" "${MLX_LIBRARY_DIRS}/*.metallib")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MLX_RUNTIME_FILES} $<TARGET_FILE_DIR:${PROJECT_NAME}>)

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()
