cmake_minimum_required(VERSION 3.21)
set(APP_NAME DirectXMiniGame)
project(${APP_NAME} VERSION 1.0 LANGUAGES CXX)

add_subdirectory(${CMAKE_SOURCE_DIR}/DirectXTK12 ${CMAKE_BINARY_DIR}/bin/DirectXTK12)

set_target_properties(${TARGET_NAME} PROPERTIES C_STANDARD 17)
set_target_properties(${TARGET_NAME} PROPERTIES CMAKE_C_STANDARD_REQUIRED True)

set(CMAKE_CXX_COMPILER "msvc")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
set(CMAKE_SUPPRESS_REGENERATION true)

if( NOT CMAKE_BUILD_TYPE )
   set( CMAKE_BUILD_TYPE Release )
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
   message("${CMAKE_CXX_FLAGS_RELEASE}")
else()
   message("${CMAKE_CXX_FLAGS_DEBUG}")
endif()

set(IMGUI_HEADER "${CMAKE_SOURCE_DIR}/engine/3d_party/imgui")
set(IMGUI_SRC
"${CMAKE_CURRENT_SOURCE_DIR}/engine/3d_party/imgui/imgui.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/engine/3d_party/imgui/imgui_tables.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/engine/3d_party/imgui/imgui_draw.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/engine/3d_party/imgui/imgui_widgets.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/engine/3d_party/imgui/backends/imgui_impl_dx12.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/engine/3d_party/imgui/backends/imgui_impl_win32.cpp"
)
add_library(imgui STATIC ${IMGUI_SRC})

file(GLOB_RECURSE HEADERS "${CMAKE_SOURCE_DIR}/engine/include/*.h")
file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/engine/src/*.cpp")
set(THIRD_PARTY_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/engine/3d_party/")
file(GLOB_RECURSE include_list LIST_DIRECTORIES true "${CMAKE_SOURCE_DIR}/engine/include")
include_directories(${include_list} ${THIRD_PARTY_INCLUDE_DIR} "${CMAKE_SOURCE_DIR}/DirectXTK12/Inc")

add_executable(${APP_NAME} ${HEADERS} ${SOURCES})
target_link_libraries(${APP_NAME} "d3d12.lib" "d3dcompiler.lib" "dxgi.lib" "dxguid.lib" DirectXTK12 imgui)
