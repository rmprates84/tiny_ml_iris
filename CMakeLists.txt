# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico_w CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(tiny_ml_02 C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# ----------------------------------------------------------------------
# Usa o port oficial do TensorFlow Lite Micro para o Pico:
#    lib/pico-tflmicro
# ----------------------------------------------------------------------
add_subdirectory(lib/pico-tflmicro EXCLUDE_FROM_ALL)

# ❌ NÃO usar mais o repositório tflite-micro puro (não tem CMakeLists)
# add_subdirectory(tflite-micro)

# Add executable. Default name is the project name, version 0.1
# (ajuste o nome dos fontes se estiverem em outra pasta, ex: src/tiny_ml_02.c)
add_executable(tiny_ml_02
    tiny_ml_02.c
    tflm_wrapper.cpp
    tflm_wrapper.h
    iris_mlp_float.h
)

pico_set_program_name(tiny_ml_02 "tiny_ml_02")
pico_set_program_version(tiny_ml_02 "0.1")

# Modify the below lines to enable/disable output over UART/USB
pico_enable_stdio_uart(tiny_ml_02 0)
pico_enable_stdio_usb(tiny_ml_02 1)

# ----------------------------------------------------------------------
# Bibliotecas de link:
#   - pico_stdlib   : stdio, etc.
#   - pico-tflmicro : TensorFlow Lite Micro portado p/ Pico
#   - hardware_sync : usado nos exemplos com pico-tflmicro
# ----------------------------------------------------------------------
target_link_libraries(tiny_ml_02
    pico_stdlib
    hardware_sync
    pico-tflmicro
)

# Add the standard include files to the build
target_include_directories(tiny_ml_02 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

pico_add_extra_outputs(tiny_ml_02)
