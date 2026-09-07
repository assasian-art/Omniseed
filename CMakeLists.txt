# =============================================================================
#  OmniSeed — Sub-400MB Multi-Modal Micro-LLM Agent Kernel
#  Pure C++17. Zero external dependencies. Build once, run everywhere.
#
#  Supported toolchains:
#    * MSVC        (Visual Studio 2019/2022/18, cl.exe)  — Windows
#    * MinGW-w64   (g++ via mingw32-make)                — Windows
#    * GCC / Clang (g++ / clang++)                       — Linux, macOS, Docker
#
#  Design constraints encoded here:
#    * C++17, no RTTI dependency in hot paths, exceptions allowed but unused
#    * No Python, no PyTorch, no external libs — everything vendored in-tree
#    * Static link where possible to produce a single portable binary
# =============================================================================
cmake_minimum_required(VERSION 3.16)

project(OmniSeed
    VERSION 0.1.0
    DESCRIPTION "OmniSeed: sub-400MB multi-modal micro-LLM agent kernel (RWKV-7 + BitNet b1.58)"
    LANGUAGES CXX)

# ----------------------------------------------------------------------------
# Global compile settings
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)          # use -std=c++17, not gnu++17

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# --------------------------------- Options -----------------------------------
option(OMNISEED_BUILD_TESTS  "Build the unit test suite"                ON)
option(OMNISEED_BUILD_CLI    "Build the omniseed CLI"                   ON)
option(OMNISEED_BUILD_SERVER "Build the optional HTTP server"           OFF)
option(OMNISEED_BUILD_SHARED "Build omniseed_core as a shared library"  OFF)
option(OMNISEED_WERROR       "Treat warnings as errors"                 OFF)

# HTTP server is opt-in; the vendored single-header lives in third_party/.
if(OMNISEED_BUILD_SERVER)
    add_compile_definitions(OMNISEED_HTTP)
endif()

# ----------------------------- Core static library ---------------------------
set(OMNISEED_CORE_SOURCES
    src/core/platform.cpp
    src/core/tensor.cpp
    src/core/bitlinear.cpp
    src/core/rwkv.cpp
    src/core/gguf_loader.cpp
    src/core/tokenizer.cpp
    src/core/bitlinear_avx2.cpp
    src/vision/mobilenet_v4.cpp
    src/vision/unicompress.cpp
    src/vision/vision_tasks.cpp
    src/audio/whisper_tiny.cpp
    src/audio/focal_codec.cpp
    src/audio/audio_events.cpp
    src/runtime/token_bus.cpp
    src/runtime/sensory.cpp
    src/runtime/emotional.cpp
    src/runtime/swarm.cpp
    src/memory/streaming_llm.cpp
    src/memory/memory_crystals.cpp
    src/agent/grammar_decoder.cpp
    src/agent/tool_registry.cpp
    src/agent/agent_loop.cpp
    src/agent/self_improvement.cpp
    src/agent/compute_throttle.cpp
    src/agent/flash_skills.cpp
    src/agent/agent_intel.cpp
)

if(OMNISEED_BUILD_SHARED)
    add_library(omniseed_core SHARED ${OMNISEED_CORE_SOURCES})
else()
    add_library(omniseed_core STATIC ${OMNISEED_CORE_SOURCES})
endif()

target_include_directories(omniseed_core
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

target_compile_features(omniseed_core PUBLIC cxx_std_17)

# x86/x64: compile the SIMD TU with AVX2 and export the arch macro the
# runtime dispatcher tests. Other targets simply keep the scalar kernels.
if((CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(e2k)") AND
   (CMAKE_CXX_COMPILER_ID MATCHES "MSVC|GNU|Clang|AppleClang"))
    target_compile_definitions(omniseed_core PRIVATE OMNISEED_X86=1)
    if(MSVC)
        set_source_files_properties(src/core/bitlinear_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(src/core/bitlinear_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
endif()

# --------------------------- Per-compiler warnings ---------------------------
if(MSVC)
    target_compile_options(omniseed_core PRIVATE
        /W4 /permissive- /Zc:__cplusplus /utf-8
        /wd4100  # unreferenced formal parameter (stub params are intentional)
        /wd4127  # conditional expression is constant
        /wd4244  # conversion warnings from float<->double narrowing in kernels
        /wd4267)
    if(OMNISEED_WERROR)
        target_compile_options(omniseed_core PRIVATE /WX)
    endif()
    # Multi-processor build for MSVC generators.
    target_compile_options(omniseed_core PRIVATE /MP)
    # Portable C stdio (fopen/sscanf/localtime) is intentional; silence the
    # MSVC CRT "safe functions" deprecation notices instead of scattering
    # #define pragmas through the sources.
    target_compile_definitions(omniseed_core PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(omniseed_core PRIVATE
        -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable)
    if(OMNISEED_WERROR)
        target_compile_options(omniseed_core PRIVATE -Werror)
    endif()
    # Core math kernels love inlining; keep it aggressive in Release.
    target_compile_options(omniseed_core PRIVATE $<$<CONFIG:Release>:-O3>)
endif()

# -----------------------------------------------------------------------------
# Keep all binaries in build/bin for every generator (VS multi-config included).
# -----------------------------------------------------------------------------
set(OMNISEED_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OMNISEED_OUTPUT_DIR})
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg} ${OMNISEED_OUTPUT_DIR})
endforeach()

# Threads (std::thread used by the runtime throttle; harmless on all targets).
find_package(Threads REQUIRED)
target_link_libraries(omniseed_core PUBLIC Threads::Threads)

if(WIN32)
    # platform.cpp uses GetProcessMemoryInfo (peak RSS).
    target_link_libraries(omniseed_core PUBLIC psapi)
    # swarm.cpp (UdpBeacon) uses WinSock2 datagram sockets.
    target_link_libraries(omniseed_core PUBLIC ws2_32)
endif()

# ------------------------------- Test executable -----------------------------
if(OMNISEED_BUILD_TESTS)
    enable_testing()
    add_executable(omniseed_tests tests/test_platform.cpp)
    target_include_directories(omniseed_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_tests PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_tests PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_tests PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_platform COMMAND omniseed_tests
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    # Real-weights validation: skips cleanly when the converted GGUF is not
    # present (fresh clones / CI), asserts coherence when it is.
    add_executable(omniseed_real_weights tests/test_real_weights.cpp)
    target_include_directories(omniseed_real_weights PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_real_weights PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_real_weights PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_real_weights PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_real_weights COMMAND omniseed_real_weights
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_executable(omniseed_sides tests/test_sides.cpp)
    target_include_directories(omniseed_sides PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_sides PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_sides PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_sides PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_sides COMMAND omniseed_sides
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# --------------------------------- CLI binary --------------------------------
if(OMNISEED_BUILD_CLI)
    add_executable(omniseed src/cli/main.cpp)
    target_include_directories(omniseed PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed PRIVATE -Wall -Wextra)
    endif()
endif()

# ------------------------- Optional HTTP server binary -----------------------
# Built only with -DOMNISEED_BUILD_SERVER=ON; guarded in source by OMNISEED_HTTP.
if(OMNISEED_BUILD_SERVER)
    add_executable(omniseed_server src/server/main.cpp)
    target_include_directories(omniseed_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_compile_definitions(omniseed_server PRIVATE OMNISEED_HTTP)
    # Same portable-stdio stance as omniseed_core (getenv/localhost helpers).
    target_compile_definitions(omniseed_server PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_link_libraries(omniseed_server PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_server PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_server PRIVATE -Wall -Wextra)
    endif()
    if(WIN32)
        target_link_libraries(omniseed_server PRIVATE ws2_32)
    endif()
endif()

# ------------------------------ Install rules --------------------------------
install(TARGETS omniseed_core
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY include/omniseed DESTINATION include)
install(FILES docs/DEPLOYMENT.md DESTINATION share/omniseed OPTIONAL)
