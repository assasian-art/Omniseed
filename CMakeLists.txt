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
option(OMNISEED_BUILD_AGENT2 "Build omniseed_agent2 (trading + introspection CLI)" ON)
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
    src/core/lora.cpp
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
    src/memory/prefix_cache.cpp
    src/core/uncertainty.cpp
    src/agent/grammar_decoder.cpp
    src/agent/tool_registry.cpp
    src/agent/agent_loop.cpp
    src/agent/self_improvement.cpp
    src/agent/compute_throttle.cpp
    src/agent/flash_skills.cpp
    src/agent/agent_intel.cpp
    src/trading/trading_engine.cpp
    src/trading/news_feed.cpp
    src/trading/simulate.cpp
    src/trading/finance.cpp
    src/trading/reasoning_tools.cpp
    src/trading/broker_alpaca.cpp
    src/runtime/introspection.cpp
    src/runtime/cloud_bridge.cpp
    src/agent/sub_agents.cpp
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
    target_compile_definitions(omniseed_real_weights PRIVATE _CRT_SECURE_NO_WARNINGS)
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
    target_compile_definitions(omniseed_sides PRIVATE _CRT_SECURE_NO_WARNINGS)

    # QAT-model ternary-kernel A/B: byte-identical greedy continuation across
    # scalar/SSE4.1/AVX2. Skips cleanly when the QAT GGUF is not present.
    add_executable(omniseed_qat_ternary tests/test_qat_ab.cpp)
    target_include_directories(omniseed_qat_ternary PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_qat_ternary PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_qat_ternary PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_qat_ternary PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_qat_ternary COMMAND omniseed_qat_ternary
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_qat_ternary PRIVATE _CRT_SECURE_NO_WARNINGS)

    # LoRA sidecar runtime test: synthetic omniseed-lora GGUF (written by the
    # test itself, no Python) validates loader/math/rejection, then a
    # real-model attach/detach A/B — skips cleanly when the base GGUF or the
    # trained smoke sidecar is absent.
    add_executable(omniseed_lora tests/test_lora.cpp)
    target_include_directories(omniseed_lora PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_lora PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_lora PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_lora PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_lora COMMAND omniseed_lora
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_lora PRIVATE _CRT_SECURE_NO_WARNINGS)

    # Assistant-LoRA E2E regression (degenerate-run guard): the driver
    # trains 12 steps on tests/fixtures/lora_chat_tiny.tsv with the venv
    # python + tools/lora_chat.py, then drives the omniseed_lora_e2e
    # harness (round-trip, C++/python logits parity, coherence). Skips
    # cleanly when the venv/base model is absent. The harness is built
    # here but invoked BY the driver, not by ctest directly.
    add_executable(omniseed_lora_e2e tests/test_lora_e2e.cpp)
    target_include_directories(omniseed_lora_e2e PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_lora_e2e PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_lora_e2e PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_lora_e2e PRIVATE -Wall -Wextra)
    endif()
    target_compile_definitions(omniseed_lora_e2e PRIVATE _CRT_SECURE_NO_WARNINGS)
    set(OMNISEED_VENV_PY
        "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
    if(NOT EXISTS "${OMNISEED_VENV_PY}")
        set(OMNISEED_VENV_PY
            "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    endif()
    if(EXISTS "${OMNISEED_VENV_PY}")
        add_test(NAME omniseed_lora_e2e
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/lora_e2e/e2e_lora_train.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_lora_e2e PROPERTIES TIMEOUT 2400)

        # gguf-base mode: train ON the served GGUF (deployment weights);
        # asserts attach-improves-ppl python-side + cross-engine parity.
        # Skips internally when the QAT GGUF is absent.
        add_test(NAME omniseed_lora_gguf
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/lora_e2e/e2e_lora_gguf.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_lora_gguf PROPERTIES TIMEOUT 2400)
    endif()

    # Trading framework: deterministic signal/risk/backtest math on synthetic
    # bars (no network, no fixtures needed).
    add_executable(omniseed_trading tests/test_trading.cpp)
    target_include_directories(omniseed_trading PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_trading PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_trading PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_trading PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_trading COMMAND omniseed_trading
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

# -------------------------- Introspection + trading CLI ----------------------
# Separate binary so the core CLI stays lean; shares omniseed_core.
if(OMNISEED_BUILD_AGENT2)
    add_executable(omniseed_agent2 src/agent2/main.cpp)
    target_include_directories(omniseed_agent2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(omniseed_agent2 PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_link_libraries(omniseed_agent2 PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_agent2 PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_agent2 PRIVATE -Wall -Wextra)
    endif()
endif()

# ------------------------------ Install rules --------------------------------
install(TARGETS omniseed_core
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY include/omniseed DESTINATION include)
install(FILES docs/DEPLOYMENT.md DESTINATION share/omniseed OPTIONAL)
