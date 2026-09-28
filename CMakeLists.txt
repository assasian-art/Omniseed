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
    # System-1 decision head ("two heads, one brain"): hidden state -> action
    # distribution in one matvec. Sits above core/ because it is a decision
    # layer that both the agent loop and the trading runtime consume.
    src/decision_head.cpp
    # "One backbone, multiple heads": the shared heads vocabulary, the router
    # that decides which heads run, and the per-head readouts of h[E].
    src/heads.cpp
    src/router.cpp
    src/classification_head.cpp
    src/scoring_head.cpp
    src/domain_decision.cpp
    src/unified_output.cpp
    # THE JOINT: joins the vision/audio encoders and the TokenBus to the head
    # stack. Everything it composes already existed; nothing turned an image or
    # a recording into the h[E] the heads read. This is that join.
    src/multimodal.cpp
    # The soul: persona + emotional resonance + honest self-report. Composes
    # runtime/emotional and runtime/introspection, which existed but had no
    # caller, into something the unified document can carry.
    src/soul.cpp
    # Language domain: deterministic, offline intent / language / sentiment.
    src/language/intent.cpp
    src/language/sentiment.cpp
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
    src/trading/walkforward.cpp
    src/trading/paper_daemon.cpp
    src/trading/market_perception.cpp
    src/trading/risk_gate.cpp
    src/trading/regime_engine.cpp
    src/trading/strategy_zoo.cpp
    src/trading/router.cpp
    src/trading/sniper.cpp
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

        # one-sidecar-many-skills coherence: combined assistant+trading
        # fixture, 50-step regime, --sample must answer BOTH domains
        # coherently (mechanism/parity lives in omniseed_lora_gguf).
        add_test(NAME omniseed_lora_chat
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/lora_e2e/e2e_lora_chat.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_lora_chat PROPERTIES TIMEOUT 2400)

        # int8-GGUF base: --gguf-base over models/rwkv7-0.1B-ternary.gguf
        # (dtype-4 int8 linears ride an identity passthrough in the trainer).
        # Skips internally when that GGUF is absent.
        add_test(NAME omniseed_lora_i8
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/lora_e2e/e2e_lora_i8.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_lora_i8 PROPERTIES TIMEOUT 2400)
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

    # Trading edge lab (T7.1): risk-budget sizing clamp, daily kill-switch,
    # walk-forward no-look-ahead + OOS determinism, paper journal append/resume.
    add_executable(omniseed_trading_edge tests/test_trading_edge.cpp)
    target_include_directories(omniseed_trading_edge PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_trading_edge PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_trading_edge PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_trading_edge PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_trading_edge COMMAND omniseed_trading_edge
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_trading_edge PRIVATE _CRT_SECURE_NO_WARNINGS)

    # Market perception + risk gates (M1/M4): feed-health staleness/error
    # thresholds, the ABSTAIN gate, composite daily/weekly drawdown governors,
    # asset-class-aware risk budgets (meme research-only), correlation cap, and
    # the code-enforced autonomy ladder (L0/L1/L2 unlock lines).
    add_executable(omniseed_market_perception tests/test_market_perception.cpp)
    target_include_directories(omniseed_market_perception PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_market_perception PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_market_perception PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_market_perception PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_market_perception COMMAND omniseed_market_perception
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_market_perception PRIVATE _CRT_SECURE_NO_WARNINGS)

    # Multi-asset paper session (M5): merged-timeline replay over several
    # streams sharing one cash account + one daily/weekly kill-switch, with
    # per-asset sizing, feed ABSTAIN, and journal-based resume.
    add_executable(omniseed_paper_session tests/test_paper_session.cpp)
    target_include_directories(omniseed_paper_session PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_paper_session PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_paper_session PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_paper_session PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_paper_session COMMAND omniseed_paper_session
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_paper_session PRIVATE _CRT_SECURE_NO_WARNINGS)

    # Market-data fetcher integrity: fully offline characterization of
    # tools/fetch_market_data.py — provenance column, explicit provider
    # failure (exit 2, no file), --years trim on every source, no implicit
    # synthetic substitution. Stdlib-only; needs only the venv python.
    if(EXISTS "${OMNISEED_VENV_PY}")
        add_test(NAME omniseed_market_data
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_market_data.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_market_data PROPERTIES TIMEOUT 300)

        # Market-feed adapters (M1): rate limiting, TTL cache, HTTP retry,
        # symbol normalisation, per-asset bar/ticker adapters, provenance-CSV
        # reuse, and the meme research-only refusal. Fully offline.
        add_test(NAME omniseed_market_feeds
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_market_feeds.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_market_feeds PROPERTIES TIMEOUT 300)

        # 24/7 paper loop (M5): watch parsing, provenance-CSV merge/dedup, feed
        # health -> ABSTAIN, engine invocation, status heartbeat. Offline
        # (probe/fetch/runner injected).
        add_test(NAME omniseed_paper_loop
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_paper_loop.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_paper_loop PROPERTIES TIMEOUT 300)

        # Nightly report + single-file dashboard (M5): journal parsing, book
        # replay, day P&L, Markdown/HTML rendering, opt-in SMTP config.
        add_test(NAME omniseed_paper_reports
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_paper_reports.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_paper_reports PROPERTIES TIMEOUT 300)

        # Monster layer (tools/monster): no-look-ahead feature primitives.
        add_test(NAME omniseed_monster_features
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_monster_features.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_monster_features PROPERTIES TIMEOUT 300)

        # Monster sniper engine: layer arithmetic, hard vetoes, the reachable
        # 0.85 bar, the [1%,2%] adaptive sizing band, cross-asset invalidation.
        add_test(NAME omniseed_monster_sniper
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_monster_sniper.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_monster_sniper PROPERTIES TIMEOUT 300)

        # Monster news hunter + state vector + scan CLI (offline).
        add_test(NAME omniseed_monster_news
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_monster_news.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_monster_news PROPERTIES TIMEOUT 300)

        # Monster regime engine: calibration of every statistic against series
        # with KNOWN ground truth (trend / OU / random walk), plus the
        # no-look-ahead prefix-invariance proof.
        add_test(NAME omniseed_monster_regime
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_monster_regime.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_monster_regime PROPERTIES TIMEOUT 300)

        # Monster strategy zoo + regime router: the exact CKS order flow
        # imbalance against hand-computed values, and the ensemble veto weight
        # floor (a lone weak signal must never veto).
        add_test(NAME omniseed_monster_strategies
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_monster_strategies.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_monster_strategies PROPERTIES TIMEOUT 300)

        # Monster funding carry: annualization arithmetic, the net-of-cost
        # stack, and the memecoin trap (high gross APR must NOT rank first).
        add_test(NAME omniseed_monster_funding
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_VENV_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_monster_funding.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_monster_funding PROPERTIES TIMEOUT 300)
    endif()

    # Decision bridge (T2 integration): the C++ DecisionResult JSON contract,
    # fail-closed screening ("silence is not consent"), entry provenance
    # round-trip through the frozen journal CSV, the dashboard panels
    # (entry/stop/confidence, day P&L, regime), the DecisionGate fail-closed
    # matrix, regime reading, and the local HTTP server routes.
    #
    # Stdlib-only, so it deliberately does NOT sit behind the .venv gate above:
    # it must run in CI (where no .venv exists) because it re-reads the C++
    # risk headers and fails if the Python copy of the 2%/3%/8% limits drifts.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(EXISTS "${OMNISEED_VENV_PY}")
        set(OMNISEED_TEST_PY "${OMNISEED_VENV_PY}")
    elseif(Python3_Interpreter_FOUND)
        set(OMNISEED_TEST_PY "${Python3_EXECUTABLE}")
    else()
        set(OMNISEED_TEST_PY "")
    endif()
    if(OMNISEED_TEST_PY)
        add_test(NAME omniseed_decision_bridge
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_TEST_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_decision_bridge.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_decision_bridge PROPERTIES TIMEOUT 300)

        # Regime parity: the C++ engine must produce the SAME NUMBERS as the
        # Python oracle it replaces. This is the gate that makes retiring the
        # Python from the runtime path safe — a hand-port of 700 lines of
        # numerics WILL drift otherwise, and a silently-diverging second
        # implementation is worse than no port at all.
        # Stdlib-only; also registered outside the .venv gate on purpose.
        add_test(NAME omniseed_regime_parity
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_TEST_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_regime_parity.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_regime_parity PROPERTIES TIMEOUT 900)

        # Strategy parity: same contract for the strategy zoo, router and
        # sniper (~1,400 lines of numerics ported from three Python modules).
        # Compares every signal's direction/confidence/reason, every router
        # field, and every sniper layer on synthetic trend/chop/squeeze/vol
        # series and on a real feed when one is present.
        add_test(NAME omniseed_strategy_parity
                 COMMAND ${CMAKE_COMMAND} -E env
                     PYTHONIOENCODING=utf-8
                     "${OMNISEED_TEST_PY}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_strategy_parity.py"
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(omniseed_strategy_parity PROPERTIES TIMEOUT 900)
    endif()

    # System-1 decision head ("two heads, one brain"): action space + labels,
    # the one-matvec contract, softmax/argmax, fail-closed routing (ABSTAIN and
    # EXPLAIN never self-route), confidence/margin gates, determinism, input
    # immutability, persistence round-trip + rejection paths, and the >=100x
    # speed claim against the cheapest possible System-2 token step.
    # Fully offline: no model, no GGUF, no network.
    add_executable(omniseed_decision_head tests/test_decision_head.cpp)
    target_include_directories(omniseed_decision_head PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_decision_head PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_decision_head PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_decision_head PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_decision_head COMMAND omniseed_decision_head
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_decision_head PRIVATE _CRT_SECURE_NO_WARNINGS)

    # C++ regime engine (port of tools/monster/regime.py): the statistics, the
    # causal window (prefix invariance), the label logic, hysteresis, and the
    # regressions for the four bugs that were fixed in the Python.
    # Fully offline: no model, no GGUF, no network.
    add_executable(omniseed_regime_engine tests/test_regime_engine.cpp)
    target_include_directories(omniseed_regime_engine PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_regime_engine PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_regime_engine PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_regime_engine PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_regime_engine COMMAND omniseed_regime_engine
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_regime_engine PRIVATE _CRT_SECURE_NO_WARNINGS)

    # C++ strategy zoo + router + sniper (ports of tools/monster/strategies.py,
    # router.py, sniper_engine.py). Covers the B4 regression (a lone
    # low-confidence signal must not veto), the "ensemble blocks only, never
    # promotes" invariant, the 0.85-gate reachability table, and prefix
    # invariance for all three layers. Fully offline.
    add_executable(omniseed_strategy_zoo tests/test_strategy_zoo.cpp)
    target_include_directories(omniseed_strategy_zoo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_strategy_zoo PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_strategy_zoo PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_strategy_zoo PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_strategy_zoo COMMAND omniseed_strategy_zoo
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_strategy_zoo PRIVATE _CRT_SECURE_NO_WARNINGS)

    # The head stack ("one backbone, multiple heads"): the shared vocabulary,
    # the classification head (many label sets over ONE projection), the scoring
    # head (three INDEPENDENT sigmoids — a softmax could not express "urgent and
    # unimportant at once"), and the domain-general decision head with its
    # fail-closed routing ladder. Fully offline: no model, no GGUF, no network.
    add_executable(omniseed_heads tests/test_heads.cpp)
    target_include_directories(omniseed_heads PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_heads PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_heads PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_heads PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_heads COMMAND omniseed_heads
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_heads PRIVATE _CRT_SECURE_NO_WARNINGS)

    # The HeadRouter: per-head gates, the domain softmax, fail-closed planning
    # (no gate fired -> run the SUPERSET, never a shortcut), every CLI mode
    # spelling, and refine()'s downgrade-only fast path. Also MEASURES the
    # mandate's "<100 microseconds" budget at the real edge width instead of
    # asserting it in a comment. Fully offline.
    add_executable(omniseed_router tests/test_router.cpp)
    target_include_directories(omniseed_router PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_router PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_router PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_router PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_router COMMAND omniseed_router
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_router PRIVATE _CRT_SECURE_NO_WARNINGS)

    # The unified document + pipeline: present-iff-it-ran, normalise() dropping
    # any field the activation plan does not name (a stale value from a previous
    # turn must not leak into this one), the mandate's JSON key order, and the
    # fast path proven by INVOCATION COUNT — a timing number cannot distinguish
    # "the text head was skipped" from "the text head was fast". Fully offline.
    add_executable(omniseed_unified_output tests/test_unified_output.cpp)
    target_include_directories(omniseed_unified_output PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_unified_output PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_unified_output PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_unified_output PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_unified_output COMMAND omniseed_unified_output
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_unified_output PRIVATE _CRT_SECURE_NO_WARNINGS)

    # The language domain, in C++ with no model and no Python: UTF-8 script
    # detection (Bengali from its own Unicode block), the tokenizer, intent
    # rules for EN and BN, and the sentiment negation SCOPING rule ("not bad,
    # it's good" must be positive). The romanised-Bengali gap is pinned as a
    # known limitation rather than left implicit. Fully offline.
    add_executable(omniseed_language_heads tests/test_language_heads.cpp)
    target_include_directories(omniseed_language_heads PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_language_heads PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_language_heads PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_language_heads PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_language_heads COMMAND omniseed_language_heads
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_language_heads PRIVATE _CRT_SECURE_NO_WARNINGS)

    # The soul: the persona's commitments and refusal rules, emotional fusion
    # through the existing runtime/emotional layer, the overclaim rewriter that
    # makes "guaranteed" unable to leave the process, and the honest
    # self-report built from resolved decisions. Registered OUTSIDE the .venv
    # gate on purpose — a gate that never executes is not a gate.
    add_executable(omniseed_soul tests/test_soul.cpp)
    target_include_directories(omniseed_soul PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_soul PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_soul PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_soul PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_soul COMMAND omniseed_soul
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_soul PRIVATE _CRT_SECURE_NO_WARNINGS)

    # MILESTONE 4 — the calibration contract. The mechanism half (temperature
    # scaling, per-set calibration, v2/v3 blob compatibility, provenance) is
    # registered OUTSIDE any .venv gate on purpose: a gate that never executes
    # is not a gate. The held-out half is gated on the committed
    # models/heads/*.bin + tests/fixtures/head_calibration/* pair and prints a
    # SKIP reason when they are absent.
    add_executable(omniseed_calibration tests/test_calibration.cpp)
    target_include_directories(omniseed_calibration PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_calibration PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_calibration PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_calibration PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_calibration COMMAND omniseed_calibration
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_calibration PRIVATE _CRT_SECURE_NO_WARNINGS)

    # MILESTONE 5 — the multimodal joint. Part A/B (TokenBus + MultimodalBridge)
    # are UNGATED: they build a minimal tokenizer and a synthetic codebook, so
    # they run everywhere including CI. TokenBus had zero tests before this.
    # Part C is gated on models/rwkv7-0.1B-ternary.gguf and prints a SKIP reason
    # when it is absent.
    add_executable(omniseed_multimodal tests/test_multimodal.cpp)
    target_include_directories(omniseed_multimodal PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_multimodal PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_multimodal PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_multimodal PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_multimodal COMMAND omniseed_multimodal
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_multimodal PRIVATE _CRT_SECURE_NO_WARNINGS)

    # MILESTONE 6 — HARD RULE 5 coverage for the four modules that had none
    # (GrammarDecoder, ComputeThrottle, SelfImprovement, FocalCodec). All four
    # are pure and model-free, so the whole suite is UNGATED and runs in CI.
    add_executable(omniseed_agent_modules tests/test_agent_modules.cpp)
    target_include_directories(omniseed_agent_modules PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_agent_modules PRIVATE omniseed_core)
    if(MSVC)
        target_compile_options(omniseed_agent_modules PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_agent_modules PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME omniseed_agent_modules COMMAND omniseed_agent_modules
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(omniseed_agent_modules PRIVATE _CRT_SECURE_NO_WARNINGS)

    # Dumper used by tests/test_regime_parity.py to diff C++ against the Python
    # oracle. Not a test itself.
    add_executable(omniseed_regime_dump tests/regime_dump.cpp)
    target_include_directories(omniseed_regime_dump PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_regime_dump PRIVATE omniseed_core)
    target_compile_definitions(omniseed_regime_dump PRIVATE _CRT_SECURE_NO_WARNINGS)

    # Dumper used by tests/test_strategy_parity.py to diff the C++ strategy zoo,
    # router and sniper against their Python oracles. Not a test itself.
    add_executable(omniseed_strategy_dump tests/strategy_dump.cpp)
    target_include_directories(omniseed_strategy_dump PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_strategy_dump PRIVATE omniseed_core)
    target_compile_definitions(omniseed_strategy_dump PRIVATE _CRT_SECURE_NO_WARNINGS)

    # Data collection for the OFFLINE head trainer (tools/train_heads.py).
    # Runs the real backbone over the real inputs and writes (h[E], label)
    # pairs. Not a test itself, and never run by ctest — it needs the 243 MB
    # GGUF, which CI does not have.
    add_executable(omniseed_dump_hidden tools/dump_hidden.cpp)
    target_include_directories(omniseed_dump_hidden PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(omniseed_dump_hidden PRIVATE omniseed_core)
    target_compile_definitions(omniseed_dump_hidden PRIVATE _CRT_SECURE_NO_WARNINGS)
    if(MSVC)
        target_compile_options(omniseed_dump_hidden PRIVATE /W4 /utf-8 /Zc:__cplusplus)
    else()
        target_compile_options(omniseed_dump_hidden PRIVATE -Wall -Wextra)
    endif()
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
