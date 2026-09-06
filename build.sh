#!/usr/bin/env bash
# ===========================================================================
#  OmniSeed build script — Linux / macOS / MinGW shell
#
#  Usage:
#    ./build.sh              Release build + tests
#    ./build.sh debug        Debug build + tests
#    ./build.sh server       Release build incl. HTTP server (OMNISEED_HTTP)
#    ./build.sh clean        Remove build directory
# ===========================================================================
set -e

BUILDDIR=build
BUILDTYPE=Release
case "${1:-}" in
    debug) BUILDTYPE=Debug ;;
    server) BUILDTYPE=Release ;;
    clean)
        echo "[omniseed] cleaning ${BUILDDIR} ..."
        rm -rf "${BUILDDIR}"
        echo "[omniseed] done."
        exit 0
        ;;
esac

if ! command -v cmake >/dev/null 2>&1; then
    echo "[omniseed] ERROR: cmake not found. Install cmake 3.16+." >&2
    exit 1
fi

# Prefer Ninja if available (fastest), fall back to Unix Makefiles.
GEN="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
    GEN="Ninja"
fi

SERVERFLAG=""
if [ "${1:-}" = "server" ]; then
    SERVERFLAG="-DOMNISEED_BUILD_SERVER=ON"
fi

echo "[omniseed] Toolchain: $(command -v g++ || command -v clang++) with ${GEN}"
echo "[omniseed] Configuring (${BUILDTYPE})..."
cmake -S . -B "${BUILDDIR}" -G "${GEN}" \
    -DCMAKE_BUILD_TYPE="${BUILDTYPE}" ${SERVERFLAG}

echo "[omniseed] Building..."
cmake --build "${BUILDDIR}" --parallel "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "[omniseed] Running platform tests..."
ctest --test-dir "${BUILDDIR}" --output-on-failure || {
    echo "[omniseed] WARNING: tests failed."
    exit 1
}

echo
echo "[omniseed] Build OK. Binaries in ${BUILDDIR}/bin/"
[ -f "${BUILDDIR}/bin/omniseed" ]        && echo "[omniseed]   omniseed         - CLI agent kernel"
[ -f "${BUILDDIR}/bin/omniseed_tests" ]  && echo "[omniseed]   omniseed_tests   - test suite"
