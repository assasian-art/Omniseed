# =============================================================================
#  OmniSeed — Dockerfile (deploy image)
#  Pure C++17 runtime, Alpine + GCC, no Python, no external dependencies.
#
#  Build:   docker build -t omniseed .
#  Run:     docker run -p 8080:8080 -v $(pwd)/models:/app/models omniseed
#
#  Memory budget: the image runs the HTTP server with a hard RSS target of
#  <300 MB (see the bench command; the kernel exits non-zero if exceeded).
# =============================================================================

FROM alpine:3.20 AS builder

RUN apk add --no-cache build-base cmake

WORKDIR /src
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src
COPY tests ./tests

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DOMNISEED_BUILD_TESTS=ON \
        -DOMNISEED_BUILD_CLI=ON \
        -DOMNISEED_BUILD_SERVER=ON \
    && cmake --build build --config Release -j"$(nproc)" \
    && ctest --test-dir build --output-on-failure

# ---------------------------------------------------------------------------
# Runtime stage: tiny image, static-ish binaries, model mounted read-only.
# ---------------------------------------------------------------------------
FROM alpine:3.20

RUN apk add --no-cache libstdc++ && adduser -D -H omniseed

WORKDIR /app
# Single-config CMake generators (Ninja/unix-makefiles) emit binaries into
# build/ directly (build/bin/ is only the VS multi-config layout).
COPY --from=builder /src/build/omniseed /app/omniseed
COPY --from=builder /src/build/omniseed_server /app/omniseed_server

USER omniseed
EXPOSE 8080

# State dir for memory crystals / skills (mount a volume here to persist).
VOLUME ["/app/state"]
ENV OMNISEED_MODEL=/app/models/omniseed.gguf

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8080/health || exit 1

ENTRYPOINT ["/app/omniseed_server"]
CMD ["--port", "8080", "--model", "/app/models/omniseed.gguf"]
