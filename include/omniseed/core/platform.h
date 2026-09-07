// =============================================================================
//  OmniSeed — platform.h
//  Portable OS abstraction: memory-mapped files, peak RSS, page size, timing.
//
//  Target platforms:
//    * Windows (MSVC / MinGW-w64)  — Win32 API
//    * Linux / Docker / Render     — POSIX
//
//  Part of the OmniSeed "Sparse Fusion Core" runtime (sub-400MB constraint).
//  Everything here is header-only declarations; implementations in
//  src/core/platform.cpp. Zero external dependencies.
//
//  Why mmap matters for OmniSeed: the GGUF model file is mapped read-only
//  into the address space (zero-copy), so tensor weights are faulted in
//  lazily by the OS page cache instead of being duplicated into the heap.
//  This keeps peak RSS at roughly (mapped weights actually touched +
//  small activation buffers), which is essential for the <300MB target.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace omniseed {

// ---------------------------------------------------------------------------
// OS / compiler identification (compile-time constants)
// ---------------------------------------------------------------------------
#define OMNISEED_PLATFORM_WINDOWS 0
#define OMNISEED_PLATFORM_LINUX   0
#define OMNISEED_PLATFORM_MACOS   0
#define OMNISEED_PLATFORM_POSIX   0

#if defined(_WIN32)
#undef  OMNISEED_PLATFORM_WINDOWS
#define OMNISEED_PLATFORM_WINDOWS 1
#elif defined(__APPLE__) && defined(__MACH__)
#undef  OMNISEED_PLATFORM_MACOS
#define OMNISEED_PLATFORM_MACOS   1
#undef  OMNISEED_PLATFORM_POSIX
#define OMNISEED_PLATFORM_POSIX   1
#elif defined(__unix__) || defined(__unix)
#undef  OMNISEED_PLATFORM_LINUX
#define OMNISEED_PLATFORM_LINUX   1
#undef  OMNISEED_PLATFORM_POSIX
#define OMNISEED_PLATFORM_POSIX   1
#endif

#if OMNISEED_PLATFORM_WINDOWS
    #define OMNISEED_EXPORT __declspec(dllexport)
#else
    #define OMNISEED_EXPORT __attribute__((visibility("default")))
#endif

namespace platform {

// ---------------------------------------------------------------------------
// Human-readable OS name, e.g. "Windows", "Linux", "macOS".
// ---------------------------------------------------------------------------
const char* os_name();

// ---------------------------------------------------------------------------
// Memory-mapped file — read-only zero-copy mapping of a model file.
//
// Windows: CreateFileMappingW + MapViewOfFile
// Linux:   open + fstat + mmap(PROT_READ, MAP_PRIVATE)
//
// The mapping is backed by the OS page cache: pages are faulted in on first
// access and can be evicted under memory pressure, so RSS tracks the subset
// of weights actually touched by inference.
// ---------------------------------------------------------------------------
class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    // Non-copyable (owns an OS mapping handle).
    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    // Movable.
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    // Maps `path` read-only. Returns false (and logs via last_error())
    // if the file cannot be opened or mapped.
    bool open(const std::string& path);

    // Unmaps the file (idempotent).
    void close();

    bool        valid()  const { return data_ != nullptr && size_ > 0; }
    const void* data()   const { return data_;   }
    uint8_t*    bytes()  const { return static_cast<uint8_t*>(const_cast<void*>(data_)); }
    uint64_t    size()   const { return size_;   }
    const std::string& path() const { return path_; }
    // True when the bytes are served from the heap fread fallback (no real mmap).
    bool        using_fallback() const { return !fallback_buf_.empty(); }

    // Last OS error string (useful for diagnostics on open failure).
    const std::string& last_error() const { return last_error_; }

private:
    // Real mmap path (Windows MapViewOfFile / POSIX mmap). On failure falls
    // back to open_fallback_impl() — a full fread into heap memory.
    bool open_mapped_impl(const std::string& path);
    bool open_fallback_impl(const std::string& path);

    void*       data_       = nullptr;
    uint64_t    size_       = 0;
    std::string path_;
    std::string last_error_;

#if OMNISEED_PLATFORM_WINDOWS
    void* file_handle_     = nullptr;   // HANDLE
    void* mapping_handle_  = nullptr;   // HANDLE
#else
    int   fd_              = -1;        // POSIX open fd (real mmap path)
#endif
    // fread fallback backing (used when real mmap is unavailable or fails).
    std::vector<uint8_t> fallback_buf_;
};

// ---------------------------------------------------------------------------
// Page size of the host system in bytes (4096 on x86_64 Windows/Linux).
// ---------------------------------------------------------------------------
uint32_t page_size();

// ---------------------------------------------------------------------------
// Did open() succeed via a real mmap (true) or the fread fallback (false)?
// The fallback engages automatically on platforms/conditions where mmap is
// unavailable or fails; tensor views are byte-identical either way — only the
// backing changes (OS page cache vs a heap copy).
// ---------------------------------------------------------------------------
bool mapped_with_fallback(const MappedFile& mf);

// ---------------------------------------------------------------------------
// Cooperative thread yield (poll loops, beacon polling).
// ---------------------------------------------------------------------------
void yield_now();

// ---------------------------------------------------------------------------
// Best-effort UTF-8 console enablement (Windows codepage 65001).
// No-op on POSIX (terminals are already UTF-8-clean); never fatal.
// ---------------------------------------------------------------------------
void enable_utf8_console();

// ---------------------------------------------------------------------------
// fopen wrapper that tolerates UTF-8 paths on Windows (fopen_s/_wfopen path)
// and is plain fopen() elsewhere. Signature matches std::fopen for drop-in use.
// ---------------------------------------------------------------------------
std::FILE* open_file_c(const char* path_utf8, const char* mode);

// ---------------------------------------------------------------------------
// Minimal UDP socket shim — the ONLY place in the codebase that touches
// WinSock/POSIX sockets. UdpBeacon (swarm.cpp) is ifdef-free on top of this.
//   * socket_udp_open() binds INADDR_ANY:port (SO_REUSEADDR + SO_BROADCAST)
//     and sets non-blocking mode; returns -1 on failure.
//   * sendto/broadcast; ip4 args are HOST order (0x7F000001 = 127.0.0.1) —
//     the shim owns all htons/htonl conversions internally.
//   * poll returns >=0 bytes received (-1 = nothing / error), fills src ip4
//     (host order) and src port (host order) when the out-params are set.
// ---------------------------------------------------------------------------
int   socket_udp_open(uint16_t port);
void  socket_udp_close(int fd);
bool  socket_udp_broadcast(int fd, const void* buf, size_t len, uint16_t port);
bool  socket_udp_send(int fd, const void* buf, size_t len,
                      uint32_t ip4_host_order, uint16_t port);
int   socket_udp_poll(int fd, void* buf, size_t cap,
                      uint32_t* src_ip4_host_order_out,
                      uint16_t* src_port_host_order_out);

// ---------------------------------------------------------------------------
// Aligned allocation helper (needed for potential SIMD-friendly buffers).
// Frees with aligned_free(). size must be > 0; alignment must be a power of 2.
// ---------------------------------------------------------------------------
void*  aligned_alloc_omni(size_t size, size_t alignment = 64);
void   aligned_free_omni(void* ptr);

// ---------------------------------------------------------------------------
// Peak (high-water mark) resident set size of THIS process, in bytes.
//
//   Windows: GetProcessMemoryInfo -> PeakWorkingSetSize   (psapi.h)
//   Linux:   /proc/self/status -> VmHWM                   (kernel-maintained)
//   macOS:   mach task_info -> resident_size peak approximation
//
// This is the number the <300MB budget is validated against. VmHWM on Linux
// is monotonic since process start, exactly matching PeakWorkingSetSize.
// ---------------------------------------------------------------------------
uint64_t peak_rss_bytes();

// ---------------------------------------------------------------------------
// Current resident set size, in bytes.
//   Windows: GetProcessMemoryInfo -> WorkingSetSize
//   Linux:   /proc/self/status -> VmRSS
// ---------------------------------------------------------------------------
uint64_t current_rss_bytes();

// ---------------------------------------------------------------------------
// Monotonic high-resolution clock, in milliseconds (for tok/s benchmarks).
// ---------------------------------------------------------------------------
double now_ms();

// ---------------------------------------------------------------------------
// Simple stdout logger with level prefixes. Thread-safe enough for CLI use.
//   [omniseed] INFO  ...
//   [omniseed] WARN  ...
//   [omniseed] ERROR ...
// ---------------------------------------------------------------------------
void log_info(const char* fmt, ...);
void log_warn(const char* fmt, ...);
void log_error(const char* fmt, ...);

// Set to false to silence info logs (errors always print).
void set_quiet(bool quiet);

// ---------------------------------------------------------------------------
// CPU feature detection (CPUID on x86/x64; conservative on other arches).
// Used for runtime SIMD kernel dispatch — a machine without AVX2 must never
// execute AVX2 code, independent of what the build compiled.
// ---------------------------------------------------------------------------
struct CpuFeatures {
    bool sse41 = false;
    bool avx2  = false;
    bool fma   = false;
};
CpuFeatures cpu_features();

} // namespace platform
} // namespace omniseed
