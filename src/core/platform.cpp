// =============================================================================
//  OmniSeed — platform.cpp
//  Portable implementations: mmap, peak RSS, timing, logging.
//  Windows: Win32 API (CreateFileMapping / GetProcessMemoryInfo)
//  Linux:   POSIX (mmap / VmHWM from /proc)
// =============================================================================
#include "omniseed/core/platform.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <string>

#if defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64))
#include <intrin.h>   // __cpuid, __cpuidex, _xgetbv
#endif

// ---------------------------------------------------------------------------
// Windows includes
// ---------------------------------------------------------------------------
#if OMNISEED_PLATFORM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <winsock2.h>   // MUST precede windows.h (winsock1 conflict guard)
    #include <ws2tcpip.h>
    #include <windows.h>
    #include <psapi.h>
#else
    // ---------------------------------------------------------------------------
    // POSIX includes
    // ---------------------------------------------------------------------------
    #include <cerrno>
    #include <cstring>
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/socket.h>
    #include <sys/stat.h>
    #include <unistd.h>
    #include <arpa/inet.h>   // htonl/ntohl/htons/ntohs

    #if OMNISEED_PLATFORM_MACOS
        #include <mach/mach.h>
    #endif
#endif

namespace omniseed {
namespace platform {

// ===========================================================================
// OS name
// ===========================================================================
const char* os_name() {
#if OMNISEED_PLATFORM_WINDOWS
    return "Windows";
#elif OMNISEED_PLATFORM_MACOS
    return "macOS";
#elif OMNISEED_PLATFORM_LINUX
    return "Linux";
#else
    return "Unknown";
#endif
}

// ===========================================================================
// MappedFile — read-only, zero-copy model mapping
// ===========================================================================
MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(other.data_), size_(other.size_), path_(std::move(other.path_)),
      last_error_(std::move(other.last_error_)),
      fallback_buf_(std::move(other.fallback_buf_))
#if OMNISEED_PLATFORM_WINDOWS
      , file_handle_(other.file_handle_), mapping_handle_(other.mapping_handle_)
#else
      , fd_(other.fd_)
#endif
{
#if OMNISEED_PLATFORM_WINDOWS
    other.file_handle_    = nullptr;
    other.mapping_handle_ = nullptr;
#else
    other.fd_             = -1;
#endif
    other.data_           = nullptr;
    other.size_           = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        close();
        data_       = other.data_;
        size_       = other.size_;
        path_       = std::move(other.path_);
        last_error_ = std::move(other.last_error_);
        fallback_buf_ = std::move(other.fallback_buf_);
#if OMNISEED_PLATFORM_WINDOWS
        file_handle_    = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        other.file_handle_    = nullptr;
        other.mapping_handle_ = nullptr;
#else
        fd_       = other.fd_;
        other.fd_ = -1;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

namespace {
// Fires the fallback WARN at most once per process: a model file is opened
// by several components (weights + GGUF-resident tokenizer), and one notice
// that the RSS profile differs is all the operator needs.
std::atomic<bool> g_fallback_warned{false};
} // namespace

bool MappedFile::open(const std::string& path) {
    close();
    path_ = path;

    // ---------------------------------------------------------------
    // Test/portability escape hatch: OMNISEED_FORCE_FREAD=1 skips mmap
    // entirely and serves tensor views from a full heap copy. Proves the
    // fallback path is byte-identical to the mmap path (same weights,
    // same outputs) on hosts where mmap is unavailable.
    // ---------------------------------------------------------------
    const char* force = std::getenv("OMNISEED_FORCE_FREAD");
    if (force != nullptr &&
        (force[0] == '1' || force[0] == 'y' || force[0] == 'Y')) {
        if (open_fallback_impl(path)) {
            if (!g_fallback_warned.exchange(true)) {
                log_warn(
                    "MappedFile: OMNISEED_FORCE_FREAD=1 — '%s' served from a "
                    "full heap copy (RSS will reflect the whole file).",
                    path.c_str());
            }
            return true;
        }
        // fread failed too (missing file etc.): fall through so the mmap
        // attempt produces the more descriptive OS error.
    }

    // ---------------------------------------------------------------
    // Common path: try a real OS mmap first (zero-copy, lazy paging
    // keeps RSS at touched-weights-only). On any failure fall back to
    // a full read into heap memory — slower to start, byte-identical
    // views, and available on EVERY platform (no mmap syscall needed).
    // ---------------------------------------------------------------
    if (open_mapped_impl(path)) return true;
    const std::string mmap_error = last_error_;
    if (open_fallback_impl(path)) {
        if (!g_fallback_warned.exchange(true)) {
            log_warn(
                "MappedFile: mmap unavailable for '%s' (%s); serving from a "
                "full heap copy (RSS will reflect the whole file).",
                path.c_str(), mmap_error.c_str());
        }
        return true;
    }
    return false;
}

bool MappedFile::open_mapped_impl(const std::string& path) {
#if OMNISEED_PLATFORM_WINDOWS
    // ---------------------------------------------------------------
    // Windows: CreateFileW -> CreateFileMappingW -> MapViewOfFile
    // ---------------------------------------------------------------
    std::wstring wpath;
    if (!path.empty()) {
        const int n = ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(),
                                            static_cast<int>(path.size()),
                                            nullptr, 0);
        wpath.resize(static_cast<size_t>(n));
        ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(),
                              static_cast<int>(path.size()), &wpath[0], n);
    }

    HANDLE hFile = ::CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                 nullptr, OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        last_error_ = "CreateFileW failed (error " +
                      std::to_string(::GetLastError()) + ")";
        return false;
    }

    LARGE_INTEGER fsize;
    if (!::GetFileSizeEx(hFile, &fsize)) {
        last_error_ = "GetFileSizeEx failed (error " +
                      std::to_string(::GetLastError()) + ")";
        ::CloseHandle(hFile);
        return false;
    }
    if (fsize.QuadPart <= 0) {
        last_error_ = "file is empty";
        ::CloseHandle(hFile);
        return false;
    }

    HANDLE hMap = ::CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0,
                                       nullptr);
    if (hMap == nullptr) {
        last_error_ = "CreateFileMappingW failed (error " +
                      std::to_string(::GetLastError()) + ")";
        ::CloseHandle(hFile);
        return false;
    }

    void* view = ::MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
        last_error_ = "MapViewOfFile failed (error " +
                      std::to_string(::GetLastError()) + ")";
        ::CloseHandle(hMap);
        ::CloseHandle(hFile);
        return false;
    }

    file_handle_    = hFile;
    mapping_handle_ = hMap;
    data_           = view;
    size_           = static_cast<uint64_t>(fsize.QuadPart);
    last_error_.clear();
    return true;
#else
    // ---------------------------------------------------------------
    // POSIX: open -> fstat -> mmap
    // ---------------------------------------------------------------
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        last_error_ = std::string("open failed: ") + std::strerror(errno);
        return false;
    }

    struct stat st;
    if (::fstat(fd, &st) != 0) {
        last_error_ = std::string("fstat failed: ") + std::strerror(errno);
        ::close(fd);
        return false;
    }
    if (st.st_size <= 0) {
        last_error_ = "file is empty";
        ::close(fd);
        return false;
    }

    void* mapped = ::mmap(nullptr, static_cast<size_t>(st.st_size),
                          PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        last_error_ = std::string("mmap failed: ") + std::strerror(errno);
        ::close(fd);
        return false;
    }

    fd_   = fd;
    data_ = mapped;
    size_ = static_cast<uint64_t>(st.st_size);
    last_error_.clear();
    return true;
#endif
}

// ---------------------------------------------------------------------------
// fread fallback: read the whole file into a heap buffer and serve tensor
// views from it. Used when mmap is unavailable (exotic platforms, sandboxed
// filesystems) or the mmap syscalls fail. data_ points at fallback_buf_.data().
// ---------------------------------------------------------------------------
bool MappedFile::open_fallback_impl(const std::string& path) {
    std::FILE* f = open_file_c(path.c_str(), "rb");
    if (f == nullptr) {
        last_error_ = "open failed (mmap and fread fallback)";
        return false;
    }
    if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); last_error_ = "seek failed"; return false; }
    const long fsize = std::ftell(f);
    if (fsize <= 0)      { std::fclose(f); last_error_ = "file is empty (fallback)"; return false; }
    std::rewind(f);

    fallback_buf_.resize(static_cast<size_t>(fsize));
    const size_t got = fallback_buf_.empty()
        ? 0u
        : std::fread(fallback_buf_.data(), 1, fallback_buf_.size(), f);
    std::fclose(f);
    if (got != fallback_buf_.size()) {
        fallback_buf_.clear();
        fallback_buf_.shrink_to_fit();
        last_error_ = "short read (fallback)";
        return false;
    }
    data_ = fallback_buf_.data();
    size_ = fallback_buf_.size();
    last_error_.clear();
    return true;
}

void MappedFile::close() {
#if OMNISEED_PLATFORM_WINDOWS
    if (data_ != nullptr && fallback_buf_.empty()) {
        ::UnmapViewOfFile(data_);
    }
    data_ = nullptr;
    if (mapping_handle_ != nullptr) {
        ::CloseHandle(static_cast<HANDLE>(mapping_handle_));
        mapping_handle_ = nullptr;
    }
    if (file_handle_ != nullptr) {
        ::CloseHandle(static_cast<HANDLE>(file_handle_));
        file_handle_ = nullptr;
    }
#else
    if (data_ != nullptr && fallback_buf_.empty()) {
        ::munmap(data_, static_cast<size_t>(size_));
    }
    data_ = nullptr;
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
    fallback_buf_.clear();
    fallback_buf_.shrink_to_fit();
    size_ = 0;
}

// ===========================================================================
// Page size
// ===========================================================================
uint32_t page_size() {
#if OMNISEED_PLATFORM_WINDOWS
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return si.dwPageSize;
#else
    long v = ::sysconf(_SC_PAGESIZE);
    return v > 0 ? static_cast<uint32_t>(v) : 4096u;
#endif
}

// ===========================================================================
// Aligned allocation
// ===========================================================================
void* aligned_alloc_omni(size_t size, size_t alignment) {
    if (size == 0) size = 1;
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
#if OMNISEED_PLATFORM_WINDOWS
    return ::_aligned_malloc(size, alignment);
#else
    void* p = nullptr;
    if (::posix_memalign(&p, alignment, size) != 0) return nullptr;
    return p;
#endif
}

void aligned_free_omni(void* ptr) {
#if OMNISEED_PLATFORM_WINDOWS
    ::_aligned_free(ptr);
#else
    ::free(ptr);
#endif
}

// ===========================================================================
// Peak / current RSS
// ===========================================================================
uint64_t peak_rss_bytes() {
#if OMNISEED_PLATFORM_WINDOWS
    PROCESS_MEMORY_COUNTERS pmc;
    if (::GetProcessMemoryInfo(::GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<uint64_t>(pmc.PeakWorkingSetSize);
    }
    return 0;
#elif OMNISEED_PLATFORM_MACOS
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<uint64_t>(info.resident_size);
    }
    return 0;
#else
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    uint64_t vm_hwm = 0;
    while (std::fgets(line, sizeof(line), f)) {
        // VmHWM:    123456 kB   (peak resident set size, kernel-maintained)
        if (std::sscanf(line, "VmHWM: %lu kB", &vm_hwm) == 1) break;
    }
    std::fclose(f);
    return vm_hwm * 1024ull;
#endif
}

uint64_t current_rss_bytes() {
#if OMNISEED_PLATFORM_WINDOWS
    PROCESS_MEMORY_COUNTERS pmc;
    if (::GetProcessMemoryInfo(::GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<uint64_t>(pmc.WorkingSetSize);
    }
    return 0;
#elif OMNISEED_PLATFORM_MACOS
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<uint64_t>(info.resident_size);
    }
    return 0;
#else
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    uint64_t vm_rss = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::sscanf(line, "VmRSS: %lu kB", &vm_rss) == 1) break;
    }
    std::fclose(f);
    return vm_rss * 1024ull;
#endif
}

// ===========================================================================
// mmap-mode probe + thread yield + UTF-8 console + portable fopen
// ===========================================================================
bool mapped_with_fallback(const MappedFile& mf) {
    // A fallback-backed mapping owns its bytes in fallback_buf_; real mmaps
    // do not. Probe via the accessor on the real object (no const hack).
    return mf.using_fallback();
}

void yield_now() {
#if OMNISEED_PLATFORM_WINDOWS
    ::SwitchToThread();
#else
    ::sched_yield();
#endif
}

void enable_utf8_console() {
#if OMNISEED_PLATFORM_WINDOWS
    ::SetConsoleOutputCP(CP_UTF8);
#endif
    // POSIX terminals: nothing to do — already UTF-8-clean.
}

std::FILE* open_file_c(const char* path_utf8, const char* mode) {
#if OMNISEED_PLATFORM_WINDOWS
    // Convert UTF-8 -> wide and use _wfopen so non-ASCII model paths work.
    if (path_utf8 == nullptr) return nullptr;
    const int n = ::MultiByteToWideChar(CP_UTF8, 0, path_utf8, -1, nullptr, 0);
    if (n <= 0) return nullptr;
    std::wstring w(static_cast<size_t>(n), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, path_utf8, -1, &w[0], n);
    const int mn = ::MultiByteToWideChar(CP_UTF8, 0, mode, -1, nullptr, 0);
    if (mn <= 0) return nullptr;
    std::wstring wm(static_cast<size_t>(mn), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, mode, -1, &wm[0], mn);
    return ::_wfopen(w.c_str(), wm.c_str());
#else
    return std::fopen(path_utf8, mode);
#endif
}

// ===========================================================================
// Minimal UDP socket shim (the ONLY socket code in the repo)
// ===========================================================================
#if OMNISEED_PLATFORM_WINDOWS
namespace {
struct WinsockOnce {
    bool ok = false;
    WinsockOnce() {
        WSADATA d;
        ok = ::WSAStartup(MAKEWORD(2, 2), &d) == 0;
    }
    ~WinsockOnce() { if (ok) ::WSACleanup(); }
};
WinsockOnce g_winsock;
} // namespace
#endif

int socket_udp_open(uint16_t port) {
#if OMNISEED_PLATFORM_WINDOWS
    if (!g_winsock.ok) return -1;
    const SOCKET s = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == INVALID_SOCKET) return -1;
    const int fd = static_cast<int>(s);
#else
    const int fd = static_cast<int>(::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP));
    if (fd < 0) return -1;
#endif
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&one), sizeof(one));
    ::setsockopt(fd, SOL_SOCKET, SO_BROADCAST,
                 reinterpret_cast<const char*>(&one), sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        socket_udp_close(fd);
        return -1;
    }
#if OMNISEED_PLATFORM_WINDOWS
    u_long nb = 1;
    ::ioctlsocket(fd, FIONBIO, &nb);
#else
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags >= 0) ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
#endif
    return fd;
}

void socket_udp_close(int fd) {
    if (fd < 0) return;
#if OMNISEED_PLATFORM_WINDOWS
    ::closesocket(static_cast<SOCKET>(fd));
#else
    ::close(fd);
#endif
}

bool socket_udp_broadcast(int fd, const void* buf, size_t len, uint16_t port) {
    if (fd < 0 || buf == nullptr || len == 0) return false;
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_addr.s_addr = htonl(INADDR_BROADCAST);   // host 255.255.255.255
    dst.sin_port = htons(port);
    const int n = static_cast<int>(::sendto(
        fd, static_cast<const char*>(buf), static_cast<int>(len), 0,
        reinterpret_cast<const sockaddr*>(&dst), sizeof(dst)));
    return n > 0;
}

bool socket_udp_send(int fd, const void* buf, size_t len,
                     uint32_t ip4_host_order, uint16_t port) {
    if (fd < 0 || buf == nullptr || len == 0) return false;
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_addr.s_addr = htonl(ip4_host_order);
    dst.sin_port = htons(port);
    const int n = static_cast<int>(::sendto(
        fd, static_cast<const char*>(buf), static_cast<int>(len), 0,
        reinterpret_cast<const sockaddr*>(&dst), sizeof(dst)));
    return n > 0;
}

int socket_udp_poll(int fd, void* buf, size_t cap,
                    uint32_t* src_ip4_host_order_out,
                    uint16_t* src_port_host_order_out) {
    if (fd < 0 || buf == nullptr || cap == 0) return -1;
    sockaddr_in src{};
#if OMNISEED_PLATFORM_WINDOWS
    int slen = sizeof(src);
    const int n = static_cast<int>(::recvfrom(
        fd, static_cast<char*>(buf), static_cast<int>(cap), 0,
        reinterpret_cast<sockaddr*>(&src), &slen));
#else
    socklen_t slen = sizeof(src);
    const ssize_t n = ::recvfrom(
        fd, buf, cap, 0, reinterpret_cast<sockaddr*>(&src), &slen);
#endif
    if (n <= 0) return -1;
    if (src_ip4_host_order_out != nullptr)
        *src_ip4_host_order_out = ntohl(src.sin_addr.s_addr);
    if (src_port_host_order_out != nullptr)
        *src_port_host_order_out = ntohs(src.sin_port);
    return n;
}

// ===========================================================================
// Monotonic clock
// ===========================================================================
double now_ms() {
#if OMNISEED_PLATFORM_WINDOWS
    // QPC: monotonic, high-resolution (~100ns), unaffected by system time.
    static LARGE_INTEGER freq = [] {
        LARGE_INTEGER f;
        ::QueryPerformanceFrequency(&f);
        return f;
    }();
    LARGE_INTEGER now;
    ::QueryPerformanceCounter(&now);
    return static_cast<double>(now.QuadPart) * 1000.0 /
           static_cast<double>(freq.QuadPart);
#else
    struct timespec ts;
    ::clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) * 1000.0 +
           static_cast<double>(ts.tv_nsec) / 1.0e6;
#endif
}

// ===========================================================================
// Logging
// ===========================================================================
namespace {
std::mutex g_log_mutex;
bool       g_quiet = false;

void vlog(const char* level, const char* fmt, va_list args) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_quiet && std::strcmp(level, "INFO") == 0) return;
    std::fprintf(stdout, "[omniseed] %-5s ", level);
    std::vfprintf(stdout, fmt, args);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}
} // namespace

void log_info(const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    vlog("INFO", fmt, args);
    va_end(args);
}
void log_warn(const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    vlog("WARN", fmt, args);
    va_end(args);
}
void log_error(const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    vlog("ERROR", fmt, args);
    va_end(args);
}
void set_quiet(bool quiet) { g_quiet = quiet; }

// ---------------------------------------------------------------------------
// CPU feature detection
// ---------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define OMNISEED_X86_ 1
#else
#define OMNISEED_X86_ 0
#endif

CpuFeatures cpu_features() {
    CpuFeatures f;
#if OMNISEED_X86_
#if defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 0);
    const int max_leaf = regs[0];
    if (max_leaf >= 1) {
        __cpuid(regs, 1);
        // ECX: bit 19 = SSE4.1, bit 27 = OSXSAVE, bit 28 = AVX, bit 12 = FMA
        f.sse41 = (regs[2] & (1u << 19)) != 0;
        const bool avx      = (regs[2] & (1u << 28)) != 0;
        const bool osxsave  = (regs[2] & (1u << 27)) != 0;
        const bool fma_flag = (regs[2] & (1u << 12)) != 0;
        if (avx && osxsave && max_leaf >= 7) {
            // OS must have enabled YMM state before AVX2/FMA execution
            const unsigned long long xcr = _xgetbv(0);
            const bool ymm_saved = (xcr & 0x6ull) == 0x6ull;
            __cpuidex(regs, 7, 0);
            const bool avx2 = (regs[1] & (1u << 5)) != 0;   // EBX bit 5
            f.avx2 = ymm_saved && avx2;
            f.fma  = ymm_saved && fma_flag;
        }
    }
#elif defined(__GNUC__)
    __builtin_cpu_init();
    f.sse41 = __builtin_cpu_supports("sse4.1") != 0;
    f.avx2 = __builtin_cpu_supports("avx2") != 0;
    f.fma = __builtin_cpu_supports("fma") != 0;
#endif
#endif // OMNISEED_X86_
    return f;
}

} // namespace platform
} // namespace omniseed
