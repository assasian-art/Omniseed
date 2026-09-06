// =============================================================================
//  OmniSeed — platform.cpp
//  Portable implementations: mmap, peak RSS, timing, logging.
//  Windows: Win32 API (CreateFileMapping / GetProcessMemoryInfo)
//  Linux:   POSIX (mmap / VmHWM from /proc)
// =============================================================================
#include "omniseed/core/platform.h"

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <new>
#include <string>

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
    #include <sys/stat.h>
    #include <unistd.h>

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
      last_error_(std::move(other.last_error_))
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

bool MappedFile::open(const std::string& path) {
    close();
    path_ = path;

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

void MappedFile::close() {
#if OMNISEED_PLATFORM_WINDOWS
    if (data_ != nullptr) {
        ::UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (mapping_handle_ != nullptr) {
        ::CloseHandle(static_cast<HANDLE>(mapping_handle_));
        mapping_handle_ = nullptr;
    }
    if (file_handle_ != nullptr) {
        ::CloseHandle(static_cast<HANDLE>(file_handle_));
        file_handle_ = nullptr;
    }
#else
    if (data_ != nullptr) {
        ::munmap(data_, static_cast<size_t>(size_));
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
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

} // namespace platform
} // namespace omniseed
