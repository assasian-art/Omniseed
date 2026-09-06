@echo off
REM ===========================================================================
REM  OmniSeed build script — Windows
REM
REM  Automatically picks the best available toolchain, in order:
REM    1. Visual Studio (MSVC, found via vswhere)
REM    2. MinGW-w64 (g++ on PATH)
REM
REM  Usage:
REM    build.bat              Release build + tests
REM    build.bat debug        Debug build + tests
REM    build.bat server       Release build incl. HTTP server (OMNISEED_HTTP)
REM    build.bat clean        Remove build directory
REM ===========================================================================
setlocal enabledelayedexpansion

set BUILDDIR=build
set BUILDTYPE=Release
if /I "%1"=="debug"  set BUILDTYPE=Debug
if /I "%1"=="server" set BUILDTYPE=Release

if /I "%1"=="clean" (
    echo [omniseed] cleaning %BUILDDIR% ...
    if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
    echo [omniseed] done.
    exit /b 0
)

where cmake >nul 2>nul
if errorlevel 1 (
    echo [omniseed] ERROR: cmake not found on PATH.
    echo            Install CMake 3.16+ or use VS bundled cmake:
    echo            "C:\Program Files\Microsoft Visual Studio\...\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
    exit /b 1
)

REM ------------------------- Toolchain detection ------------------------------
set GENERATOR=
set EXTRA=-A x64

where g++ >nul 2>nul
if not errorlevel 1 (
    set HAVE_MINGW=1
)

REM Prefer MSVC when Visual Studio is present (matches the dev machine).
REM Detect the installed VS year so the exact generator name is used
REM (e.g. "Visual Studio 18 2026" for VS 2026, "Visual Studio 17 2022"...).
set VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist %VSWHERE% (
    for /f "usebackq delims=" %%i in (`%VSWHERE% -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSDIR=%%i
    for /f "usebackq delims=" %%v in (`%VSWHERE% -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property catalog_productLineVersion`) do set VSVER=%%v
)
if defined VSDIR (
    REM Map product line version to release year for the CMake generator name.
    set VSYEAR=
    if "!VSVER!"=="18" set VSYEAR=2026
    if "!VSVER!"=="17" set VSYEAR=2022
    if "!VSVER!"=="16" set VSYEAR=2019
    if "!VSVER!"=="15" set VSYEAR=2017
    if not defined VSYEAR (
        echo [omniseed] ERROR: unmapped Visual Studio version !VSVER!.
        exit /b 1
    )
    echo [omniseed] Toolchain: Visual Studio !VSVER! ^(!VSYEAR^) at !VSDIR!
    set GENERATOR=Visual Studio !VSVER! !VSYEAR!
) else if defined HAVE_MINGW (
    echo [omniseed] Toolchain: MinGW-w64 g++
    set GENERATOR=MinGW Makefiles
    set EXTRA=
) else (
    echo [omniseed] ERROR: no MSVC and no MinGW g++ found.
    exit /b 1
)

REM ------------------------------ Configure -----------------------------------
set SERVERFLAG=
if /I "%1"=="server" set SERVERFLAG=-DOMNISEED_BUILD_SERVER=ON

echo [omniseed] Configuring (%BUILDTYPE%)...
cmake -S . -B %BUILDDIR% -G "%GENERATOR%" %EXTRA% ^
    -DCMAKE_BUILD_TYPE=%BUILDTYPE% %SERVERFLAG%
if errorlevel 1 exit /b 1

REM ------------------------------- Build --------------------------------------
echo [omniseed] Building...
cmake --build %BUILDDIR% --config %BUILDTYPE% --parallel
if errorlevel 1 exit /b 1

REM -------------------------------- Test --------------------------------------
echo [omniseed] Running platform tests...
ctest --test-dir %BUILDDIR% -C %BUILDTYPE% --output-on-failure
if errorlevel 1 (
    echo [omniseed] WARNING: tests failed.
    exit /b 1
)

echo.
echo [omniseed] Build OK. Binaries in %BUILDDIR%\bin\
if exist %BUILDDIR%\bin\omniseed.exe echo [omniseed]   omniseed.exe  - CLI agent kernel
if exist %BUILDDIR%\bin\omniseed_tests.exe echo [omniseed]   omniseed_tests.exe  - test suite
endlocal
