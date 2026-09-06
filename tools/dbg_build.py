import subprocess

bat = r'''@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /nologo /std:c++17 /EHsc /I include /Fe:build\dbg.exe tools\dbg_sensory.cpp src\runtime\sensory.cpp src\core\platform.cpp
'''

with open('build/dbg_build.bat', 'w') as f:
    f.write(bat)

r = subprocess.run(['cmd', '/c', 'build\\dbg_build.bat'], capture_output=True, text=True)
print(r.stdout[-1500:] if r.stdout else '')
print(r.stderr[-300:] if r.stderr else '')

r2 = subprocess.run(['build\\dbg.exe'], capture_output=True, text=True)
print("PROBE OUTPUT:")
print(r2.stdout)
print(r2.stderr)
