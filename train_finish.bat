@echo off
cd /d C:\Users\sakim\OneDrive\Desktop\omniseed
for /L %%i in (1,1,8) do (
  echo [%time%] === CHUNK %%i ===
  .venv\Scripts\python.exe -u tools\qat_ternary.py --corpus wikitext+tinystories --steps 12000 --eval-every 100 --window 16 --batch 8 --kd-weight 1.0 --kd-temp 2.0 --lr 5e-5 --time-budget 3300
  if not errorlevel 3 goto :eof
)
echo [%time%] === SHESH ===
pause