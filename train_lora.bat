@echo off
cd /d C:\Users\sakim\OneDrive\Desktop\omniseed
echo [%time%] === LoRA training শুরু ===
.venv\Scripts\python.exe -u tools\lora_chat.py --steps 3000
if errorlevel 1 (
  echo === flag সমস্যা, defaults-এ চালাচ্ছি ===
  .venv\Scripts\python.exe -u tools\lora_chat.py
)
echo [%time%] === শেষ ===
pause