@echo off
cd /d C:\Users\sakim\OneDrive\Desktop\omniseed
echo [%time%] === LoRA retrain: guarded + gguf-base ===
.venv\Scripts\python.exe -u tools\lora_chat.py --steps 3000 --gguf-base models\rwkv7-0.1B-ternary.gguf
if errorlevel 1 (
  echo === flag সমস্যা, guarded defaults ===
  .venv\Scripts\python.exe -u tools\lora_chat.py
)
echo [%time%] === শেষ ===
pause