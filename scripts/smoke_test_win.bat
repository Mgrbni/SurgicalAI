@echo off
setlocal
if not exist dist\SurgicalAI.exe (
  echo EXE not found & exit /b 1
)
set OUT=%TEMP%\sai_demo
dist\SurgicalAI.exe --demo --out %OUT%
if exist %OUT%\report.pdf (
  echo smoke test passed
  exit /b 0
) else (
  echo missing report
  exit /b 1
)
endlocal
