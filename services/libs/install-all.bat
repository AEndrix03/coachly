@echo off
setlocal enabledelayedexpansion

set "ROOT_DIR=%~dp0"
set "HAS_ERRORS=0"

for /d %%D in ("%ROOT_DIR%*") do (
  if exist "%%~fD\pom.xml" (
    echo ==> %%~nxD
    pushd "%%~fD"
    call mvn install -DskipTests=true
    if errorlevel 1 (
      echo FAILED: %%~nxD
      set "HAS_ERRORS=1"
    )
    popd
  )
)

if "%HAS_ERRORS%"=="1" (
  echo.
  echo One or more builds failed.
) else (
  echo.
  echo All builds completed successfully.
)

pause
endlocal
