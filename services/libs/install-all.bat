@echo off
setlocal enabledelayedexpansion

set "ROOT_DIR=%~dp0"

for /d %%D in ("%ROOT_DIR%*") do (
  if exist "%%~fD\pom.xml" (
    echo ==> %%~nxD
    pushd "%%~fD"
    call mvn install -DskipTests=true
    if errorlevel 1 (
      popd
      exit /b 1
    )
    popd
  )
)

endlocal
