@echo off
REM ── zluxDOF build script ───────────────────────────────────────────────
REM Builds the Release|x64 .aex and installs it into the AE MediaCore folder.
REM
REM The .vcxproj writes its output to %AE_PLUGIN_BUILD_DIR%. If you already
REM export that variable, it is honored; otherwise we default it to the local
REM build_out\ folder so the build is reproducible from a clean shell.
setlocal
if "%AE_PLUGIN_BUILD_DIR%"=="" set "AE_PLUGIN_BUILD_DIR=%~dp0build_out"
REM All zluxDOF files live in a single MediaCore\zlux subfolder (the .aex finds
REM its aperture_lib / banner relative to itself, so the relative paths still
REM resolve). AE scans MediaCore recursively, so the plugin still loads.
set "INSTALL_DIR=C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\zlux"

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d %~dp0zluxDOF\Win
msbuild zluxDOF.vcxproj /t:Rebuild /p:Configuration=Release /p:Platform=x64 /verbosity:minimal
if errorlevel 1 (
    echo.
    echo *** BUILD FAILED -- not installing. ***
    exit /b 1
)

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
REM Install the freshly built plugin so After Effects picks it up on relaunch.
copy /Y "%AE_PLUGIN_BUILD_DIR%\zluxDOF.aex" "%INSTALL_DIR%\zluxDOF.aex" >nul
if errorlevel 1 (
    echo.
    echo *** Built OK but could not copy to "%INSTALL_DIR%".
    echo *** Close After Effects and/or run this script as Administrator.
    exit /b 1
)

REM v2.11+: every image asset (banner, picker montage, all 80 aperture maps)
REM is embedded in the .aex as RCDATA resources (zluxDOF_Assets.rc), so the
REM install is a single self-contained file. Remove the legacy loose assets
REM so the embedded path is what actually runs.
if exist "%INSTALL_DIR%\aperture_lib" rmdir /S /Q "%INSTALL_DIR%\aperture_lib"
if exist "%INSTALL_DIR%\zluxDOF_banner.png" del /Q "%INSTALL_DIR%\zluxDOF_banner.png"
echo.
echo Build + install OK -^> "%INSTALL_DIR%\zluxDOF.aex"
endlocal
