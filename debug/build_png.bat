@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
set INC=/I..\AfterEffectsSDK\Examples\Headers /I..\AfterEffectsSDK\Examples\Headers\SP /I..\AfterEffectsSDK\Examples\Headers\Win /I..\AfterEffectsSDK\Examples\Resources /I..\AfterEffectsSDK\Examples\Util
set UTIL=..\AfterEffectsSDK\Examples\Util
set CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3

REM Build the CUDA gather kernel first. The harness only ever runs on the build
REM machine, so it targets sm_120 alone and compiles in a few seconds -- the
REM shipped .aex uses the full architecture list (see zluxDOF.vcxproj).
REM The C++ side is then built with /DZLUX_CUDA, which is what makes RenderCore
REM route the gather to the GPU when a device is present.
REM Set ZLUX_NOGPU=1 at run time to force the CPU gather (the reference path).
"%CUDA%\bin\nvcc.exe" -c ..\zluxDOF\zluxDOF_Kernel.cu -o zluxDOF_Kernel.obj ^
   --use-local-env -O3 -m64 -std=c++17 -gencode arch=compute_120,code=sm_120 -Xcompiler /MT
if errorlevel 1 (
    echo.
    echo *** NVCC FAILED
    exit /b 1
)

cl /nologo /EHsc /O2 /MT /std:c++17 /DNDEBUG /DZLUX_PROFILE /DZLUX_CUDA /DMSWindows /DWIN32 /D_WINDOWS /D_CRT_SECURE_NO_WARNINGS %INC% /I"%CUDA%\include" ^
   dof_png.cpp ..\zluxDOF\zluxDOF_Strings.cpp %UTIL%\AEGP_SuiteHandler.cpp %UTIL%\MissingSuiteError.cpp %UTIL%\AEFX_SuiteHelper.c ^
   /Fe:dof_png.exe ^
   zluxDOF_Kernel.obj ^
   /link /LIBPATH:"%CUDA%\lib\x64" cudart.lib ^
   gdiplus.lib user32.lib gdi32.lib ole32.lib shell32.lib advapi32.lib
endlocal
