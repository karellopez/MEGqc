@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

:: ═══════════════════════════════════════════════════════════════════════
:: MEGqc Installer — Windows (x86_64)
:: Version:  1.0.0
:: License:  MIT — ANCP Lab, University of Oldenburg
:: Homepage: https://github.com/ANCPLabOldenburg/MEGqc
::
:: Downloads an embeddable Python 3.10 runtime, creates an isolated
:: virtual environment, installs MEGqc from PyPI, and places a Desktop
:: shortcut (.lnk with icon) and a Start Menu entry.
::
:: Icon source: bundled with the meg-qc PyPI package at
::   meg_qc\miscellaneous\GUI\assets\macos\AppIcon256.png  (preferred, converted to .ico)
::   meg_qc\miscellaneous\GUI\logo.png  (last-resort fallback)
:: ═══════════════════════════════════════════════════════════════════════

:: ── Configuration ───────────────────────────────────────────────────
set "INSTALLER_VERSION=1.0.0"
set "APP_NAME=MEGqc"
set "BASEDIR=%USERPROFILE%\MEGqc"
set "PYDIR=%BASEDIR%\python310"
set "ENVDIR=%BASEDIR%\env"
set "LOG_FILE=%BASEDIR%\install.log"
set "PYPI_PKG=meg-qc"
set "ZIPURL=https://raw.githubusercontent.com/ANCPLabOldenburg/MEGqc/main/external/python-embed/python-3.10.11-embed-amd64.zip"

:: Determine Desktop and Start Menu paths
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set "DESKTOP=%%D"
set "STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs\MEGqc"

:: ── Banner ──────────────────────────────────────────────────────────
cls
echo.
echo     ╔══════════════════════════════════════════════════╗
echo     ║                                                  ║
echo     ║         M E G q c   I n s t a l l e r            ║
echo     ║                                                  ║
echo     ║      Automated MEG ^& EEG Quality Control        ║
echo     ║        ANCP Lab · University of Oldenburg        ║
echo     ║                                                  ║
echo     ╚══════════════════════════════════════════════════╝
echo.
echo   Installer version:  %INSTALLER_VERSION%
echo   Platform:           Windows (x86_64)
echo   Install directory:  %BASEDIR%
echo.

:: ── Pre-flight checks ──────────────────────────────────────────────

:: 1. Check for existing installation
if exist "%ENVDIR%\Scripts\activate.bat" (
    echo   [!] An existing MEGqc installation was detected at:
    echo       %BASEDIR%
    echo.
    echo   [u] Upgrade — keep settings, reinstall the package
    echo   [r] Reinstall — remove everything and start fresh
    echo   [a] Abort
    echo.
    :CHOICE_LOOP
    set /p "CHOICE=  Your choice [u/r/a]: "
    if /i "!CHOICE!"=="u" (
        echo   [*] Upgrading existing installation...
        set "UPGRADE_MODE=1"
        goto :AFTER_CHOICE
    )
    if /i "!CHOICE!"=="r" (
        echo   [*] Removing existing installation...
        rmdir /s /q "%BASEDIR%" 2>nul
        set "UPGRADE_MODE=0"
        goto :AFTER_CHOICE
    )
    if /i "!CHOICE!"=="a" (
        echo   Installation aborted.
        pause
        exit /b 0
    )
    echo   Please enter u, r, or a.
    goto :CHOICE_LOOP
) else (
    set "UPGRADE_MODE=0"
)
:AFTER_CHOICE

:: 2. Confirmation
echo.
echo   The installer will:
echo     1. Download an embeddable Python 3.10 runtime (~12 MB)
echo     2. Create an isolated virtual environment
echo     3. Install MEGqc and all dependencies (~1.5 GB)
echo     4. Create a Desktop shortcut and Start Menu entry with icon
echo        MEGqc opens in a terminal window so you can follow its progress.
echo.
set /p "CONFIRM=  Continue with installation? [Y/n] "
if /i "!CONFIRM!"=="n" (
    echo   Installation cancelled.
    pause
    exit /b 0
)

:: ── Prepare ─────────────────────────────────────────────────────────
if not exist "%BASEDIR%" mkdir "%BASEDIR%"
if %ERRORLEVEL% NEQ 0 (
    echo   [X] ERROR: Could not create base folder.
    pause & exit /b 1
)
cd /d "%BASEDIR%"

echo MEGqc installation started at %DATE% %TIME% > "%LOG_FILE%"
echo Installer version: %INSTALLER_VERSION% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

:: ═════════════════════════════════════════════════════════════════════
:: Step 1/5 — Download embeddable Python
:: ═════════════════════════════════════════════════════════════════════
if "%UPGRADE_MODE%"=="1" if exist "%PYDIR%\python.exe" (
    echo   [OK] Portable Python already present — skipping download.
    goto :SKIP_PYTHON_DL
)

echo   [*] Step 1/5 — Downloading embeddable Python 3.10...

where curl.exe >nul 2>&1
if %ERRORLEVEL%==0 (
    curl.exe -fSL --retry 3 --retry-delay 5 -o python310.zip "%ZIPURL%" >> "%LOG_FILE%" 2>&1
) else (
    powershell -NoProfile -Command "Invoke-WebRequest -Uri '%ZIPURL%' -OutFile 'python310.zip'" >> "%LOG_FILE%" 2>&1
)
if not exist "python310.zip" (
    echo   [X] ERROR: Download failed. Please check your internet connection.
    echo       See log: %LOG_FILE%
    pause & exit /b 1
)

echo   [*] Step 2/5 — Extracting Python...
powershell -NoProfile -Command "Expand-Archive -Force 'python310.zip' 'python310'" >> "%LOG_FILE%" 2>&1
if not exist "%PYDIR%\python.exe" (
    echo   [X] ERROR: Extraction failed — python.exe not found.
    pause & exit /b 1
)
del /q python310.zip
echo   [OK] Python 3.10 ready.

:SKIP_PYTHON_DL

:: ═════════════════════════════════════════════════════════════════════
:: Step 3/5 — Bootstrap pip and create virtual environment
:: ═════════════════════════════════════════════════════════════════════
echo   [*] Step 3/5 — Bootstrapping pip and creating virtual environment...

powershell -NoProfile -Command ^
  "(Get-Content '%PYDIR%\python310._pth') -replace '^#\s*import site','import site' | Set-Content '%PYDIR%\python310._pth'" >> "%LOG_FILE%" 2>&1

if not exist "%PYDIR%\Scripts\pip.exe" (
    echo       Downloading get-pip.py...
    where curl.exe >nul 2>&1
    if %ERRORLEVEL%==0 (
        curl.exe -fSL -o "%PYDIR%\get-pip.py" "https://bootstrap.pypa.io/get-pip.py" >> "%LOG_FILE%" 2>&1
    ) else (
        powershell -NoProfile -Command "Invoke-WebRequest 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYDIR%\get-pip.py'" >> "%LOG_FILE%" 2>&1
    )
    if not exist "%PYDIR%\get-pip.py" (
        echo   [X] ERROR: Could not download get-pip.py.
        pause & exit /b 1
    )
    "%PYDIR%\python.exe" "%PYDIR%\get-pip.py" >> "%LOG_FILE%" 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo   [X] ERROR: pip bootstrap failed.
        pause & exit /b 1
    )
    del /q "%PYDIR%\get-pip.py"
)

echo       Creating virtual environment...
"%PYDIR%\python.exe" -m pip install virtualenv >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [X] ERROR: virtualenv installation failed.
    pause & exit /b 1
)
"%PYDIR%\python.exe" -m virtualenv "%ENVDIR%" >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [X] ERROR: Virtual environment creation failed.
    pause & exit /b 1
)
echo   [OK] Virtual environment ready.

:: ═════════════════════════════════════════════════════════════════════
:: Step 4/5 — Install MEGqc
:: ═════════════════════════════════════════════════════════════════════
echo   [*] Step 4/5 — Installing MEGqc (this may take several minutes)...
call "%ENVDIR%\Scripts\activate.bat"

echo       Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >> "%LOG_FILE%" 2>&1

echo       Installing %PYPI_PKG%...
python -m pip install %PYPI_PKG% >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [X] ERROR: MEGqc installation failed.
    echo       See log: %LOG_FILE%
    pause & exit /b 1
)
echo   [OK] MEGqc installed successfully.

:: ── App icon (PNG → ICO via Pillow — multi-resolution) ──────────────
:: Ask the installed package where it lives — avoids hardcoding the Python
:: version or site-packages path.
set "PKG_PATH="
for /f "usebackq delims=" %%P in (
    `"%ENVDIR%\Scripts\python.exe" -c "import meg_qc, os; print(os.path.dirname(meg_qc.__file__))" 2^>nul`
) do set "PKG_PATH=%%P"

set "ICO_FILE=%BASEDIR%\megqc.ico"
set "HAS_ICON=0"
set "ICON_SRC="

if defined PKG_PATH (
    if exist "%PKG_PATH%\miscellaneous\GUI\assets\macos\AppIcon256.png" (
        set "ICON_SRC=%PKG_PATH%\miscellaneous\GUI\assets\macos\AppIcon256.png"
        echo   [*] Using bundled package icon: assets\macos\AppIcon256.png
    ) else if exist "%PKG_PATH%\miscellaneous\GUI\logo.png" (
        set "ICON_SRC=%PKG_PATH%\miscellaneous\GUI\logo.png"
    )
)

:: Icon creation is delegated to a subroutine to avoid cmd.exe treating the
:: parentheses inside the Python source lines as block delimiters.
if defined ICON_SRC call :MAKE_ICON

:: ═════════════════════════════════════════════════════════════════════
:: Step 5/5 — Create launcher, uninstaller, Desktop and Start Menu
:: ═════════════════════════════════════════════════════════════════════
echo   [*] Step 5/5 — Creating shortcuts...

:: ── Launcher .bat (stored in install dir, not on Desktop directly) ───
(
    echo @echo off
    echo title MEGqc
    echo call "%ENVDIR%\Scripts\activate.bat"
    echo megqc %%*
) > "%BASEDIR%\run_MEGqc.bat"

:: ── Uninstaller .bat (stored in install dir) ─────────────────────────
(
    echo @echo off
    echo chcp 65001 ^>nul 2^>^&1
    echo echo.
    echo echo   MEGqc Uninstaller
    echo echo.
    echo echo   This will completely remove MEGqc from:
    echo echo     %BASEDIR%
    echo echo.
    echo set /p "CONFIRM=  Are you sure? [y/N] "
    echo if /i "%%CONFIRM%%"=="y" ^(
    echo     echo.
    echo     echo   [*] Removing installation...
    echo     rmdir /s /q "%BASEDIR%"
    echo     rd /s /q "%STARTMENU%" 2^>nul
    echo     del /q "%DESKTOP%\MEGqc.lnk" 2^>nul
    echo     del /q "%DESKTOP%\Uninstall MEGqc.lnk" 2^>nul
    echo     echo   [OK] MEGqc has been completely removed.
    echo     echo.
    echo     echo   Closing this window...
    echo     ^(goto^) 2^>nul ^& del /q "%%~f0"
    echo ^) else ^(
    echo     echo   Uninstallation cancelled.
    echo ^)
    echo pause
) > "%BASEDIR%\uninstall_MEGqc.bat"

:: ── Desktop & Start Menu shortcuts via a temp .ps1 file ──────────────
:: powershell -Command "..." with ^-continuation breaks as soon as the
:: opening " is encountered: subsequent ^ are inside a cmd.exe quoted
:: string and are passed literally to PowerShell ("^ is not recognised").
:: Writing a .ps1 file and calling  powershell -File  avoids this entirely.
if not exist "%STARTMENU%" mkdir "%STARTMENU%"
call :MAKE_SHORTCUTS

echo   [OK] Desktop shortcut and Start Menu entry created.

:: ═════════════════════════════════════════════════════════════════════
:: Done!
:: ═════════════════════════════════════════════════════════════════════
echo.
echo   ═══════════════════════════════════════════════════════
echo     OK  MEGqc was successfully installed!
echo   ═══════════════════════════════════════════════════════
echo.
echo   Launch MEGqc:
echo     * Double-click MEGqc on your Desktop
echo     * Or search MEGqc in the Start Menu
echo     A terminal window will open showing MEGqc's progress.
echo.
echo   Uninstall:    Double-click  Uninstall MEGqc  on your Desktop
echo   Log file:     %LOG_FILE%
echo.
echo   Thank you for using MEGqc!
echo.
pause
exit /b 0

:: ═════════════════════════════════════════════════════════════════════
:: Subroutine: MAKE_ICON
::   Writes a small Python script and runs it with Pillow to convert the
::   source PNG into a square-padded, multi-resolution .ico file.
::
::   Must be a subroutine (called with  call :MAKE_ICON) so that the echo
::   lines that contain Python parentheses are NOT inside a cmd.exe
::   parenthesised block — where  )  would be mis-parsed as the block end.
:: ═════════════════════════════════════════════════════════════════════
:MAKE_ICON
echo   [*] Creating app icon (multi-resolution .ico)...
set "ICON_PY=%TEMP%\megqc_icon.py"
:: NOTE: No "if/else" or "!=" in the Python source — the "!" character is
:: consumed by cmd.exe's EnableDelayedExpansion before being written to the file.
:: Centre-on-square is a no-op for already-square images, so no branch needed.
echo from PIL import Image > "%ICON_PY%"
echo src = r"%ICON_SRC%" >> "%ICON_PY%"
echo dst = r"%ICO_FILE%" >> "%ICON_PY%"
echo img = Image.open(src).convert('RGBA') >> "%ICON_PY%"
echo w, h = img.size >> "%ICON_PY%"
echo side = max(w, h) >> "%ICON_PY%"
echo square = Image.new('RGBA', (side, side), (0, 0, 0, 0)) >> "%ICON_PY%"
echo square.paste(img, ((side - w) // 2, (side - h) // 2)) >> "%ICON_PY%"
echo sizes = [(16, 16), (32, 32), (48, 48), (256, 256)] >> "%ICON_PY%"
echo imgs = [square.resize(s, Image.LANCZOS) for s in sizes] >> "%ICON_PY%"
echo imgs[0].save(dst, format='ICO', sizes=sizes, append_images=imgs[1:]) >> "%ICON_PY%"
"%ENVDIR%\Scripts\python.exe" "%ICON_PY%" >> "%LOG_FILE%" 2>&1
del /q "%ICON_PY%"
if exist "%ICO_FILE%" (
    echo   [OK] App icon created.
    set "HAS_ICON=1"
) else (
    echo   [!] Icon creation failed — shortcuts will use the default icon.
)
exit /b 0

:: ═════════════════════════════════════════════════════════════════════
:: Subroutine: MAKE_SHORTCUTS
::   Writes a temporary .ps1 script and runs it with  powershell -File.
::   This is necessary because  powershell -Command "..."  with ^-
::   continuation breaks once the opening " is found: the ^ characters
::   end up inside a cmd.exe quoted string and are passed literally to
::   PowerShell, causing "^ is not recognised" parse errors.
:: ═════════════════════════════════════════════════════════════════════
:MAKE_SHORTCUTS
set "PS1=%TEMP%\megqc_shortcuts.ps1"
echo $sh  = New-Object -ComObject WScript.Shell > "%PS1%"
echo $ico = '%ICO_FILE%' >> "%PS1%"
echo. >> "%PS1%"
echo # MEGqc Desktop shortcut >> "%PS1%"
echo $lnk = $sh.CreateShortcut('%DESKTOP%\MEGqc.lnk') >> "%PS1%"
echo $lnk.TargetPath      = '%BASEDIR%\run_MEGqc.bat' >> "%PS1%"
echo $lnk.WorkingDirectory = '%BASEDIR%' >> "%PS1%"
echo $lnk.Description     = 'Launch MEGqc - Automated MEG and EEG Quality Control' >> "%PS1%"
echo if (Test-Path $ico) { $lnk.IconLocation = "$ico,0" } >> "%PS1%"
echo $lnk.Save() >> "%PS1%"
echo. >> "%PS1%"
echo # Uninstall Desktop shortcut >> "%PS1%"
echo $lnk2 = $sh.CreateShortcut('%DESKTOP%\Uninstall MEGqc.lnk') >> "%PS1%"
echo $lnk2.TargetPath      = '%BASEDIR%\uninstall_MEGqc.bat' >> "%PS1%"
echo $lnk2.WorkingDirectory = '%BASEDIR%' >> "%PS1%"
echo $lnk2.Description     = 'Uninstall MEGqc' >> "%PS1%"
echo $lnk2.Save() >> "%PS1%"
echo. >> "%PS1%"
echo # Start Menu shortcut >> "%PS1%"
echo $lnk3 = $sh.CreateShortcut('%STARTMENU%\MEGqc.lnk') >> "%PS1%"
echo $lnk3.TargetPath      = '%BASEDIR%\run_MEGqc.bat' >> "%PS1%"
echo $lnk3.WorkingDirectory = '%BASEDIR%' >> "%PS1%"
echo $lnk3.Description     = 'Launch MEGqc - Automated MEG and EEG Quality Control' >> "%PS1%"
echo if (Test-Path $ico) { $lnk3.IconLocation = "$ico,0" } >> "%PS1%"
echo $lnk3.Save() >> "%PS1%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" >> "%LOG_FILE%" 2>&1
del /q "%PS1%"
exit /b 0
