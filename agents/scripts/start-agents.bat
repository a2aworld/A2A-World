@echo off
REM A2A World Platform - Start Agents Script (Windows)
REM Starts the multi-agent system with all required components

setlocal enabledelayedexpansion

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\..\"
set "LOG_DIR=%PROJECT_ROOT%logs"
set "PID_DIR=%PROJECT_ROOT%pids"

REM Create directories
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%PID_DIR%" mkdir "%PID_DIR%"

REM Configuration
set "LOG_LEVEL=INFO"
set "CONFIG_DIR=%PROJECT_ROOT%config"

REM Default agents
set "DEFAULT_AGENTS=monitoring validation parser discovery"

REM Colors (if supported)
set "RED="
set "GREEN="
set "YELLOW="
set "BLUE="
set "NC="

:print_status
echo [INFO] %~1
exit /b

:print_success
echo [SUCCESS] %~1
exit /b

:print_warning
echo [WARNING] %~1
exit /b

:print_error
echo [ERROR] %~1
exit /b

:check_dependencies
call :print_status "Checking dependencies..."

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Python is required but not installed"
    exit /b 1
)

REM Check NATS (simplified check)
curl -s http://localhost:4222 >nul 2>&1
if errorlevel 1 (
    call :print_warning "NATS server not responding at localhost:4222"
    call :print_status "Please ensure NATS is running: docker-compose up nats"
)

REM Check Consul (simplified check)  
curl -s http://localhost:8500 >nul 2>&1
if errorlevel 1 (
    call :print_warning "Consul server not responding at localhost:8500"
    call :print_status "Please ensure Consul is running: docker-compose up consul"
)

call :print_success "Dependencies checked"
exit /b

:start_agent
set "agent_type=%~1"
set "agent_id=%~2"
set "config_file=%~3"

call :print_status "Starting %agent_type% agent (ID: %agent_id%)"

set "log_file=%LOG_DIR%\%agent_id%.log"
set "pid_file=%PID_DIR%\%agent_id%.pid"
set "cmd_args="

REM Add configuration file if provided
if defined config_file (
    if exist "%config_file%" (
        set "cmd_args=--config %config_file%"
    )
)

REM Start agent in background (Windows doesn't have nohup, use start)
cd /d "%PROJECT_ROOT%"
start /b "Agent-%agent_id%" cmd /c "python -m agents.scripts.agent_launcher %agent_type% --agent-id %agent_id% --log-level %LOG_LEVEL% %cmd_args% >> "%log_file%" 2>&1"

REM Get the PID would be complex in batch, so we'll use a simplified approach
echo Agent started > "%pid_file%"

call :print_success "%agent_type% agent started"
echo   Log file: %log_file%
echo   PID file: %pid_file%

exit /b

:list_agents
call :print_status "Listing agent status:"

set "found_agents=0"
for %%f in ("%PID_DIR%\*.pid") do (
    set "agent_id=%%~nf"
    echo   !agent_id! - RUNNING (simplified status)
    set /a found_agents+=1
)

if %found_agents%==0 (
    call :print_warning "No agent PID files found"
) else (
    call :print_success "Found %found_agents% agent PID files"
)

exit /b

:show_usage
echo Usage: %~nx0 [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   start [AGENT_TYPE] [AGENT_ID]  Start specific agent or all agents
echo   stop                           Stop all agents (simplified)
echo   status                         Show status of agents
echo   clean                          Clean up PID and log files
echo   help                           Show this help message
echo.
echo Agent Types:
echo   monitoring                     System monitoring agent
echo   validation                     Pattern validation agent  
echo   parser                         KML/GeoJSON parser agent
echo   discovery                      Pattern discovery agent
echo.
echo Examples:
echo   %~nx0 start                    # Start all default agents
echo   %~nx0 start monitoring         # Start monitoring agent
echo   %~nx0 status                   # Show agent status
echo   %~nx0 clean                    # Clean up files
echo.
echo Note: This Windows script provides basic functionality.
echo For full features, use the Linux/Mac version or the Python CLI tool.
exit /b

REM Parse command line arguments
set "COMMAND=%~1"
set "AGENT_TYPE=%~2"
set "AGENT_ID=%~3"

if "%COMMAND%"=="" set "COMMAND=help"

call :print_status "A2A World Agent Management Script (Windows)"
call :print_status "============================================="

if "%COMMAND%"=="start" (
    call :check_dependencies
    
    if defined AGENT_TYPE (
        REM Start specific agent
        if not defined AGENT_ID (
            REM Generate timestamp-based ID
            for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list ^| findstr "="') do set datetime=%%I
            set "AGENT_ID=%AGENT_TYPE%-!datetime:~0,8!!datetime:~8,6!"
        )
        
        set "config_file="
        if exist "%CONFIG_DIR%\%AGENT_TYPE%.yaml" (
            set "config_file=%CONFIG_DIR%\%AGENT_TYPE%.yaml"
        )
        
        call :start_agent "%AGENT_TYPE%" "!AGENT_ID!" "!config_file!"
    ) else (
        REM Start all default agents
        call :print_status "Starting all default agents..."
        for %%a in (%DEFAULT_AGENTS%) do (
            REM Generate timestamp-based ID
            for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list ^| findstr "="') do set datetime=%%I
            set "agent_id=%%a-!datetime:~0,8!!datetime:~8,6!"
            
            set "config_file="
            if exist "%CONFIG_DIR%\%%a.yaml" (
                set "config_file=%CONFIG_DIR%\%%a.yaml"
            )
            
            call :start_agent "%%a" "!agent_id!" "!config_file!"
            timeout /t 2 >nul
        )
        
        call :print_success "All agents started"
        call :print_status "Use '%~nx0 status' to check agent status"
    )
    
) else if "%COMMAND%"=="stop" (
    call :print_status "Stopping agents..."
    REM Simplified stop - just remove PID files
    del /q "%PID_DIR%\*.pid" >nul 2>&1
    call :print_success "Agent tracking cleared (processes may still be running)"
    call :print_status "Use Task Manager to manually stop Python processes if needed"
    
) else if "%COMMAND%"=="status" (
    call :list_agents
    
) else if "%COMMAND%"=="clean" (
    call :print_status "Cleaning up PID and log files..."
    
    REM Remove PID files
    del /q "%PID_DIR%\*.pid" >nul 2>&1
    
    REM Remove old log files (older than 7 days)
    forfiles /p "%LOG_DIR%" /s /m "*.log" /d -7 /c "cmd /c del @path" >nul 2>&1
    
    call :print_success "Cleanup completed"
    
) else if "%COMMAND%"=="help" (
    call :show_usage
    
) else (
    call :print_error "Unknown command: %COMMAND%"
    call :show_usage
    exit /b 1
)

call :print_status "Script completed"