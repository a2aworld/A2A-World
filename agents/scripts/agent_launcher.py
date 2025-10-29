"""
A2A World Platform - Agent Launcher

Universal agent launcher script that can start any type of agent
with proper configuration and process management.
"""

import asyncio
import argparse
import sys
import signal
import os
import logging
from pathlib import Path
from typing import Optional
import subprocess

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.validation.validation_agent import ValidationAgent
from agents.monitoring.monitor_agent import MonitorAgent
from agents.parsers.kml_parser import KMLParserAgent
from agents.discovery.pattern_discovery import PatternDiscoveryAgent
from agents.core.config import load_agent_config


class AgentLauncher:
    """Universal agent launcher with process management."""
    
    def __init__(self):
        self.agent = None
        self.shutdown_requested = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("agent_launcher")
    
    async def launch_agent(
        self, 
        agent_type: str, 
        agent_id: Optional[str] = None,
        config_file: Optional[str] = None,
        log_level: str = "INFO"
    ) -> bool:
        """Launch an agent of the specified type."""
        
        try:
            self.logger.info(f"Launching {agent_type} agent...")
            
            # Set log level
            logging.getLogger().setLevel(getattr(logging, log_level.upper()))
            
            # Create agent based on type
            if agent_type == "validation":
                self.agent = ValidationAgent(agent_id=agent_id, config_file=config_file)
            elif agent_type == "monitoring":
                self.agent = MonitorAgent(agent_id=agent_id, config_file=config_file)
            elif agent_type == "parser":
                self.agent = KMLParserAgent(agent_id=agent_id, config_file=config_file)
            elif agent_type == "discovery":
                self.agent = PatternDiscoveryAgent(agent_id=agent_id, config_file=config_file)
            else:
                self.logger.error(f"Unknown agent type: {agent_type}")
                return False
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Start the agent
            self.logger.info(f"Starting agent {self.agent.agent_id}")
            await self.agent.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to launch agent: {e}")
            return False
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_requested = True
            if self.agent:
                asyncio.create_task(self.agent.shutdown())
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Windows specific
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)


def main():
    """Main entry point for the agent launcher."""
    
    parser = argparse.ArgumentParser(
        description="A2A World Agent Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a validation agent with default settings
  python agent_launcher.py validation
  
  # Start a monitoring agent with custom ID
  python agent_launcher.py monitoring --agent-id monitor-001
  
  # Start an agent with custom configuration
  python agent_launcher.py parser --config ./config/parser.yaml
  
  # Start with debug logging
  python agent_launcher.py discovery --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "agent_type",
        choices=["validation", "monitoring", "parser", "discovery"],
        help="Type of agent to launch"
    )
    
    parser.add_argument(
        "--agent-id",
        help="Custom agent ID (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon process (background)"
    )
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = AgentLauncher()
    
    if args.daemon:
        # For daemon mode, we'd typically use proper daemonization
        # This is a simplified version
        print(f"Starting {args.agent_type} agent as daemon...")
        print(f"Use 'ps aux | grep {args.agent_type}' to check status")
        print("Use 'kill <PID>' to stop the agent")
    
    try:
        # Launch the agent
        success = asyncio.run(
            launcher.launch_agent(
                agent_type=args.agent_type,
                agent_id=args.agent_id,
                config_file=args.config,
                log_level=args.log_level
            )
        )
        
        if success:
            print(f"{args.agent_type} agent stopped gracefully")
        else:
            print(f"Failed to start {args.agent_type} agent")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Agent launcher failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()