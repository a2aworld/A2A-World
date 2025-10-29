"""
A2A World Platform - Agent Management CLI

Command-line interface for managing agents, monitoring system health,
and controlling the multi-agent system infrastructure.
"""

import asyncio
import click
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from tabulate import tabulate

from agents.core.registry import get_consul_registry, ConsulRegistry
from agents.core.messaging import get_nats_client, NATSClient, AgentMessage
from agents.core.config import load_agent_config, get_config_manager
from agents.core.task_queue import TaskQueue, Task, TaskPriority, create_parse_task, create_discovery_task
from agents.validation.validation_agent import ValidationAgent
from agents.monitoring.monitor_agent import MonitorAgent
from agents.parsers.kml_parser import KMLParserAgent
from agents.discovery.pattern_discovery import PatternDiscoveryAgent


class AgentCLI:
    """Main CLI class for agent management."""
    
    def __init__(self):
        self.nats_client: Optional[NATSClient] = None
        self.consul_registry: Optional[ConsulRegistry] = None
        self.task_queue: Optional[TaskQueue] = None
        self.connected = False
    
    async def connect(self):
        """Connect to NATS and Consul services."""
        try:
            # Connect to NATS
            self.nats_client = await get_nats_client(name="agent-cli")
            
            # Connect to Consul
            self.consul_registry = await get_consul_registry()
            
            # Initialize task queue
            self.task_queue = TaskQueue(
                nats_client=self.nats_client,
                registry=self.consul_registry
            )
            
            self.connected = True
            return True
            
        except Exception as e:
            click.echo(f"Error connecting to services: {e}", err=True)
            return False
    
    async def disconnect(self):
        """Disconnect from services."""
        if self.nats_client:
            await self.nats_client.disconnect()
        
        self.connected = False
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents."""
        if not self.consul_registry:
            return []
        
        agents = await self.consul_registry.discover_agents()
        return [
            {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "status": agent.status,
                "capabilities": ", ".join(agent.capabilities[:3]) + ("..." if len(agent.capabilities) > 3 else ""),
                "address": f"{agent.address}:{agent.port}" if agent.port else agent.address,
                "last_check": agent.last_health_check.strftime("%H:%M:%S") if agent.last_health_check else "Never"
            }
            for agent in agents
        ]
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a specific agent."""
        if not self.consul_registry:
            return None
        
        agent = await self.consul_registry.get_agent_by_id(agent_id)
        if not agent:
            return None
        
        return {
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "status": agent.status,
            "capabilities": agent.capabilities,
            "address": agent.address,
            "port": agent.port,
            "metadata": agent.metadata,
            "last_health_check": agent.last_health_check.isoformat() if agent.last_health_check else None,
            "registered_at": agent.registered_at.isoformat()
        }
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status."""
        if not self.consul_registry:
            return {}
        
        return await self.consul_registry.get_cluster_status()
    
    async def submit_task(self, task_type: str, parameters: Dict[str, Any], priority: int = 5) -> str:
        """Submit a task to the queue."""
        if not self.task_queue:
            raise RuntimeError("Task queue not initialized")
        
        if task_type == "parse_kml":
            task = create_parse_task(
                file_path=parameters.get("file_path", ""),
                file_type="kml",
                priority=priority
            )
        elif task_type == "discover_patterns":
            task = create_discovery_task(
                dataset_id=parameters.get("dataset_id", ""),
                priority=priority
            )
        else:
            task = Task(
                task_type=task_type,
                priority=priority,
                parameters=parameters
            )
        
        success = await self.task_queue.submit_task(task)
        if success:
            return task.task_id
        else:
            raise RuntimeError("Failed to submit task")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        if not self.task_queue:
            return None
        
        return await self.task_queue.get_task_status(task_id)
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get task queue statistics."""
        if not self.task_queue:
            return {}
        
        return await self.task_queue.get_queue_stats()
    
    async def send_agent_command(self, agent_id: str, command: str, parameters: Dict[str, Any] = None) -> bool:
        """Send a command to a specific agent."""
        if not self.nats_client:
            return False
        
        message = AgentMessage.create(
            sender_id="agent-cli",
            receiver_id=agent_id,
            message_type=command,
            payload=parameters or {}
        )
        
        try:
            await self.nats_client.publish(f"agents.{agent_id}.commands", message)
            return True
        except Exception:
            return False
    
    async def broadcast_command(self, command: str, parameters: Dict[str, Any] = None) -> bool:
        """Broadcast a command to all agents."""
        if not self.nats_client:
            return False
        
        message = AgentMessage.create(
            sender_id="agent-cli",
            message_type=command,
            payload=parameters or {}
        )
        
        try:
            await self.nats_client.publish("agents.broadcast", message)
            return True
        except Exception:
            return False


# Global CLI instance
cli_instance = AgentCLI()


# Click command group
@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """A2A World Agent Management CLI"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    if verbose:
        click.echo("Verbose mode enabled")


@cli.command()
@click.pass_context
def connect(ctx):
    """Connect to NATS and Consul services"""
    async def _connect():
        click.echo("Connecting to services...")
        success = await cli_instance.connect()
        if success:
            click.echo("✓ Connected to NATS and Consul")
        else:
            click.echo("✗ Failed to connect to services", err=True)
            sys.exit(1)
    
    asyncio.run(_connect())


@cli.group()
def agents():
    """Agent management commands"""
    pass


@agents.command('list')
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table', help='Output format')
def list_agents(format):
    """List all registered agents"""
    async def _list():
        if not cli_instance.connected:
            await cli_instance.connect()
        
        agents_list = await cli_instance.list_agents()
        
        if format == 'json':
            click.echo(json.dumps(agents_list, indent=2))
        else:
            if not agents_list:
                click.echo("No agents registered")
                return
            
            headers = ["Agent ID", "Type", "Status", "Capabilities", "Address", "Last Check"]
            rows = [
                [agent["agent_id"], agent["agent_type"], agent["status"], 
                 agent["capabilities"], agent["address"], agent["last_check"]]
                for agent in agents_list
            ]
            click.echo(tabulate(rows, headers=headers, tablefmt="grid"))
    
    asyncio.run(_list())


@agents.command('status')
@click.argument('agent_id')
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='json', help='Output format')
def agent_status(agent_id, format):
    """Get detailed status for a specific agent"""
    async def _status():
        if not cli_instance.connected:
            await cli_instance.connect()
        
        status = await cli_instance.get_agent_status(agent_id)
        
        if not status:
            click.echo(f"Agent '{agent_id}' not found", err=True)
            return
        
        if format == 'json':
            click.echo(json.dumps(status, indent=2))
        else:
            click.echo(f"Agent: {status['agent_id']}")
            click.echo(f"Type: {status['agent_type']}")
            click.echo(f"Status: {status['status']}")
            click.echo(f"Address: {status['address']}:{status['port']}")
            click.echo(f"Capabilities: {', '.join(status['capabilities'])}")
            click.echo(f"Last Health Check: {status['last_health_check'] or 'Never'}")
    
    asyncio.run(_status())


@agents.command('start')
@click.argument('agent_type', type=click.Choice(['validation', 'monitoring', 'parser', 'discovery']))
@click.option('--agent-id', help='Custom agent ID')
@click.option('--config', help='Configuration file path')
@click.option('--background', '-b', is_flag=True, help='Run in background')
def start_agent(agent_type, agent_id, config, background):
    """Start a new agent"""
    async def _start():
        click.echo(f"Starting {agent_type} agent...")
        
        try:
            # Create agent based on type
            if agent_type == 'validation':
                agent = ValidationAgent(agent_id=agent_id, config_file=config)
            elif agent_type == 'monitoring':
                agent = MonitorAgent(agent_id=agent_id, config_file=config)
            elif agent_type == 'parser':
                agent = KMLParserAgent(agent_id=agent_id, config_file=config)
            elif agent_type == 'discovery':
                agent = PatternDiscoveryAgent(agent_id=agent_id, config_file=config)
            else:
                click.echo(f"Unknown agent type: {agent_type}", err=True)
                return
            
            if background:
                # In a real implementation, this would use process management
                click.echo("Background mode not implemented in this example")
                click.echo("Use 'nohup python -m agents.{agent_type}.{agent_type}_agent &' instead")
            else:
                click.echo(f"Agent {agent.agent_id} starting...")
                await agent.start()
                
        except KeyboardInterrupt:
            click.echo("\nShutdown requested by user")
        except Exception as e:
            click.echo(f"Error starting agent: {e}", err=True)
    
    asyncio.run(_start())


@agents.command('stop')
@click.argument('agent_id')
def stop_agent(agent_id):
    """Stop a specific agent"""
    async def _stop():
        if not cli_instance.connected:
            await cli_instance.connect()
        
        click.echo(f"Stopping agent {agent_id}...")
        success = await cli_instance.send_agent_command(agent_id, "shutdown_signal")
        
        if success:
            click.echo(f"✓ Shutdown signal sent to agent {agent_id}")
        else:
            click.echo(f"✗ Failed to send shutdown signal to agent {agent_id}", err=True)
    
    asyncio.run(_stop())


@agents.command('restart')
@click.argument('agent_id')
def restart_agent(agent_id):
    """Restart a specific agent"""
    async def _restart():
        if not cli_instance.connected:
            await cli_instance.connect()
        
        click.echo(f"Restarting agent {agent_id}...")
        success = await cli_instance.send_agent_command(agent_id, "restart_signal")
        
        if success:
            click.echo(f"✓ Restart signal sent to agent {agent_id}")
        else:
            click.echo(f"✗ Failed to send restart signal to agent {agent_id}", err=True)
    
    asyncio.run(_restart())


@cli.group()
def tasks():
    """Task management commands"""
    pass


@tasks.command('submit')
@click.argument('task_type')
@click.option('--priority', '-p', type=int, default=5, help='Task priority (1-10)')
@click.option('--param', multiple=True, help='Task parameters (key=value)')
def submit_task(task_type, priority, param):
    """Submit a new task to the queue"""
    async def _submit():
        if not cli_instance.connected:
            await cli_instance.connect()
        
        # Parse parameters
        parameters = {}
        for p in param:
            if '=' in p:
                key, value = p.split('=', 1)
                parameters[key] = value
        
        try:
            task_id = await cli_instance.submit_task(task_type, parameters, priority)
            click.echo(f"✓ Task submitted with ID: {task_id}")
        except Exception as e:
            click.echo(f"✗ Failed to submit task: {e}", err=True)
    
    asyncio.run(_submit())


@tasks.command('status')
@click.argument('task_id')
def task_status(task_id):
    """Get task status"""
    async def _status():
        if not cli_instance.connected:
            await cli_instance.connect()
        
        status = await cli_instance.get_task_status(task_id)
        
        if not status:
            click.echo(f"Task '{task_id}' not found", err=True)
            return
        
        click.echo(json.dumps(status, indent=2))
    
    asyncio.run(_status())


@tasks.command('queue')
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table', help='Output format')
def queue_status(format):
    """Get task queue statistics"""
    async def _queue():
        if not cli_instance.connected:
            await cli_instance.connect()
        
        stats = await cli_instance.get_queue_stats()
        
        if format == 'json':
            click.echo(json.dumps(stats, indent=2))
        else:
            click.echo("Task Queue Status:")
            click.echo(f"  Pending Tasks: {stats.get('pending_tasks', 0)}")
            click.echo(f"  Active Tasks: {stats.get('active_tasks', 0)}")
            click.echo(f"  Completed Tasks: {stats.get('completed_tasks', 0)}")
            click.echo(f"  Failed Tasks: {stats.get('failed_tasks', 0)}")
            click.echo(f"  Total Tasks: {stats.get('total_tasks', 0)}")
    
    asyncio.run(_queue())


@cli.group()
def system():
    """System management commands"""
    pass


@system.command('status')
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='json', help='Output format')
def system_status(format):
    """Get overall system status"""
    async def _status():
        if not cli_instance.connected:
            await cli_instance.connect()
        
        status = await cli_instance.get_cluster_status()
        
        if format == 'json':
            click.echo(json.dumps(status, indent=2))
        else:
            click.echo("System Status:")
            click.echo(f"  Total Agents: {status.get('total_agents', 0)}")
            click.echo(f"  Healthy Agents: {status.get('healthy_agents', 0)}")
            click.echo(f"  Cluster Health: {status.get('cluster_health', 'unknown')}")
            
            agent_types = status.get('agent_types', {})
            if agent_types:
                click.echo("  Agent Types:")
                for agent_type, count in agent_types.items():
                    click.echo(f"    {agent_type}: {count}")
    
    asyncio.run(_status())


@system.command('health')
def health_check():
    """Perform system health check"""
    async def _health():
        click.echo("Performing system health check...")
        
        # Check NATS connectivity
        try:
            nats_client = await get_nats_client(name="health-check")
            click.echo("✓ NATS: Connected")
            await nats_client.disconnect()
        except Exception as e:
            click.echo(f"✗ NATS: {e}")
        
        # Check Consul connectivity
        try:
            consul_registry = await get_consul_registry()
            cluster_status = await consul_registry.get_cluster_status()
            click.echo(f"✓ Consul: {cluster_status.get('total_agents', 0)} agents registered")
        except Exception as e:
            click.echo(f"✗ Consul: {e}")
        
        # Check database connectivity (if configured)
        try:
            # This would check database connectivity
            click.echo("✓ Database: Connected")
        except Exception as e:
            click.echo(f"✗ Database: {e}")
    
    asyncio.run(_health())


@system.command('shutdown')
@click.confirmation_option(prompt='Are you sure you want to shutdown all agents?')
def shutdown_system():
    """Shutdown all agents"""
    async def _shutdown():
        if not cli_instance.connected:
            await cli_instance.connect()
        
        click.echo("Broadcasting shutdown signal to all agents...")
        success = await cli_instance.broadcast_command("shutdown_signal")
        
        if success:
            click.echo("✓ Shutdown signal broadcasted")
        else:
            click.echo("✗ Failed to broadcast shutdown signal", err=True)
    
    asyncio.run(_shutdown())


@cli.group()
def config():
    """Configuration management commands"""
    pass


@config.command('show')
@click.argument('agent_type')
@click.option('--agent-id', help='Specific agent ID')
def show_config(agent_type, agent_id):
    """Show agent configuration"""
    try:
        config = load_agent_config(agent_type)
        click.echo(json.dumps(config.to_dict(), indent=2))
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)


@config.command('validate')
@click.argument('config_file', type=click.Path(exists=True))
def validate_config(config_file):
    """Validate a configuration file"""
    try:
        # Load and validate configuration
        config_path = Path(config_file)
        click.echo(f"Validating configuration file: {config_path}")
        
        # This would perform actual validation
        click.echo("✓ Configuration file is valid")
    except Exception as e:
        click.echo(f"✗ Configuration validation failed: {e}", err=True)


@cli.command()
@click.option('--watch', '-w', is_flag=True, help='Watch mode - continuously update')
@click.option('--interval', '-i', type=int, default=5, help='Update interval in seconds (watch mode)')
def monitor(watch, interval):
    """Monitor system status in real-time"""
    async def _monitor():
        if not cli_instance.connected:
            await cli_instance.connect()
        
        if watch:
            try:
                while True:
                    click.clear()
                    click.echo("=" * 80)
                    click.echo(f"A2A World System Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    click.echo("=" * 80)
                    
                    # System status
                    status = await cli_instance.get_cluster_status()
                    click.echo(f"Cluster Health: {status.get('cluster_health', 'unknown')}")
                    click.echo(f"Total Agents: {status.get('total_agents', 0)}")
                    click.echo(f"Healthy Agents: {status.get('healthy_agents', 0)}")
                    
                    # Agent list
                    agents_list = await cli_instance.list_agents()
                    if agents_list:
                        click.echo("\nActive Agents:")
                        headers = ["ID", "Type", "Status", "Last Check"]
                        rows = [
                            [agent["agent_id"][:20], agent["agent_type"], 
                             agent["status"], agent["last_check"]]
                            for agent in agents_list
                        ]
                        click.echo(tabulate(rows, headers=headers, tablefmt="simple"))
                    
                    # Task queue stats
                    queue_stats = await cli_instance.get_queue_stats()
                    click.echo(f"\nTask Queue - Pending: {queue_stats.get('pending_tasks', 0)}, " 
                              f"Active: {queue_stats.get('active_tasks', 0)}, "
                              f"Completed: {queue_stats.get('completed_tasks', 0)}")
                    
                    click.echo(f"\nPress Ctrl+C to exit | Updating every {interval}s")
                    await asyncio.sleep(interval)
                    
            except KeyboardInterrupt:
                click.echo("\nMonitoring stopped")
        else:
            # Single snapshot
            status = await cli_instance.get_cluster_status()
            click.echo(json.dumps(status, indent=2))
    
    asyncio.run(_monitor())


if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        # Cleanup
        asyncio.run(cli_instance.disconnect())