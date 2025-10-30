"""
A2A World Platform - MARL Agents for HDBSCAN Parameter Optimization

Multi-Agent Reinforcement Learning agents using PPO algorithm for optimizing
HDBSCAN clustering parameters through collaborative learning.
"""

import asyncio
import logging
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import os

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    import ray
    from ray import tune
    from ray.tune.registry import register_env
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

from agents.discovery.marl_environment import HDBSCANOptimizationEnv, MultiAgentHDBSCANEnv
from agents.core.base_agent import BaseAgent
from agents.core.messaging import AgentMessage


class MARLCallback(BaseCallback):
    """
    Custom callback for MARL training progress tracking.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.logger = logging.getLogger(__name__)
        self.best_reward = float('-inf')
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Log progress every 1000 steps
        if self.n_calls % 1000 == 0:
            self.logger.info(f"MARL Training Step {self.n_calls}: "
                           f"Mean Reward = {self.locals.get('episode_reward', 0):.3f}")

        return True

    def _on_rollout_end(self) -> None:
        episode_reward = self.locals.get('episode_reward', 0)
        self.episode_rewards.append(episode_reward)

        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.logger.info(f"New best reward: {self.best_reward:.3f}")


class PPOClusteringAgent:
    """
    PPO-based agent for HDBSCAN parameter optimization.

    This agent learns optimal clustering parameters through reinforcement learning,
    using the HDBSCAN optimization environment as the learning domain.
    """

    def __init__(self,
                 agent_id: str,
                 dataset: List[Dict[str, Any]],
                 model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize PPO clustering agent.

        Args:
            agent_id: Unique identifier for this agent
            dataset: Dataset for training
            model_path: Path to pre-trained model (optional)
            config: PPO configuration parameters
        """
        self.agent_id = agent_id
        self.dataset = dataset
        self.model_path = model_path or f"models/marl_agent_{agent_id}"

        # Default PPO configuration
        self.config = config or {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 1
        }

        self.logger = logging.getLogger(__name__)
        self.model = None
        self.environment = None
        self.training_stats = {
            "episodes_completed": 0,
            "total_steps": 0,
            "best_reward": float('-inf'),
            "average_reward": 0.0,
            "convergence_episode": None
        }

        if RL_AVAILABLE:
            self._initialize_model()
        else:
            self.logger.warning("RL libraries not available - MARL agent functionality limited")

    def _initialize_model(self):
        """Initialize PPO model and environment."""

        # Create environment
        self.environment = HDBSCANOptimizationEnv(self.dataset)

        # Wrap environment for monitoring
        self.environment = Monitor(self.environment)

        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: self.environment])

        # Initialize or load PPO model
        if os.path.exists(f"{self.model_path}.zip"):
            self.logger.info(f"Loading pre-trained model from {self.model_path}")
            self.model = PPO.load(self.model_path, env=self.vec_env)
        else:
            self.logger.info("Initializing new PPO model")
            self.model = PPO(
                self.config["policy"],
                self.vec_env,
                learning_rate=self.config["learning_rate"],
                n_steps=self.config["n_steps"],
                batch_size=self.config["batch_size"],
                n_epochs=self.config["n_epochs"],
                gamma=self.config["gamma"],
                gae_lambda=self.config["gae_lambda"],
                clip_range=self.config["clip_range"],
                ent_coef=self.config["ent_coef"],
                vf_coef=self.config["vf_coef"],
                max_grad_norm=self.config["max_grad_norm"],
                verbose=self.config["verbose"]
            )

    async def train(self, total_timesteps: int = 10000,
                   save_interval: int = 5000) -> Dict[str, Any]:
        """
        Train the PPO agent.

        Args:
            total_timesteps: Total training timesteps
            save_interval: Interval for saving model checkpoints

        Returns:
            Training statistics and results
        """

        if not RL_AVAILABLE or self.model is None:
            return {"error": "RL libraries not available"}

        try:
            self.logger.info(f"Starting PPO training for agent {self.agent_id}")

            # Custom callback for progress tracking
            callback = MARLCallback()

            # Train the model
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True
            )

            # Update training statistics
            self.training_stats.update({
                "episodes_completed": len(callback.episode_rewards),
                "total_steps": total_timesteps,
                "best_reward": callback.best_reward,
                "average_reward": np.mean(callback.episode_rewards) if callback.episode_rewards else 0.0
            })

            # Save the trained model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")

            # Get best parameters found during training
            best_params = self.environment.get_best_parameters()

            return {
                "agent_id": self.agent_id,
                "training_stats": self.training_stats,
                "best_parameters": best_params,
                "model_path": self.model_path,
                "training_completed": True
            }

        except Exception as e:
            self.logger.error(f"Training failed for agent {self.agent_id}: {e}")
            return {
                "agent_id": self.agent_id,
                "error": str(e),
                "training_completed": False
            }

    async def predict_parameters(self, observation: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Predict optimal HDBSCAN parameters for new data.

        Args:
            observation: Current environment observation (optional)

        Returns:
            Predicted parameters and confidence
        """

        if not RL_AVAILABLE or self.model is None:
            return {"error": "Model not available"}

        try:
            # Get observation if not provided
            if observation is None:
                obs, _ = self.environment.reset()
                observation = obs

            # Get action from trained model
            action, _ = self.model.predict(observation, deterministic=True)

            # Execute action to get new parameters
            obs, reward, terminated, truncated, info = self.environment.step(action)

            return {
                "predicted_params": info.get("current_params", {}),
                "confidence_score": info.get("best_score", 0.0),
                "action_taken": int(action),
                "reward": float(reward),
                "info": info
            }

        except Exception as e:
            self.logger.error(f"Parameter prediction failed: {e}")
            return {"error": str(e)}

    def save_model(self, path: Optional[str] = None) -> bool:
        """Save the trained model."""
        if self.model is None:
            return False

        save_path = path or self.model_path
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: Optional[str] = None) -> bool:
        """Load a trained model."""
        load_path = path or self.model_path
        try:
            if os.path.exists(f"{load_path}.zip"):
                self.model = PPO.load(load_path, env=self.vec_env)
                return True
            else:
                self.logger.warning(f"Model file not found: {load_path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False


class CollaborativeMARLSystem:
    """
    Multi-Agent Reinforcement Learning system for collaborative HDBSCAN optimization.

    This system coordinates multiple PPO agents that work together to find
    optimal clustering parameters through collaborative learning.
    """

    def __init__(self,
                 dataset: List[Dict[str, Any]],
                 num_agents: int = 3,
                 collaboration_config: Optional[Dict[str, Any]] = None):
        """
        Initialize collaborative MARL system.

        Args:
            dataset: Dataset for training
            num_agents: Number of collaborating agents
            collaboration_config: Configuration for agent collaboration
        """

        self.dataset = dataset
        self.num_agents = num_agents
        self.agents = []

        # Default collaboration configuration
        self.collaboration_config = collaboration_config or {
            "communication_interval": 100,  # Steps between agent communication
            "knowledge_sharing": True,      # Enable parameter sharing
            "consensus_threshold": 0.8,     # Agreement threshold for convergence
            "max_collaboration_rounds": 5   # Maximum collaboration iterations
        }

        self.logger = logging.getLogger(__name__)
        self.training_results = {}
        self.best_global_params = None
        self.convergence_achieved = False

        # Initialize individual agents
        for i in range(num_agents):
            agent_id = f"marl_agent_{i+1}"
            agent = PPOClusteringAgent(agent_id, dataset)
            self.agents.append(agent)

        # Multi-agent environment for coordination
        self.multi_env = MultiAgentHDBSCANEnv(dataset, num_agents)

    async def train_collaborative(self,
                                timesteps_per_agent: int = 5000,
                                collaboration_rounds: int = 3) -> Dict[str, Any]:
        """
        Train agents collaboratively with knowledge sharing.

        Args:
            timesteps_per_agent: Training timesteps per agent per round
            collaboration_rounds: Number of collaboration rounds

        Returns:
            Collaborative training results
        """

        self.logger.info(f"Starting collaborative MARL training with {self.num_agents} agents")

        all_results = {}

        for round_num in range(collaboration_rounds):
            self.logger.info(f"Collaboration Round {round_num + 1}/{collaboration_rounds}")

            # Train all agents concurrently
            training_tasks = []
            for agent in self.agents:
                task = agent.train(timesteps_per_agent)
                training_tasks.append(task)

            round_results = await asyncio.gather(*training_tasks)

            # Store results
            for i, result in enumerate(round_results):
                agent_id = self.agents[i].agent_id
                all_results[f"{agent_id}_round_{round_num+1}"] = result

            # Knowledge sharing phase
            if self.collaboration_config["knowledge_sharing"]:
                await self._share_knowledge(round_results)

            # Check for convergence
            if self._check_convergence(round_results):
                self.logger.info("Convergence achieved!")
                self.convergence_achieved = True
                break

        # Determine global best parameters
        self.best_global_params = self._select_global_best(all_results)

        return {
            "training_results": all_results,
            "global_best_parameters": self.best_global_params,
            "convergence_achieved": self.convergence_achieved,
            "collaboration_rounds_completed": round_num + 1,
            "num_agents": self.num_agents
        }

    async def _share_knowledge(self, round_results: List[Dict[str, Any]]):
        """Share knowledge between agents."""

        # Collect best parameters from each agent
        agent_params = []
        for result in round_results:
            if "best_parameters" in result:
                params = result["best_parameters"]
                agent_params.append({
                    "params": params.get("parameters", {}),
                    "score": params.get("metrics", {}).get("pattern_quality_score", 0.0)
                })

        if not agent_params:
            return

        # Find consensus parameters (weighted average of top performers)
        sorted_params = sorted(agent_params, key=lambda x: x["score"], reverse=True)
        top_params = sorted_params[:min(3, len(sorted_params))]  # Top 3 performers

        consensus_params = {}
        total_weight = sum(p["score"] for p in top_params)

        if total_weight > 0:
            for param_name in ["min_samples", "min_cluster_size", "cluster_selection_epsilon"]:
                weighted_sum = sum(p["params"].get(param_name, 0) * p["score"] for p in top_params)
                consensus_params[param_name] = weighted_sum / total_weight

            # Update agents with consensus knowledge
            for agent in self.agents:
                # This is a simplified knowledge sharing - in practice,
                # you might update agent policies or experience buffers
                self.logger.debug(f"Shared consensus parameters with agent {agent.agent_id}")

    def _check_convergence(self, round_results: List[Dict[str, Any]]) -> bool:
        """Check if agents have converged on similar solutions."""

        scores = []
        for result in round_results:
            if "best_parameters" in result:
                score = result["best_parameters"].get("metrics", {}).get("pattern_quality_score", 0.0)
                scores.append(score)

        if len(scores) < 2:
            return False

        # Check if scores are within convergence threshold
        max_score = max(scores)
        min_score = min(scores)

        if max_score == 0:
            return False

        convergence_ratio = min_score / max_score
        return convergence_ratio >= self.collaboration_config["consensus_threshold"]

    def _select_global_best(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Select the globally best parameters across all agents and rounds."""

        best_score = 0.0
        best_params = None
        best_agent = None

        for key, result in all_results.items():
            if "best_parameters" in result:
                params = result["best_parameters"]
                score = params.get("metrics", {}).get("pattern_quality_score", 0.0)

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_agent = key

        self.logger.info(f"Global best parameters from {best_agent} with score {best_score:.3f}")

        return best_params

    async def predict_optimal_parameters(self,
                                       evaluation_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Predict optimal parameters using the collaborative system.

        Args:
            evaluation_data: Data to evaluate (uses training data if None)

        Returns:
            Optimal parameters with confidence scores
        """

        if evaluation_data is None:
            evaluation_data = self.dataset

        # Get predictions from all agents
        prediction_tasks = []
        for agent in self.agents:
            # Create temporary environment for evaluation
            eval_env = HDBSCANOptimizationEnv(evaluation_data)
            obs, _ = eval_env.reset()

            task = agent.predict_parameters(obs)
            prediction_tasks.append(task)

        predictions = await asyncio.gather(*prediction_tasks)

        # Ensemble predictions (weighted by agent performance)
        if self.best_global_params:
            # Use global best as primary prediction
            ensemble_params = self.best_global_params["parameters"].copy()
            ensemble_score = self.best_global_params["metrics"]["pattern_quality_score"]
        else:
            # Fallback to average of agent predictions
            ensemble_params = {"min_samples": 0, "min_cluster_size": 0, "cluster_selection_epsilon": 0.0}
            total_weight = 0

            for pred in predictions:
                if "predicted_params" in pred:
                    params = pred["predicted_params"]
                    weight = pred.get("confidence_score", 0.0)

                    for param_name in ensemble_params:
                        ensemble_params[param_name] += params.get(param_name, 0) * weight

                    total_weight += weight

            if total_weight > 0:
                for param_name in ensemble_params:
                    ensemble_params[param_name] /= total_weight

            ensemble_score = total_weight / len(predictions) if predictions else 0.0

        return {
            "optimal_parameters": ensemble_params,
            "confidence_score": ensemble_score,
            "individual_predictions": predictions,
            "ensemble_method": "weighted_average" if self.best_global_params else "simple_average",
            "num_agents": self.num_agents
        }


class MARLParameterOptimizer(BaseAgent):
    """
    MARL-based parameter optimizer that integrates with the pattern discovery system.

    This agent manages the MARL system and provides optimized parameters
    for HDBSCAN clustering through NATS-based communication.
    """

    def __init__(self,
                 agent_id: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize MARL parameter optimizer agent.
        """
        super().__init__(
            agent_id=agent_id or "marl_optimizer",
            agent_type="marl_optimizer",
            config=config
        )

        self.collaborative_system = None
        self.parameter_cache = {}  # Cache for optimized parameters
        self.training_status = "not_started"

        self.logger.info(f"MARL Parameter Optimizer {self.agent_id} initialized")

    async def agent_initialize(self) -> None:
        """Initialize MARL optimizer specific components."""
        try:
            # Initialize collaborative MARL system when data becomes available
            self.logger.info("MARL Parameter Optimizer initialization complete")
        except Exception as e:
            self.logger.error(f"Failed to initialize MARL optimizer: {e}")
            raise

    async def setup_subscriptions(self) -> None:
        """Setup NATS subscriptions for MARL optimization requests."""
        if not self.messaging:
            return

        # Subscribe to parameter optimization requests
        opt_sub_id = await self.nats_client.subscribe(
            "agents.marl.optimize_parameters",
            self._handle_parameter_optimization,
            queue_group="marl_optimizers"
        )
        self.subscription_ids.append(opt_sub_id)

        # Subscribe to training requests
        train_sub_id = await self.nats_client.subscribe(
            "agents.marl.train",
            self._handle_training_request,
            queue_group="marl_trainers"
        )
        self.subscription_ids.append(train_sub_id)

    async def _handle_parameter_optimization(self, message: "AgentMessage") -> None:
        """Handle parameter optimization requests."""
        try:
            request_data = message.payload
            dataset = request_data.get("dataset", [])
            dataset_id = request_data.get("dataset_id", "unknown")

            # Check cache first
            cache_key = f"{dataset_id}_{len(dataset)}"
            if cache_key in self.parameter_cache:
                cached_result = self.parameter_cache[cache_key]
                await self._send_optimization_response(message, cached_result)
                return

            # Perform optimization
            if not self.collaborative_system:
                # Initialize system with this dataset
                await self._initialize_system(dataset)

            result = await self.collaborative_system.predict_optimal_parameters(dataset)

            # Cache result
            self.parameter_cache[cache_key] = result

            await self._send_optimization_response(message, result)

        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {e}")
            error_response = {"error": str(e), "dataset_id": message.payload.get("dataset_id")}
            await self._send_optimization_response(message, error_response)

    async def _handle_training_request(self, message: "AgentMessage") -> None:
        """Handle MARL training requests."""
        try:
            request_data = message.payload
            dataset = request_data.get("dataset", [])
            training_config = request_data.get("training_config", {})

            if not dataset:
                await self._send_training_response(message, {"error": "No dataset provided"})
                return

            # Initialize and train system
            await self._initialize_system(dataset)

            self.training_status = "training"
            training_result = await self.collaborative_system.train_collaborative(
                timesteps_per_agent=training_config.get("timesteps_per_agent", 5000),
                collaboration_rounds=training_config.get("collaboration_rounds", 3)
            )

            self.training_status = "completed"
            await self._send_training_response(message, training_result)

        except Exception as e:
            self.logger.error(f"Training request failed: {e}")
            self.training_status = "failed"
            await self._send_training_response(message, {"error": str(e)})

    async def _initialize_system(self, dataset: List[Dict[str, Any]]) -> None:
        """Initialize collaborative MARL system with dataset."""
        if not self.collaborative_system:
            self.collaborative_system = CollaborativeMARLSystem(
                dataset=dataset,
                num_agents=3  # Configurable
            )
            self.logger.info("Initialized collaborative MARL system")

    async def _send_optimization_response(self, original_message: "AgentMessage",
                                        result: Dict[str, Any]) -> None:
        """Send optimization response via NATS."""
        response = self.messaging.create_response(
            sender_id=self.agent_id,
            receiver_id=original_message.sender_id,
            message_type="marl_optimization_response",
            payload=result,
            correlation_id=original_message.correlation_id
        )

        if original_message.reply_to:
            await self.nats_client.publish(original_message.reply_to, response)

    async def _send_training_response(self, original_message: "AgentMessage",
                                    result: Dict[str, Any]) -> None:
        """Send training response via NATS."""
        response = self.messaging.create_response(
            sender_id=self.agent_id,
            receiver_id=original_message.sender_id,
            message_type="marl_training_response",
            payload=result,
            correlation_id=original_message.correlation_id
        )

        if original_message.reply_to:
            await self.nats_client.publish(original_message.reply_to, response)

    async def collect_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect MARL optimizer metrics."""
        return {
            "training_status": self.training_status,
            "cache_size": len(self.parameter_cache),
            "system_initialized": self.collaborative_system is not None,
            "num_agents": self.collaborative_system.num_agents if self.collaborative_system else 0
        }