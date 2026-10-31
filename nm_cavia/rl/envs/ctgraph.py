# -*- coding: utf-8 -*-

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_CTgraph.CTgraph_env import CTgraphEnv
from gym_CTgraph.CTgraph_conf import CTgraph_conf
from gym_CTgraph.CTgraph_images import CTgraph_images

from itertools import product


def convert_old_config_to_new(conf):
    """
    Convert the configuration format used by the original
    neuromodulation paper to the current CT-graph format.
    """

    graph = conf["graph_shape"]

    if "depth" in graph:
        graph["d"] = graph.pop("depth")

    if "branching_factor" in graph:
        graph["b"] = graph.pop("branching_factor")

    if "delay_prob" in graph:
        graph["p"] = graph.pop("delay_prob")


    reward = conf["reward"]

    if "high_reward_value" in reward:
        reward["high_r"] = reward.pop("high_reward_value")

    if "crash_reward_value" in reward:
        reward["fail_r"] = reward.pop("crash_reward_value")

    if "reward_std" in reward:
        reward["std_r"] = reward.pop("reward_std")


    observations = conf["observations"]

    if "MDP_decisions" in observations:
        observations["MDP_D"] = observations.pop("MDP_decisions")

    if "MDP_delays" in observations:
        observations["MDP_W"] = observations.pop("MDP_delays")

    if "delay_states" in observations:
        observations["W_IDs"] = observations.pop("delay_states")

    if "decision_points" in observations:
        observations["D_IDs"] = observations.pop("decision_points")


    image_data = conf["image_dataset"]

    if "noise_on_images_on_read" in image_data:
        image_data["noise_on_read"] = image_data.pop(
            "noise_on_images_on_read"
        )

    if "small_rotation_on_read" in image_data:
        image_data["rotation_on_read"] = image_data.pop(
            "small_rotation_on_read"
        )

    return conf


class CTgraphEnvGoalChange(CTgraphEnv):

    def __init__(self):
        # IMPORTANT:
        # The modern CTgraphEnv requires configuration during __init__.
        # CAVIA creates the environment first and calls set_config later.
        # Therefore we deliberately postpone CTgraphEnv.__init__.
        pass


    def set_config(self, env_config_path):

        configuration = CTgraph_conf(env_config_path)
        conf_data = configuration.getParameters()

        # Convert original paper configuration -> modern CT-graph
        conf_data = convert_old_config_to_new(conf_data)

        image_dataset = CTgraph_images(conf_data)

        # Initialise modern CT-graph
        CTgraphEnv.__init__(
            self,
            conf_data,
            image_dataset
        )

        depth = conf_data["graph_shape"]["d"]
        branch = conf_data["graph_shape"]["b"]

        self.all_tasks = list(
            product(
                list(range(branch)),
                repeat=depth
            )
        )

        self.all_tasks = [
            {"goal": g}
            for g in self.all_tasks
        ]

        self._goal = self.get_high_reward_path()
        self._task = {"goal": self._goal}

        # Get one observation so we know its flattened size
        obs = self._process_observation(
            self.get_observation_images()
        )

        self._state = obs

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._state.shape,
            dtype=np.float32
        )


    def _process_observation(self, obs):

        obs = np.asarray(obs)

        # Original paper normalisation
        if obs.dtype == np.uint8 or np.max(obs) > 1:
            obs = obs.astype(np.float32) / 255.0
        else:
            obs = obs.astype(np.float32)

        return obs.ravel()


    def seed(self, seed=None):
        # Keep behaviour close to the original implementation.
        # CT-graph already receives a seed from its configuration.
        return None


    def sample_tasks(self, num_tasks):

        idxs = np.random.randint(
            low=0,
            high=len(self.all_tasks),
            size=num_tasks
        )

        return [
            self.all_tasks[idx]
            for idx in idxs
        ]


    def reset_task(self, task):

        self._task = task
        self._goal = task["goal"]

        self.set_high_reward_path(
            np.asarray(self._goal)
        )


    def reset(self, env=True, seed=None, options=None):

        # Modern Gymnasium:
        # obs, info = reset()

        obs, info = super().reset(
            seed=seed,
            options=options
        )

        obs = self._process_observation(obs)

        self._state = obs

        # CAVIA expects ONLY observation
        return self._state


    def step(self, action):

        # Modern Gymnasium:
        # obs, reward, terminated, truncated, info

        obs, reward, terminated, truncated, info = \
            super().step(action)

        obs = self._process_observation(obs)

        done = terminated or truncated

        # CAVIA expects the old four-value API
        return (
            obs,
            reward,
            done,
            {"info": info}
        )