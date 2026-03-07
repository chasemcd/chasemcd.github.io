---
layout: page
title: FootsiesGym
description: A multi-agent RL benchmark environment based on the Footsies fighting game.
img: assets/gif/footsies_trim.gif
github: https://github.com/chasemcd/FootsiesGym
github_stars: chasemcd/FootsiesGym
importance: 3
---

FootsiesGym is a multi-agent reinforcement learning benchmark built on [HiFight's Footsies](https://hifight.github.io/footsies/) fighting game. It serves as a benchmark environment for complex two-player zero-sum games executed in a real-world environment. 

The environment was adopted by Ray's [RLlib](https://docs.ray.io/en/latest/rllib/index.html) as a testing environment and example [here](https://github.com/ray-project/ray/blob/master/rllib/examples/algorithms/ppo/multi_agent_footsies_ppo.py). 


To get started, install the package with pip:

```bash
pip install footsies-gym
```

Then, you can use it as any other PettingZoo environment:

```python
from footsiesgym.footsies import footsies_env

env = footsies_env.FootsiesEnv(
    config={              
        "max_t": 4000,
        "frame_skip": 4,
        "action_delay": 8,
        "guard_break_reward": 0.0,
        "win_reward_scaling_coeff": 10.0, 
        "launch_binaries": True,
        "use_special_charge_action": True,
    },
)
obs, infos = env.reset()

while True:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
    if terminateds["__all__"] or truncateds["__all__"]:
        break
```

---
#### FootsiesGym & Footsies Unity game

FootsiesGym is essentially a wrapper around the Footsies Unity game. We've added gRPC support to the Unity game, then built a Python harness that hooks into the gRPC server to control the game. The game runs in headless mode, allowing for (relatively) fast training without the overhead of rendering. 

The game binaries are managed automatically through environment creation: when you launch an instance of the `FootsiesEnv`, the harness will download (or check for them if already downloaded) the game binaries and launch them on an available port on your machine.

---
#### Fighting Games and Partial Information
While the state of the environment is fully observable, we introduce partial observability into training in Footsies through *action delay*. This is critical: full observability makes moves in fighting games perfectly reactable, resulting in degenerate equilibriums. By introducing a delay between when an action is selected and when it is executed, agents must learn to anticipate what will happen in intermediate states. This is what creates interesting mixed-strategy equilibriums and the rock-paper-scissors dynamics that make fighting games compelling.

---
#### Action Space & Observations

The action space for FootsiesGym is simple, but has one important configuation. The base action space is: No-Op, Back, Forward, Attack, Back-Attack, Forward-Attack. However, there is an optional "special charge" action that can be enabled in the environment configuration (`use_special_charge_action: True`). Special attacks are executed when the agenet holds any attack for 60 frames (15 steps at 4 frame skip). This can be hard to learn, as it not only must be held for the full duration, but executed at precisely the right time---often implying much more than that duration. 

The `use_special_charge_action` adds an additional action to the action space: Special Charge. This toggles on a "holding" option, where every attack is converted to its charged variant. For example: if the agent selects `SPECIAL_CHARGE -> NO-OP -> BACK -> SPECIAL_CHARGE`, the agent will execute `ATTACK -> ATTACK -> BACK_ATTACK -> BACK` (the release of special charge reverts to the previous selected action, which is why the sequence ends with `BACK`).

The observation space does not rely on the game pixels, but rather a featurized representation. The full feauture space definition can be seen via the [encoder implementation](https://github.com/chasemcd/FootsiesGym/blob/main/footsiesgym/footsies/encoder.py). Importantly, when the special charge action is enabled, we add a corresponding feature to the observation space.  


#### Visualizing the Game

We can easily update the configuration to run Footsies in headed mode, which will open a windowing showing the game as its played. Just as in headless mode, the corresponding binaries will be automatically downloaded and launched. 

To run in headed mode to visualize policies, add `"headless": False,` to the configuration dictionary on environment initialization. 