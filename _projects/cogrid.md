---
layout: page
title: CoGrid
description: A multi-agent grid-world research framework with NumPy and JAX backends.
img: assets/img/cogrid_logo_clean.png
github: https://github.com/chasemcd/cogrid
github_stars: chasemcd/cogrid
pypi: cogrid
importance: 1
---

**tl;dr** CoGrid is a library for building multi-agent grid-world environments based on the [PettingZoo](https://pettingzoo.farama.org/) API. It supports both NumPy and JAX backends, enabling hardware-accelerated simulation with JIT compilation, alongside WASM compilation through Pyodide for web deployment. 

CoGrid is designed with two primary goals in mind:
- An easy-to-use API that simplifies the development of JAX-based environments.
- A modular, composable architecture that streamlines the design of new environments from a component API. 

For full details, see the online documentation at [cogrid.readthedocs.io](https://cogrid.readthedocs.io).

--- 

#### Basic Overview

**Installation**

<div class="tabs" data-tab-group="install">
  <ul class="tab-nav">
    <li class="tab-link active" data-tab="install-numpy">Basic (NumPy)</li>
    <li class="tab-link" data-tab="install-jax">With JAX</li>
  </ul>
  <div class="tab-content active" id="install-numpy">

```bash
pip install cogrid
```

  </div>
  <div class="tab-content" id="install-jax">

```bash
pip install cogrid[jax]
```

  </div>
</div>

For GPU support, see the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

**Quick Start**

<div class="tabs" data-tab-group="backend">
  <ul class="tab-nav">
    <li class="tab-link active" data-tab="qs-numpy">NumPy</li>
    <li class="tab-link" data-tab="qs-jax">JAX</li>
  </ul>
  <div class="tab-content active" id="qs-numpy">

```python
from cogrid.envs import registry
import cogrid.envs.overcooked

env = registry.make("Overcooked-CrampedRoom-V0")
obs, info = env.reset(seed=42)  # obs: dict[str, ndarray]

while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminateds, truncateds, info = env.step(actions)
```

  </div>
  <div class="tab-content" id="qs-jax">

```python
import jax
from cogrid.envs import registry
import cogrid.envs.overcooked

env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
env.reset(seed=0)  # builds JIT-compiled functions

obs, state, info = env.jax_reset(jax.random.key(0)) 
actions = jax.numpy.array([0, 3], dtype=jax.numpy.int32)
obs, state, rewards, terminateds, truncateds, info = env.jax_step(state, actions)
```

  </div>
</div>

---
#### The Dual Backend

Why do we need both NumPy and JAX backends? For most cases, the modern libraries and training in these (relatively) simple environments should be done in JAX. The training speeded up massive: before JAX, training in environments like Carroll et al.'s Overcooked would take several hours. JaxMARL showed that by re-writing in JAX, training could complete in less than a minute. 

Other than code reability and intuitiveness, JAX wins on almost every dimension. The critical reason that we maintain NumPy as an option is that it can be compiled in WASM with [Pyodide](https://pyodide.org/), whereas JAX cannot. This allows us to deploy CoGrid environments in [Multi-User Gymnasium (MUG)](https://chasemcd.com/projects/mug): a library for conducting human-human and human-AI experiments in the same environments that we train AI in. 

A full speed comparison is shown below, benchmarking against JaxMARL and the original Overcooked-AI. 


---
#### Building Blocks

CoGrid environments are composed from three component types that the engine autowires into array-level code — no manual composition needed.

**GridObject** — An entity in the grid world. Subclass `GridObj` and register it to define visual properties, collision behavior, and state:

```python
@register_object_type("goal")
class Goal(GridObj):
    color = Colors.Green
    char = "g"
    can_overlap = when()  # agents can walk onto this cell
```

**Reward** — Declarative reward conditions. Rather than manually computing rewards, subclass `InteractionReward` and specify when it fires:

```python
class GoalReward(InteractionReward):
    action = None        # no action required
    overlaps = "goal"    # fires when agent stands on a goal cell
```

The step pipeline sums all reward instances each step automatically.

**Feature** — Observable features extracted for agents. Configured declaratively in the environment config:

```python
"features": ["agent_dir", "agent_position", "can_move_direction", "inventory"]
```

For full details on building custom environments, see the [Custom Environment guide](https://cogrid.readthedocs.io/en/latest/custom-environment/).

---
#### Example: Overcooked

CoGrid includes a full Overcooked implementation as a reference environment. Environments are config-driven — define layouts, rewards, and features through a dictionary:

```python
cramped_room_config = {
    "name": "overcooked",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "features": ["agent_dir", "overcooked_inventory", "next_to_counter",
                 "next_to_pot", "object_type_masks", "ordered_pot_features",
                 "dist_to_other_players", "agent_position", "can_move_direction"],
    "rewards": [DeliveryReward(coefficient=1.0, common_reward=True),
                OnionInPotReward(coefficient=0.1, common_reward=False),
                SoupInDishReward(coefficient=0.3, common_reward=False)],
    "grid": {"layout": "overcooked_cramped_room_v0"},
    "max_steps": 1000,
}
```

Pre-registered layouts include Cramped Room, Asymmetric Advantages, Coordination Ring, Forced Coordination, Counter Circuit, and more.

Beyond the standard onion-only recipes from Carroll et al., CoGrid's Overcooked supports multiple ingredients (onions, tomatoes, etc.) and custom recipes. This is configured through an order system — specify which recipes are available, how many ingredients each requires, and how orders are generated during an episode. This makes it easy to study coordination under richer task structures, such as agents needing to divide labor across different dish types or prioritize orders dynamically.

<div style="text-align: center; margin: 2rem 0;">
  <img src="/assets/gif/overcooked_orders_episode.gif" alt="Overcooked episode with multiple recipes and order system" style="max-width: 100%;">
</div>

See the [Overcooked documentation](https://cogrid.readthedocs.io/en/latest/overcooked/) for the full list of layouts and configuration options.

Finally, we benchmark the Overcooked environment against JaxMARL and the original Overcooked-AI to demonstrate the performance gains achieved through JAX compilation. 

<div style="text-align: center; margin: 2rem 0;">
  <img src="/assets/img/OvercookedSPS.png" alt="Overcooked Benchmarking" style="max-width: 100%;">
</div>

Environment throughput in the CoGrid Overcooked environment, comparing to the original Overcooked-AI and JaxMARL~\citep{flair2024jaxmarl} implementations as we increases the number of parallelized environments. The former has a constant rate of roughly 3,400 steps per second, while the latter scales from roughly 4,300 with a single instance to 2.9 million with 1,024 parallel instances. CoGrid's JAX-backed is competitive: scaling from 4,500 steps per second with a single instance to 5.6 million with 1,024 instances, leading to a 1.9x throughput improvement. CoGrid's NumPy backend is by far the slowest at roughly 450 steps per second; however, it offers a pure-Python mode that is entirely absent with JaxMARL. Hardware accelerated execution was run on a single NVIDIA aGeForce RTX 3090.



