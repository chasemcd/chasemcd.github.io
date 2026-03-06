---
layout: page
title: MUG
description: Multi-User Gymnasium — run Gymnasium and PettingZoo environments directly in the browser for user experiments.
img: assets/img/mug_logo.png
github: https://github.com/chasemcd/mug
github_stars: chasemcd/mug
pypi: multi-user-gymnasium
importance: 2
---

**tl;dr** Multi-User Gymnasium (MUG) converts standard Gymnasium and PettingZoo environments into browser-based experiments, running Python-based environments directly in the browser and handling participant networking and AI inference. Core functionality includes:

- Execution of pure-Python Gymnasium or PettingZoo environments directly in the browser via Pyodide with automatic data collection. Server-client architecture is available for non-pure-Python environments.
- Multi-player matchmaking and experiments with generalized rollback netcode (GGPO) to account for network latency.
- Full experiment flow with participant exclusion criteria, completion codes, static pages, surveys, and more.

Full documentation is available at [multi-user-gymnasium.readthedocs.io](https://multi-user-gymnasium.readthedocs.io).

---

## Context

The development of MUG was inspired by a need I faced in graduate school: I needed to be able to run experiments with humans in the exact same environments that I train AI in. The major issue here is that most reinforcement learning environments are written in Python, aren't configured to render in web browsers, and don't have any mechanism to handle multi-player interactions (alongside all the other experiment boilerplate: data collection, experiment pipeline management, etc.).

While there were some existing approaches to do something like this (e.g., [HIPPO-Gym](https://hippogym.irll.net/) and [CrowdPlay](https://github.com/Farama-Foundation/CrowdPlay)), nothing quite satisfied the requirements I had. This was particularly due to the fact that environments nearly always ran on a server, meaning that participants had to have an extremely low amount of latency to have a good experience. The one exception to this was the [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) project, which provided an one-off implementation for their single environment: a reimplementation of Overcooked in JavaScript that ran entirely in the browser, with AI inference running through `TensorFlow.js`.

Excited by this functionality---but frightened at the prospect of reimplementation in JavaScript---I searched for alternative ways to run exact replicas of Python environments in the browser. This led me to [Pyodide](https://pyodide.org/), a project that compiles CPython to WebAssembly, allowing Python code to run directly in the browser. MUG was initially built around Pyodide, esesntially providing a wrapper around the core reinforment learning loop and providing primitive rendering logic.

After much trial-and-error in getting the setup correct and fully built out, MUG ended up being a platform with functionality well beyond its original purpose.

---

#### Gymnasium in the Browser

The key innovation of MUG is the ability to run pure-Python environments in the browser, rather than relying on either a separate implementation or a server-client communication loop. This allows for true parity between the environments that are run for AI training and those that human participants interact with, while side-stepping latency issues that would otherwise be introduced by network communication.

Users simply define how their Gymnasium- or PettingZoo-compatible environment should be initialized and rendered, and MUG handles the rest. Users provide initialization code and implement `render_mode="mug"` to render the environment in the browser:

```python
class MyEnv(pettingzoo.ParallelEnv):
    # Persistent surface automatically applies delta compression
    surface = Surface(width=WIDTH, height=HEIGHT)

    def render(self):
        assert self.render_mode == "mug"

        self.surface.image(
             id=obj.uuid,
                x=x,
                y=y,
                w=width,
                h=height,
        )
        [...]
        return surface

# Initialize an instance of the environment.
# Pyodide will then access `env` to run the environment.
env = MyEnv(render_mode="mug")
```

In JavaScript, we then run the environment using the Pyodide interface. Simplifying
quite a bit, we end up running something like this:

```javascript
class RemoteGame {
    [...]
    async step(actions) {
        const pyActions = this.pyodide.toPy(actions);

        this.pipelineMetrics.stepCallTimestamp = performance.now();

        const result = await this.pyodide.runPythonAsync(`
            obs, rewards, terminateds, truncateds, infos = env.step(${pyActions})
            render_state = env.render()
        `);
        [...]
    }
}
```

Rendering is handled through [Phaser](https://phaser.io/), which we use as our front-end game engine. We translate the logic from `Env.render()` into Phaser commands to render the environment in the browser.

In our example, we've set up a replication of Overcooked (see [the example script](https://github.com/chasemcd/mug/blob/main/mug/examples/cogrid/overcooked_human_ai.py)). We use the sprites from Carroll et al's original[Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) implementation. It ends up looking like this:

<div style="text-align: center; margin: 2rem 0;">
  <img src="/assets/gif/overcooked_in_browser.gif" alt="CoGrid's Overcooked running in the browser with Carroll et al.'s graphics." style="max-width: 100%;">
</div>

---

#### Multi-Player Functionality & Latency Handling

Multi-player matchmaking and experiments with generalized rollback netcode (GGPO) to account for network latency.

<div style="text-align: center; margin: 2rem 0;">
  <img src="/assets/img/rollback_diagram.svg" alt="Rollback netcode diagram" style="max-width: 100%;">
</div>

---

### Experiment Flow

Full experiment flow with participant exclusion criteria, completion codes, static pages, surveys, and more.

[details coming soon!]
