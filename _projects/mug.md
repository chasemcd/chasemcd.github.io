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

        # Add objects to the surface that we'll render
        self.surface.image(
          id=obj.uuid,
          x=x,
          y=y,
          w=width,
          h=height,
          image_name="my_image",
          frame="image.png",
        )
        [...]

        # Commiting the surface returns a dictionary of changes that occurred since the last commit (delta compression), and tells us what we need to render.
        result = self.surface.commit().to_dict()
        return result

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


Importantly, this only works with environments that are pure Python (or only rely on [packages built in Pyodide](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)). In cases where this isn't the case, we support a fallback regime of a client-server architecture where the environment runs on the server and sends the rendering information to the client. This mode is drastically less robust to bad connections and is not as scalable as participants with poor connection to the server will have poor experiences in fast-paced tasks (e.g., >15 frames per second).

---

#### Multi-Player Functionality & Latency Handling

Another major feature of MUG is the implementation of peer-to-peer networking and generalized rollback netcode. MUG offers configurable matchmaking in multi-player environments that allow us to group specific players together (e.g., first-in first-out matchmaking or filtering so we only allows groups to form that have connections below a latency threshold). However, despite this matchmaking it's still important that we have mechanisms to handle latency and ensure that all clients have the best experience possible. The naive approach to multi-player environments is to simply wait until we have inputs from all clients before advancing the environment---but if any client has a degraded connection or disconnection, it will cause stutter or freezing for all connected clients. 


To remedy this, we draw on the extensive amount of work done in online video games. The specific approach we adopt is derived from [Good Game, Peace Out](https://www.ggpo.net/) (GGPO). GGPO is an implementation of rollback netcode that fixes multi-player latency in an elegant and simple way: run the environment on each client and have it execute inputs as soon as the client selects them. If other clients' inputs haven't arrived, predict them (typically with a fixed default or by repeating the previous action). When the inputs do arrive, we do one of two things: (1) if the predicted input was correct continue execution or (2) if the predicted input was wrong, restore the state before the prediction and quickly restore the environment and replay the game with the correct inputs in the background and restore the environment to the correct state. This approach has been shown to be extremely effective and is used in fast-paced fighting games that rely heavily on precise execution in multi-player games (see the [Wikipedia article](https://en.wikipedia.org/wiki/GGPO) for some examples).

<div style="text-align: center; margin: 2rem 0;">
  <img src="/assets/img/rollback_diagram.svg" alt="Rollback netcode diagram" style="max-width: 100%;">
</div>

The above figure is an illustration of GGPO rollback netcode in a two-player environment. The main timeline shows the simulation state at each tick; each transition \(\mathcal{T}(s_t, a^1_t, a^2_t) \to s_{t+1}\) requires both players' actions. Client 1's actions (\(a^1\)) arrive on time at every tick. Client 2's action at \(t+1\) is delayed, so Client 1 predicts it (\(\hat{a}^2\)), producing a speculative state. When the delayed input arrives as a bundle \(a^2_{t:t+1}\) at \(t+1\) (green), the client rolls back to the last confirmed state \(s_t\) and re-simulates with the correct actions (green timeline), merging the corrected state back into the main timeline. Rollback occurs without rendering re-simulated frames and it takes place between rendered ticks, causing minimal visual disruption.


In MUG, we provide a generalized implementation of GGPO that is automatically activated for multi-player experiments that are running in the browser. To set it up, users must have implemented `Env.get_state()` and `Env.set_state(state)` methods on their environment that allow full state serialization and restoration (and critically, environments must be deterministic given a fixed random seed). 

---

### Experiment Flow

Importantly, MUG supports a full experiment flow rather than just the ability to run experiments with simulation environments. This allows users to define customized pages, messages, randomization, etc. so that a full, proper experiment can be run entirely through MUG. 

The experiment flow primarily relies on an experiment `Stager` and the `Scene` abstraction. A `Stager` dictates the flow of an experiment and `Scenes` represent the individual pages that a participant will see. Simulation environments are run through `GymScenes`, but there are many more options to set up an experiment (e.g., surveys, static pages, completion codes, etc.).

<div style="text-align: center; margin: 2rem 0;">
  <img src="/assets/img/experiment_flow.svg" alt="Experiment flow diagram showing Stager with StartScene, GymScene, SurveyScene, and EndScene" style="max-width: 100%;">
</div>


This setup allows us to have a simple and configurable way to define experiments. We can also introduce other features around `Scenes` via `SceneWrapper`s. For example, adding randomization. Below is an example of how we launched an experiment with Overcooked, randomizing participants to two of five different layouts:

```python
stager = stager.Stager(
    scenes=[
        start_scene,  # welcome page with instructions
        RandomizeOrder(
            scenes=[
                cramped_room_gym_scene,
                counter_circuit_gym_scene,
                forced_coordination_gym_scene,
                asymmetric_advantages_gym_scene,
                coordination_ring_gym_scene,
            ],
            keep_n=2,  # Randomly select two of the five layouts
        ),
        feedback_scene,  # a survey
        end_scene,  # completion code for MTurk
    ]
)


if __name__ == "__main__":
    experiment_config = (
        ExperimentConfig()
        .experiment(
          stager=stager, 
          experiment_id="overcooked_test"
        )
        .hosting(port=5702, host="0.0.0.0")
    )

    app.run(experiment_config)
```


Each scene has a number of configurations and ways to customize them---any custom HTML can be inserted for any number of advanced features. 


For all documentation and additional examples, see the [MUG documentation](https://multi-user-gymnasium.readthedocs.io/). 