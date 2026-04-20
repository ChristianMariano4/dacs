# DACS
## Toward Context-aware LLM-driven UAVs

> This repository is forked from [TypeFly](https://github.com/typefly/TypeFly) and extended with DACS-specific capabilities.

`DACS` (Drone Adaptive Context-aware System) is an HDI (Human-Drone Interaction) framework for turning natural-language drone control into a continuous, adaptive collaboration.

UAV interaction is moving from manual piloting to high-level intent. In this project, interaction is not treated as a sequence of isolated commands: it is modeled as a perception-action loop where user and system co-adapt over time.

DACS combines:
- LLM-based planning
- multimodal perception
- persistent memory
- safety-aware spatial constraints

The result is a drone system that can remember, adapt, and reason about context instead of repeating stateless behaviors.

## Core Capabilities

- Adaptive interaction over time:
  user feedback is stored and reused as behavioral refinement, not only as immediate one-off correction.
- Dynamic skill evolution:
  reusable high-level skills can emerge from recurring successful behaviors.
- Short-term execution memory:
  traces from the current task guide replanning and avoid repeating failed attempts.
- Context-aware exploration:
  planning reasons on visual context and environment structure, not only on object lists.
- Environment graph:
  explored regions, object associations, and spatial relations are accumulated and reused.
- Flyzone constraints:
  navigation is bounded to user-defined safe operating regions.

## System Overview

At runtime, DACS follows this flow:
1. User sends a natural-language request through the web UI.
2. The drone streams sensory data.
3. The edge stack aggregates perception, memory state, and graph context.
4. LLM planners generate a structured plan.
5. The interpreter executes the plan over robot skills.
6. Execution traces and outcomes feed short-term and long-term adaptation.

Reasoning is modular and distributed:
- sensing and actuation stay close to the robot
- adaptation, memory, and orchestration run on edge software
- heavy LLM inference is invoked via cloud APIs

## Prototype Platform

Current implementation supports:
- DJI Tello for flight execution and RGB stream
- Crazyflie + Lighthouse as localization support (when configured)
- Edge workstation for perception, planning, memory, and orchestration

## Video Evidence

The following videos in `assets/` summarize the key behavioral differences.

### TypeFly baseline on hidden target

TypeFly cannot reliably reason over contextual scene characteristics to find hidden objects, and tends to fall back to reactive scanning behavior.

assets/typefly_hidden_apple.mp4

### DACS context-aware search on hidden target

DACS can reason over context to infer where hidden objects are likely to be, improving exploration decisions when the target is not directly visible.

assets/dacs_hidden_apple.mp4

### DACS graph memory reuse

DACS uses persistent graph memory of regions and previously seen objects. In this example, it exploits prior observations (a bottle seen during earlier exploration) to guide the next action.

assets/dacs_graph.mp4

## Repository Map

- `controller/llm/`: planning, prompting, orchestration
- `controller/minispec_interpreter.py`: `MiniSpec` execution engine
- `controller/context_map/`: environment graph and spatial memory
- `controller/visual_sensing/`: vision skills and object-centric utilities
- `controller/robot_implementations/`: Tello, Crazyflie, virtual wrappers
- `serving/webui/typefly.py`: main web UI entrypoint

## Requirements

### API key
```bash
export OPENAI_API_KEY=...
```

### Vision stack
YOLO service (gRPC) and optional router are available via Docker.

Build YOLO service:
```bash
make SERVICE=yolo build
```

Optional router:
```bash
make SERVICE=router build
```

If remote, set `VISION_SERVICE_IP`.

### Hardware notes
- Default target: DJI Tello
- Alternative: Crazyflie mode
- For Tello + cloud LLM usage, ensure both drone connectivity and internet availability.

## Run

Main UI:
```bash
make typefly
```

Alternative modes:
```bash
make typefly-cf   # Crazyflie mode
make typefly-v    # Virtual robot mode
```

Web UI endpoint:
- `http://localhost:50001`

## Example Requests

- `Find an apple`
- `Can you find something edible?`
- `Can you see a person behind you?`
- `Tell me how many people you can see?`

## Contributing

Contributions are welcome from both research and engineering perspectives.

- Start from [CONTRIBUTING.md](CONTRIBUTING.md) for branch workflow and review expectations.
- Open focused pull requests with clear motivation, reproducible validation steps, and docs updates when behavior changes.
- For perception/planning/UI changes, include visual evidence (screenshots or short videos) in the PR description.
