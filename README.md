d# RL League System

A model-agnostic, "serverless" rating and matchmaking system for Reinforcement Learning agents.

## Setup

1.  **Installation**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Usage**:
    Import the `League` class in your training script to matchmake against historical agents.

    ```python
    from league_manager import League
    league = League(repo_path=".")
    ```

## Collaboration

*   **Registry**: `data/registry/` contains metadata json files for every agent.
*   **Snapshots**: `data/snapshots/` contains the actual model weights.
*   **Matches**: `data/matches.csv` is the single source of truth for match history.

## Workflow

1.  **Before Training**: `git pull` to get the latest agents and match history.
2.  **During Training**: The `League` class handles matchmaking and reporting.
3.  **After Training**: `git add . && git commit -m "Update league" && git push` to share your results.
