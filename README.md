# RL League System

A model-agnostic, "serverless" rating and matchmaking system for Reinforcement Learning agents. Logic is strictly file-based, using the local filesystem for state management.

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

## Structure

*   **Registry**: `data/leagues/{branch}/registry/` contains metadata JSON files for every agent.
*   **Snapshots**: `data/leagues/{branch}/snapshots/` contains the actual model weights (`.pt` files).
*   **Matches**: `data/leagues/{branch}/matches.csv` is the single source of truth for match history.

## Workflow

1.  **During Training**: The `League` class handles matchmaking, agent snapshotting, and match reporting.
2.  **Organization**: Use the `branch` parameter to isolate different experiments or agent populations into separate subdirectories under `data/leagues/`.
3.  **Synchronization**: Since the system is file-based, synchronization across machines (if needed) should be handled via external tools (e.g., rsync, Shared Drives, or manual file transfers).
