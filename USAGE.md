# League Setup & Usage Guide

## 1. Directory Structure
The league is designed to be a standalone directory (e.g., `rl-league`). The internal logic is strictly file-based, meaning all state is stored within the `data/` subdirectory.

## 2. Using the League
To train with the league enabled, update your project's `config.json`. Note that `auto_sync` and `sync_on_add` are legacy parameters and no longer trigger Git commands.

```json
{
    "league": {
        "enabled": true,
        "owner": "your_name",
        "branch": "main",
        "start_episode": 100,
        "prob": 0.5,
        "strategy": "pfsp"
    },
    ...
}
```

## 3. Storage & Collaboration
Since the league is strictly file-based:
- **Agents**: Adding an agent snapshots its weights to `data/leagues/{branch}/snapshots/` and creates a metadata record in `data/leagues/{branch}/registry/`.
- **Match Results**: Stored in `data/leagues/{branch}/matches.csv`.
- **Sharing**: To share the league with others, you must manually transfer or sync the `data/` directory (e.g., via a network drive, rsync, or a separate Git repository if desired, though the system itself will not perform Git operations).

## 4. Sub-Leagues & Experiments
You can use the `branch` parameter to isolate different experiments or parallel training runs.

**How it works:**
- Specify a name in your config (`league.branch`) or via CLI (`--league_branch`).
- The League Manager will create/use a subdirectory in `data/leagues/` specifically for that name.
- Data remains isolated, ensuring that agents from one experiment don't matchmake against agents from another unless they share the same branch name.

**Example:**
```bash
# Start a new experiment in a separate subdirectory
python main.py --league_branch "experiment-alpha"
```

Workflow for merging:
If an experiment yields a strong agent that you want to introduce to the `main` branch, you can manually copy the `.json` registry and `.pt` snapshot files from the experiment branch to the `main` branch directory. The league manager will automatically find them upon reload.

