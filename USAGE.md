# League Setup & Usage Guide

## 1. Initializing the Git Repository
The league is designed to be a **separate** Git repository. This allows you to share it with your team without bloating your main project.

Run these commands in your terminal:

```bash
# 1. Go to the league directory
cd ../rl-league

# 2. Initialize Git
git init

# 3. Add files
git add .

# 4. Commit
git commit -m "Initial commit for RL League"

# 5. Add your remote (GitHub/GitLab)
# Replace URL with your actual repository URL
# git remote add origin https://github.com/your-username/rl-league.git

# 6. Push
# git push -u origin main
```

## 2. Using the League
To train with the league enabled, you need to update your `config.json` in your main project:

```json
{
    "league": {
        "enabled": true,
        "owner": "your_name",
        "branch": "main",
        "auto_sync": true,
        "sync_on_add": true,
        "sync_on_match": false,
        "start_episode": 100,
        "prob": 0.5,
        "strategy": "pfsp"
    },
    ...
}
```

## 3. Collaboration Workflow
The "serverless" design means you must sync manually:

1.  **Before Training**:
    The league manager (`auto_sync=true`) will automatically pull latest changes.

2.  **During Training**:
    - **New Agents**: Automatically pushed (`sync_on_add=true`).
    - **Match Results**: Stored locally in `matches.csv` (unless `sync_on_match=true`).

3.  **After Training**:
    Since matches are local-only by default, you need to push them:
    
    ```python
    # In python console
    from league_manager import League
    l = League("../rl-league")
    l.push("Update match results")
    ```
    
    Or use terminal:
    ```bash
    cd ../rl-league
    git add data/matches.csv
    git commit -m "Update match results"
    git push
    ```
