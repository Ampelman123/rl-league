import os
import json
import uuid
import time
import shutil
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from openskill.models import PlackettLuce

from interfaces import LeagueAgent


class League:
    def __init__(self, repo_path, branch="main", auto_sync=True):
        """
        Initialize the League manager.

        Args:
            repo_path (str): Path to the root of the rl-league repository.
            branch (str): Git branch to use for this league.
            auto_sync (bool): If True, pulls from remote on initialization.
        """
        self.repo_path = Path(repo_path)
        self.branch = branch
        self.data_dir = self.repo_path / "data"
        self.registry_dir = self.data_dir / "registry"
        self.snapshots_dir = self.data_dir / "snapshots"
        self.matches_path = self.data_dir / "matches.csv"

        # Check if repo is a git repository
        self.is_git_repo = (self.repo_path / ".git").exists()
        if not self.is_git_repo:
            print(
                f"League: WARNING - Not a git repository: {self.repo_path}. Sync disabled."
            )

        # Switch Branch (only if git repo)
        if self.is_git_repo:
            current_branch = self._get_current_branch()
            if current_branch != branch:
                print(f"League: Switching from {current_branch} to {branch}...")
                if not self._checkout_branch(branch):
                    print(
                        f"League: Failed to switch to branch {branch}. Staying on {current_branch}."
                    )
            else:
                print(f"League: On branch {branch}")

        # Ensure directories exist (in case branch was empty/new)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OpenSkill model (same as server)
        self.model = PlackettLuce()

        # Load state
        self.agents = {}  # Map[agent_id] -> dict (metadata)
        self.ratings = {}  # Map[agent_id] -> openskill.Rating

        if auto_sync:
            self.pull()

        self.reload()

    def _get_current_branch(self):
        try:
            res = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        except:
            return "unknown"

    def _checkout_branch(self, branch):
        # 1. Try checkout existing
        if self._git_cmd(["checkout", branch]):
            return True
        # 2. Try checkout new local branch tracking origin
        if self._git_cmd(["checkout", "-b", branch, f"origin/{branch}"]):
            return True
        # 3. Create new orphan/empty branch? Or just standard new branch from current?
        # Standard: git checkout -b branch
        print(f"League: Branch {branch} not found. Creating new branch.")
        return self._git_cmd(["checkout", "-b", branch])

    def _git_cmd(self, args):
        """Helper to run git commands in the league repo."""
        try:
            subprocess.run(
                ["git"] + args,
                cwd=self.repo_path,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {e.stderr.decode().strip()}")
            return False

    def pull(self):
        """Pulls latest changes from remote."""
        if not self.is_git_repo:
            return True

        print("League: Pulling latest data...")
        if self._git_cmd(["pull", "--rebase", "--autostash"]):
            self.reload()
            return True
        return False

    def push(self, commit_message="Update league data"):
        """Commits and pushes changes to remote with retry logic."""
        if not self.is_git_repo:
            print("League: Not a git repository, skipping push.")
            return False

        print(f"League: Pushing changes ({commit_message})...")
        self._git_cmd(["add", "data/"])
        # Attempts to commit. If minimal changes, this might fail (empty commit) but we proceed.
        self._git_cmd(["commit", "-m", commit_message])

        # Retry loop for push (handle concurrent updates)
        for i in range(3):
            if self._git_cmd(["push", "-u", "origin", self.branch]):
                return True
            print(f"League: Push failed (attempt {i+1}/3). Pulling and retrying...")
            if not self.pull():
                time.sleep(1)  # Wait a bit before retry

        print("League: Push failed after retries.")
        return False

    def reload(self):
        """Reloads agents and recalculates ratings from match history."""
        self._load_agents()
        self._recalculate_ratings()

    def _load_agents(self):
        """Loads all agent metadata from registry/."""
        self.agents = {}
        for entry in self.registry_dir.glob("*.json"):
            try:
                with open(entry, "r") as f:
                    data = json.load(f)
                    self.agents[data["id"]] = data
            except Exception as e:
                print(f"Failed to load registry entry {entry}: {e}")

    def _recalculate_ratings(self):
        """Replays match history to determine current ratings."""
        self.ratings = {}

        # Initialize everyone with default rating
        for agent_id in self.agents:
            self.ratings[agent_id] = self.model.rating()

        if not self.matches_path.exists():
            return

        try:
            # Read history
            df = pd.read_csv(self.matches_path)

            for _, row in df.iterrows():
                p1_id = row["agent1"]
                p2_id = row["agent2"]
                s1 = row["score1"]
                s2 = row["score2"]

                # Ensure agents exist in ratings (might be deleted from registry but in history)
                if p1_id not in self.ratings:
                    self.ratings[p1_id] = self.model.rating()
                if p2_id not in self.ratings:
                    self.ratings[p2_id] = self.model.rating()

                # Calculate outcome for openskill
                # openskill expects ranks where lower is better (0=winner, 1=loser)
                # If draw, ranks are equal
                if s1 > s2:
                    ranks = [0, 1]  # p1 wins
                elif s2 > s1:
                    ranks = [1, 0]  # p2 wins
                else:
                    ranks = [0, 0]  # draw

                # Update
                p1_rating = self.ratings[p1_id]
                p2_rating = self.ratings[p2_id]

                [[new_p1], [new_p2]] = self.model.rate(
                    [[p1_rating], [p2_rating]], ranks=ranks
                )

                self.ratings[p1_id] = new_p1
                self.ratings[p2_id] = new_p2

        except Exception as e:
            print(f"Error recalculating ratings: {e}")

    def add_agent(self, agent: LeagueAgent, owner: str, tags: list = None, sync=True):
        """
        Snapshot an agent and add it to the league.

        Args:
            agent (LeagueAgent): The agent to save (must implement save()).
            owner (str): Name of the user adding the agent.
            tags (list): Optional tags (e.g. 'gen10', 'production').
            sync (bool): Whether to auto-push after adding.
        """
        timestamp = int(time.time())
        agent_id = f"{owner}_{timestamp}"

        # 1. Save Weights
        save_path = self.snapshots_dir / f"{agent_id}.pt"

        # Use the interface's save method
        agent.save(str(save_path))

        # 2. Save Metadata
        metadata = {
            "id": agent_id,
            "owner": owner,
            "timestamp": timestamp,
            "path": str(save_path.relative_to(self.repo_path)),
            "tags": tags or [],
        }

        meta_path = self.registry_dir / f"{agent_id}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Added agent {agent_id} to league.")
        self.reload()

        if sync:
            self.push(f"Add agent {agent_id}")

        return agent_id

    def report_match(self, agent1_id, agent2_id, score1, score2, sync=False):
        """
        Reports a match result and appends it to history.

        Args:
            sync (bool): If True, pushes the result to remote immediately.
        """
        # Ensure timestamp is unique/monotonically increasing if multiple writers (simple approximation)
        timestamp = int(time.time() * 1000)

        record = {
            "timestamp": timestamp,
            "agent1": agent1_id,
            "agent2": agent2_id,
            "score1": float(score1),
            "score2": float(score2),
        }

        # Append to CSV
        file_exists = self.matches_path.exists()

        # DataFrame is heavy for single row append, but handles CSV escaping well.
        # For performance in high freq, manual string append is better.
        # Here we use pandas for safety.
        df = pd.DataFrame([record])
        df.to_csv(self.matches_path, mode="a", header=not file_exists, index=False)

        # Update local ratings immediately (partial update)
        # We could optimize by only updating the two agents involved,
        # but for league size < 1000, full recalc is cheap (<100ms) and safer for consistency.
        self._recalculate_ratings()

        if sync:
            self.push(f"Match result: {agent1_id} vs {agent2_id}")

    def get_rating(self, agent_id):
        """Returns (mu, sigma) for a given agent."""
        if agent_id not in self.ratings:
            # Default rating if not found (e.g. new agent)
            r = self.model.rating()
            return r.mu, r.sigma
        r = self.ratings[agent_id]
        return r.mu, r.sigma

    def matchmake(self, agent_id, strategy="pfsp", temperature=2.0):
        """
        Selects an opponent for the given agent.

        Args:
            agent_id: The ID of the agent looking for a match.
            strategy: 'pfsp' (Prioritized Fictitious Self-Play) or 'random'
            temperature: Softmax temperature (lower = harder/strict, higher = softer/random).

        Returns:
            opponent_id (str), opponent_path (Path)
        """
        # 1. Filter candidates (exclude self)
        candidates = [aid for aid in self.agents.keys() if aid != agent_id]

        if not candidates:
            return None, None

        if strategy == "random":
            opp_id = np.random.choice(candidates)
        else:
            # PFSP: Pick someone with similar rating using softmax
            my_mu, _ = self.get_rating(agent_id)
            rating_diffs = []
            valid_candidates = []

            for cid in candidates:
                c_mu, _ = self.get_rating(cid)
                dist = abs(my_mu - c_mu)
                rating_diffs.append(dist)
                valid_candidates.append(cid)

            # Softmax: exp(-dist / temperature)
            # Temperature heuristic: For mu~25, differences of 5-10 are significant.
            # T=5.0 gives a softer distribution, T=1.0 is very sharp.
            # temperature is now passed as an argument
            logits = -np.array(rating_diffs) / temperature

            # Stable softmax
            logits -= np.max(logits)
            probs = np.exp(logits)
            probs /= probs.sum()

            opp_id = np.random.choice(valid_candidates, p=probs)

        opp_data = self.agents[opp_id]
        # Return ID and absolute path to weights
        return opp_id, self.repo_path / opp_data["path"]
