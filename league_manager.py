import os
import json
import uuid
import time
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from openskill.models import PlackettLuce

from interfaces import LeagueAgent


class League:
    def __init__(self, repo_path, branch=None, auto_sync=False):
        """
        Initialize the League manager.

        Args:
            repo_path (str): Path to the root of the rl-league directory.
            branch (str): The name/branch of the league. Data is scoped to this name.
            auto_sync (bool): Ignored (kept for compatibility).
        """
        self.repo_path = Path(repo_path)

        # Scope data by branch/name to avoid mixing different runs
        if branch is None or branch == "":
            branch = "default"

        self.data_dir = self.repo_path / "data" / "leagues" / branch
        self.registry_dir = self.data_dir / "registry"
        self.snapshots_dir = self.data_dir / "snapshots"
        self.matches_path = self.data_dir / "matches.csv"

        # Ensure directories exist
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OpenSkill model (same as server)
        self.model = PlackettLuce()

        # Load state
        self.agents = {}  # Map[agent_id] -> dict (metadata)
        self.ratings = {}  # Map[agent_id] -> openskill.Rating

        self.reload()

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

    def add_agent(
        self,
        agent: LeagueAgent,
        owner: str,
        tags: list = None,
        sync=False,
        agent_id=None,
        is_fixed=False,
    ):
        """
        Snapshot an agent and add it to the league.

        Args:
            agent (LeagueAgent): The agent to save (must implement save()). Can be None if is_fixed=True.
            owner (str): Name of the user adding the agent.
            tags (list): Optional tags (e.g. 'gen10', 'production').
            sync (bool): Ignored (kept for compatibility).
            agent_id (str): Optional explicit ID. If None, generated from timestamp.
            is_fixed (bool): If True, does not save weights file (for benchmarks).
        """
        timestamp = int(time.time())
        if agent_id is None:
            agent_id = f"{owner}_{timestamp}"

        path_str = None
        if not is_fixed:
            # 1. Save Weights
            save_path = self.snapshots_dir / f"{agent_id}.pt"
            # Use the interface's save method
            if agent:
                agent.save(str(save_path))
            # Store relative path for portability
            try:
                path_str = str(save_path.relative_to(self.repo_path))
            except ValueError:
                # Fallback if not relative (shouldn't happen with standard structure)
                path_str = str(save_path)
        else:
            path_str = "FIXED"

        # 2. Save Metadata
        metadata = {
            "id": agent_id,
            "owner": owner,
            "timestamp": timestamp,
            "path": path_str,
            "tags": tags or [],
        }

        meta_path = self.registry_dir / f"{agent_id}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Added agent {agent_id} to league.")
        self.reload()

        return agent_id

    def ensure_benchmarks(self):
        """Ensures 'weak' and 'strong' benchmark agents exist in the registry."""
        benchmarks = ["weak", "strong"]
        for b in benchmarks:
            if b not in self.agents:
                print(f"League: Registering missing benchmark '{b}'...")
                self.add_agent(
                    agent=None,
                    owner="system",
                    tags=["benchmark", b],
                    agent_id=b,
                    is_fixed=True,
                    sync=False,
                )

    def report_match(self, agent1_id, agent2_id, score1, score2, sync=False):
        """
        Reports a match result and appends it to history.

        Args:
            sync (bool): Ignored (kept for compatibility).
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
        df = pd.DataFrame([record])
        df.to_csv(self.matches_path, mode="a", header=not file_exists, index=False)

        # Update local ratings immediately (partial update)
        self._recalculate_ratings()

    def get_rating(self, agent_id):
        """Returns (mu, sigma) for a given agent."""
        if agent_id not in self.ratings:
            # Default rating if not found (e.g. new agent)
            r = self.model.rating()
            return r.mu, r.sigma
        r = self.ratings[agent_id]
        return r.mu, r.sigma

    def matchmake(
        self, agent_id, strategy="pfsp", temperature=2.0, temperature_epsilon=0.2
    ):
        """
        Selects an opponent for the given agent.

        Args:
            agent_id: The ID of the agent looking for a match.
            strategy: 'pfsp' (Prioritized Fictitious Self-Play) or 'random'
            temperature: Softmax temperature (lower = harder/strict, higher = softer/random).
            temperature_epsilon: Probability to choose purely random snapshot under PFSP.

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
            # Epsilon-Greedy PFSP: 20% of the time, force purely random selection
            # to prevent catastrophic forgetting and ensure uniform history coverage
            if strategy == "pfsp" and np.random.rand() < temperature_epsilon:
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
                if not valid_candidates:
                    return None, None

                logits = -np.array(rating_diffs) / temperature

                # Stable softmax
                logits -= np.max(logits)
                probs = np.exp(logits)
                probs /= probs.sum()

                opp_id = np.random.choice(valid_candidates, p=probs)

        opp_data = self.agents[opp_id]
        # Return ID and absolute path to weights
        return opp_id, self.repo_path / opp_data["path"]
