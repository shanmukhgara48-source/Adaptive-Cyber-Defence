"""
Network graph model for the simulated SOC environment.

Provides adjacency-based topology queries used by the attack engine
to determine valid lateral movement paths.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set

from .state import AssetType, NetworkAsset


class NetworkGraph:
    """
    Simple undirected graph over NetworkAsset nodes.

    The graph is built once during env.reset() and remains structurally
    stable throughout an episode (nodes may become isolated but edges
    are not removed — isolation is tracked on the asset itself).

    Attributes:
        assets:     id → NetworkAsset map.
        _adj:       Adjacency list (id → set of neighbour ids).
    """

    def __init__(self, assets: Dict[str, NetworkAsset]) -> None:
        self.assets: Dict[str, NetworkAsset] = assets
        self._adj: Dict[str, Set[str]] = {nid: set() for nid in assets}
        # Mirror connections from asset.connected_to
        for nid, asset in assets.items():
            for neighbour in asset.connected_to:
                if neighbour in self._adj:
                    self._adj[nid].add(neighbour)
                    self._adj[neighbour].add(nid)

    # -----------------------------------------------------------------------
    # Topology queries
    # -----------------------------------------------------------------------

    def neighbours(self, node_id: str) -> List[str]:
        """Return IDs of all directly connected nodes."""
        return list(self._adj.get(node_id, set()))

    def reachable_from(self, node_id: str, exclude_isolated: bool = True) -> List[str]:
        """
        BFS from node_id; returns all reachable node IDs.
        If exclude_isolated=True, isolated nodes block traversal.
        """
        visited: Set[str] = set()
        queue: List[str] = [node_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            asset = self.assets.get(current)
            if asset and exclude_isolated and asset.is_isolated and current != node_id:
                continue
            for nb in self.neighbours(current):
                if nb not in visited:
                    queue.append(nb)
        visited.discard(node_id)
        return list(visited)

    def active_nodes(self) -> List[str]:
        """Return IDs of all nodes that are not isolated and have health > 0."""
        return [
            nid for nid, a in self.assets.items()
            if not a.is_isolated and a.health > 0.0
        ]

    def active_neighbours(self, node_id: str) -> List[str]:
        """
        Neighbours that are not isolated and have health > 0.
        These are valid lateral movement targets.
        """
        result = []
        for nb in self.neighbours(node_id):
            asset = self.assets.get(nb)
            if asset and not asset.is_isolated and asset.health > 0.0:
                result.append(nb)
        return result

    def most_vulnerable_neighbour(
        self,
        node_id: str,
        rng: random.Random,
    ) -> Optional[str]:
        """
        Return the active neighbour with highest vulnerability score,
        with tie-breaking via rng for determinism.
        """
        candidates = self.active_neighbours(node_id)
        if not candidates:
            return None
        candidates.sort(
            key=lambda nid: (
                -self.assets[nid].vulnerability_score(),
                nid,   # stable tie-break
            )
        )
        # Small stochastic element: top candidate chosen with 80% probability
        if rng.random() < 0.80:
            return candidates[0]
        return rng.choice(candidates)

    # -----------------------------------------------------------------------
    # Factory helpers
    # -----------------------------------------------------------------------

    @classmethod
    def build_default(cls, rng: random.Random) -> "NetworkGraph":
        """
        Construct the default 8-node SOC network topology.

        Topology (roughly a corporate LAN):

            FIREWALL ── ROUTER ── SERVER-WEB
                                  │
                          SERVER-DB ── WORKSTATION-01
                          │             │
                    WORKSTATION-02   WORKSTATION-03
                          │
                      DATABASE

        Asset criticality reflects business importance.
        """
        assets: Dict[str, NetworkAsset] = {
            "fw-01": NetworkAsset(
                id="fw-01",
                asset_type=AssetType.FIREWALL,
                health=1.0,
                is_compromised=False,
                is_isolated=False,
                patch_level=rng.uniform(0.7, 1.0),
                criticality=0.9,
                connected_to=["router-01"],
            ),
            "router-01": NetworkAsset(
                id="router-01",
                asset_type=AssetType.ROUTER,
                health=1.0,
                is_compromised=False,
                is_isolated=False,
                patch_level=rng.uniform(0.6, 1.0),
                criticality=0.8,
                connected_to=["fw-01", "srv-web", "srv-db", "ws-01"],
            ),
            "srv-web": NetworkAsset(
                id="srv-web",
                asset_type=AssetType.SERVER,
                health=1.0,
                is_compromised=False,
                is_isolated=False,
                patch_level=rng.uniform(0.4, 0.9),
                criticality=0.7,
                connected_to=["router-01", "srv-db"],
            ),
            "srv-db": NetworkAsset(
                id="srv-db",
                asset_type=AssetType.DATABASE,
                health=1.0,
                is_compromised=False,
                is_isolated=False,
                patch_level=rng.uniform(0.3, 0.8),
                criticality=1.0,   # highest — contains sensitive data
                connected_to=["router-01", "srv-web", "ws-02", "db-01"],
            ),
            "db-01": NetworkAsset(
                id="db-01",
                asset_type=AssetType.DATABASE,
                health=1.0,
                is_compromised=False,
                is_isolated=False,
                patch_level=rng.uniform(0.2, 0.7),
                criticality=0.95,
                connected_to=["srv-db", "ws-02"],
            ),
            "ws-01": NetworkAsset(
                id="ws-01",
                asset_type=AssetType.WORKSTATION,
                health=1.0,
                is_compromised=False,
                is_isolated=False,
                patch_level=rng.uniform(0.2, 0.8),
                criticality=0.3,
                connected_to=["router-01", "ws-03"],
            ),
            "ws-02": NetworkAsset(
                id="ws-02",
                asset_type=AssetType.WORKSTATION,
                health=1.0,
                is_compromised=False,
                is_isolated=False,
                patch_level=rng.uniform(0.1, 0.7),
                criticality=0.3,
                connected_to=["srv-db", "db-01", "ws-03"],
            ),
            "ws-03": NetworkAsset(
                id="ws-03",
                asset_type=AssetType.WORKSTATION,
                health=1.0,
                is_compromised=False,
                is_isolated=False,
                patch_level=rng.uniform(0.1, 0.6),
                criticality=0.2,
                connected_to=["ws-01", "ws-02"],
            ),
        }
        return cls(assets)

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def summary(self) -> str:
        lines = ["NetworkGraph:"]
        for nid, neighbours in self._adj.items():
            asset = self.assets[nid]
            status = "ISOLATED" if asset.is_isolated else ("COMPROMISED" if asset.is_compromised else "ok")
            lines.append(
                f"  {nid} [{asset.asset_type.value}] "
                f"health={asset.health:.2f} patch={asset.patch_level:.2f} "
                f"crit={asset.criticality:.2f} status={status} "
                f"→ {sorted(neighbours)}"
            )
        return "\n".join(lines)
