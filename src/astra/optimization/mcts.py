"""Monte Carlo Tree Search for multi-body mission phase planning.
Plans discrete sequences of flyby bodies between origin and destination.
Each node represents a mission phase state (current body, current epoch,
current heliocentric velocity). MCTS explores flyby body sequences and
evaluates them using Lambert + SOI computation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from astra.dsl.compiler import CompiledMission
from astra.physics.flyby import SAFE_FLYBY_ALTITUDE_KM
from astra.physics.kernel import PhysicsKernel
from astra.state.orbital_state import GM, PHYSICAL_RADIUS, CelestialBody


@dataclass
class PhaseState:
    body: str
    epoch: float
    v_helio: np.ndarray  # spacecraft heliocentric velocity after arrival/departure [km/s]
    dv_spent: float      # cumulative delta-v spent [km/s]


class MCTSNode:
    def __init__(
        self,
        state: PhaseState,
        parent: MCTSNode | None = None,
        untried_actions: list[tuple[str, float]] | None = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.n_visits = 0
        self.total_value = 0.0
        self.untried_actions = untried_actions if untried_actions is not None else []


class MCTSPlanner:
    def __init__(
        self,
        mission: CompiledMission,
        kernel: PhysicsKernel,
        max_depth: int = 4,
        exploration_constant: float = 1.414,
        n_iterations: int = 500,
        dv_budget: float = 15.0,
        max_duration: float = 1000.0 * 86400.0,
        seed: int = 42,
        flyby_candidates: list[str] | None = None,
    ) -> None:
        self.mission = mission
        self.kernel = kernel
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.n_iterations = n_iterations
        self.dv_budget = dv_budget
        self.max_duration = max_duration
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.origin_body = mission.origin_body.name.upper()
        self.destination_body = mission.destination_body.name.upper()
        self.start_epoch = mission.departure_epoch_start

        if flyby_candidates is None:
            self.flyby_candidates = ["VENUS", "EARTH", "MOON"]
        else:
            self.flyby_candidates = [b.upper() for b in flyby_candidates]

        # Initialize root node with origin state
        r0_state = kernel.get_body_state(mission.origin_body, self.start_epoch)
        root_state = PhaseState(
            body=self.origin_body,
            epoch=self.start_epoch,
            v_helio=r0_state.velocity,
            dv_spent=0.0,
        )
        root_actions = self._get_valid_actions(root_state)
        self.root = MCTSNode(root_state, untried_actions=root_actions)

    def run(self) -> list[list[PhaseState]]:
        """Run the MCTS phase planner search.
        Returns a list of PhaseState sequences representing promising flyby chains,
        sorted by total delta-v ascending.
        """
        for _ in range(self.n_iterations):
            node = self._select(self.root)

            # Expand if possible
            if node.untried_actions:
                child = self._expand(node)
                if child is not None:
                    # Simulate from child
                    reward = self._simulate(child.state, self._get_node_depth(child))
                    self._backpropagate(child, reward)
                else:
                    self._backpropagate(node, 0.0)
            else:
                # Terminal node or fully expanded
                reward = self._calculate_reward(node.state)
                self._backpropagate(node, reward)

        # Collect all completed paths reaching the destination
        all_paths: list[list[PhaseState]] = []
        self._collect_paths(self.root, [], all_paths)

        valid_paths = [
            p for p in all_paths
            if p[-1].body == self.destination_body and p[-1].dv_spent <= self.dv_budget
        ]
        valid_paths.sort(key=lambda p: p[-1].dv_spent)
        return valid_paths

    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while current.children and not current.untried_actions:
            if current.state.body == self.destination_body:
                return current

            best_child = None
            best_uct = -float("inf")
            log_parent_visits = math.log(max(current.n_visits, 1))

            for child in current.children:
                if child.n_visits == 0:
                    uct = float("inf")
                else:
                    exploitation = child.total_value / child.n_visits
                    exploration = self.exploration_constant * math.sqrt(
                        log_parent_visits / child.n_visits
                    )
                    uct = exploitation + exploration

                if uct > best_uct:
                    best_uct = uct
                    best_child = child
            if best_child is None:
                break
            current = best_child
        return current

    def _expand(self, node: MCTSNode) -> MCTSNode | None:
        if not node.untried_actions:
            return None

        action = node.untried_actions.pop()
        next_state = self._apply_action(node.state, action)
        if next_state is None:
            return None

        child_actions = self._get_valid_actions(next_state)
        child = MCTSNode(next_state, parent=node, untried_actions=child_actions)
        node.children.append(child)
        return child

    def _simulate(self, state: PhaseState, depth: int) -> float:
        current_state = state
        current_depth = depth

        while current_depth < self.max_depth and current_state.body != self.destination_body:
            actions = self._get_valid_actions(current_state)
            if not actions:
                break

            self.rng.shuffle(actions)
            next_state = None
            for action in actions:
                next_state = self._apply_action(current_state, action)
                if next_state is not None:
                    break

            if next_state is None:
                return 0.0

            current_state = next_state
            current_depth += 1

        return self._calculate_reward(current_state)

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        curr: MCTSNode | None = node
        while curr is not None:
            curr.n_visits += 1
            curr.total_value += reward
            curr = curr.parent

    def _get_valid_actions(self, state: PhaseState) -> list[tuple[str, float]]:
        if state.body == self.destination_body:
            return []

        actions = []
        candidates = self.flyby_candidates + [self.destination_body]
        next_bodies = [b for b in candidates if b != state.body]

        for body in next_bodies:
            # Moon TOFs vs Interplanetary TOFs
            if body == "MOON" or state.body == "MOON":
                tofs = [1.5 * 86400.0, 2.5 * 86400.0, 3.5 * 86400.0, 4.5 * 86400.0]
            else:
                tofs = [
                    100.0 * 86400.0,
                    150.0 * 86400.0,
                    200.0 * 86400.0,
                    250.0 * 86400.0,
                    300.0 * 86400.0,
                    350.0 * 86400.0,
                ]
            for tof in tofs:
                actions.append((body, tof))
        return actions

    def _apply_action(self, state: PhaseState, action: tuple[str, float]) -> PhaseState | None:
        next_body, tof_seconds = action
        epoch_arr = state.epoch + tof_seconds

        if epoch_arr - self.start_epoch > self.max_duration:
            return None

        try:
            body_from = CelestialBody[state.body]
            body_to = CelestialBody[next_body]

            r1_state = self.kernel.get_body_state(body_from, state.epoch)
            r1 = r1_state.position
            v1_body = r1_state.velocity

            r2_state = self.kernel.get_body_state(body_to, epoch_arr)
            r2 = r2_state.position
            v2_body = r2_state.velocity
        except Exception:
            return None

        from astra.physics.lambert import find_best_transfer
        try:
            sol = find_best_transfer(
                r1=r1,
                v1_body=v1_body,
                r2=r2,
                v2_body=v2_body,
                tof=tof_seconds,
                mu=GM["SUN"],
                max_revs=0,
            )
            v_dep = sol.v1
            v_arr = sol.v2
        except Exception:
            return None

        # Compute delta-v cost
        if state.body == self.origin_body and state.dv_spent == 0.0:
            from astra.physics.maneuvers import departure_delta_v
            h_park = 200.0
            v_inf_dep = v_dep - v1_body
            dv_cost = departure_delta_v(v_inf_dep, h_park, state.body)
        else:
            v_inf_in = state.v_helio - v1_body
            v_inf_out = v_dep - v1_body

            v_inf_in_mag = float(np.linalg.norm(v_inf_in))
            v_inf_out_mag = float(np.linalg.norm(v_inf_out))
            if v_inf_in_mag <= 1e-6 or v_inf_out_mag <= 1e-6:
                return None

            cos_theta = float(np.dot(v_inf_in, v_inf_out) / (v_inf_in_mag * v_inf_out_mag))
            cos_theta = max(-1.0, min(1.0, cos_theta))
            target_turn_angle_rad = math.acos(cos_theta)

            r_p = self._solve_periapsis_for_turn_angle(
                v_inf_in_mag, v_inf_out_mag, target_turn_angle_rad, state.body
            )

            mu = GM[state.body.upper()]
            R_body = PHYSICAL_RADIUS[CelestialBody[state.body.upper()]]
            safe_alt = SAFE_FLYBY_ALTITUDE_KM.get(state.body.upper(), 300.0)
            r_min = R_body + safe_alt

            e_in_min = 1.0 + r_min * v_inf_in_mag**2 / mu
            e_out_min = 1.0 + r_min * v_inf_out_mag**2 / mu
            max_turn = math.asin(1.0 / e_in_min) + math.asin(1.0 / e_out_min)

            v_peri_in = math.sqrt(v_inf_in_mag**2 + 2.0 * mu / r_p)
            v_peri_out = math.sqrt(v_inf_out_mag**2 + 2.0 * mu / r_p)
            powered_dv = abs(v_peri_out - v_peri_in)

            deflection_dv = 0.0
            if target_turn_angle_rad > max_turn:
                delta_theta = target_turn_angle_rad - max_turn
                deflection_dv = 2.0 * v_inf_in_mag * math.sin(delta_theta / 2.0)

            dv_cost = powered_dv + deflection_dv

        new_dv_spent = state.dv_spent + dv_cost

        if next_body == self.destination_body:
            from astra.physics.maneuvers import arrival_delta_v
            h_cap = 300.0
            v_inf_arr = v2_body - v_arr
            dv_cap = arrival_delta_v(v_inf_arr, h_cap, next_body)
            new_dv_spent += dv_cap

        return PhaseState(
            body=next_body,
            epoch=epoch_arr,
            v_helio=v_arr,
            dv_spent=new_dv_spent,
        )

    def _solve_periapsis_for_turn_angle(
        self,
        v_inf_in_mag: float,
        v_inf_out_mag: float,
        target_turn_angle_rad: float,
        body: str,
    ) -> float:
        mu = GM[body.upper()]
        r_body = PHYSICAL_RADIUS[CelestialBody[body.upper()]]
        safe_alt = SAFE_FLYBY_ALTITUDE_KM.get(body.upper(), 300.0)
        r_min = r_body + safe_alt

        e_in_min = 1.0 + r_min * v_inf_in_mag**2 / mu
        e_out_min = 1.0 + r_min * v_inf_out_mag**2 / mu
        max_turn = math.asin(1.0 / e_in_min) + math.asin(1.0 / e_out_min)

        if target_turn_angle_rad >= max_turn:
            return r_min

        low = r_min
        high = 100.0 * r_min
        for _ in range(30):
            mid = (low + high) / 2.0
            e_in = 1.0 + mid * v_inf_in_mag**2 / mu
            e_out = 1.0 + mid * v_inf_out_mag**2 / mu
            turn = math.asin(1.0 / e_in) + math.asin(1.0 / e_out)
            if turn < target_turn_angle_rad:
                high = mid
            else:
                low = mid
        return low

    def _calculate_reward(self, state: PhaseState) -> float:
        if state.body != self.destination_body:
            return 0.0
        if state.dv_spent > self.dv_budget:
            return 0.0
        return max(0.0, 1.0 - state.dv_spent / self.dv_budget)

    def _get_node_depth(self, node: MCTSNode) -> int:
        depth = 0
        curr = node
        while curr.parent is not None:
            depth += 1
            curr = curr.parent
        return depth

    def _collect_paths(
        self,
        node: MCTSNode,
        current_path: list[PhaseState],
        all_paths: list[list[PhaseState]],
    ) -> None:
        path = current_path + [node.state]
        if node.state.body == self.destination_body:
            all_paths.append(path)
            return
        for child in node.children:
            self._collect_paths(child, path, all_paths)
