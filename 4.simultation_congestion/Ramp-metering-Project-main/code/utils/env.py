import os
import sys
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import traci


class RampMeterEnv:
    """
    SUMO Environment for ramp metering control (scenario-agnostic).

    Key idea:
    - Do NOT hard-code edge IDs or traffic light IDs.
    - Read them from config, or auto-detect after SUMO starts.

    Recommended config:
    {
        "sumo_binary": "sumo" or "sumo-gui",
        "gui": False,
        "sumocfg": "../../config/highway.sumocfg",   # recommended
        # OR:
        # "net_file": "../../config/highway.net.xml",
        # "route_file": "../../config/highway.rou.xml",

        # Optional explicit IDs (recommended for custom scenarios)
        "main_in_edges": ["highway_in"],
        "main_out_edges": ["highway_out"],
        "ramp_edges": ["ramp"],
        "tl_id": "merge",

        # Control mapping (choose one)
        # If you keep 4 phases (0 mainline green/ramp red, 2 mainline red/ramp green):
        "action_to_phase": {0: 0, 1: 2},

        # Episode horizon
        "episode_seconds": 3600,   # stop by simulation time (seconds)
        # or "max_steps": 3600

        # Step length in SUMO (optional)
        "step_length": 0.1,
    }
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # SUMO binary
        self.sumo_binary = config.get(
            "sumo_binary",
            "sumo-gui" if config.get("gui", False) else "sumo"
        )

        # Either sumocfg OR net+route
        self.sumocfg = config.get("sumocfg", None)
        self.net_file = config.get("net_file", None)
        self.route_file = config.get("route_file", None)

        # Episode termination
        self.episode_seconds = config.get("episode_seconds", None)
        self.max_steps = config.get("max_steps", None)

        # step length (used only for converting steps<->seconds if needed)
        self.step_length = float(config.get("step_length", 1.0))

        # IDs (optional, can be auto-detected)
        self.main_in_edges: Optional[List[str]] = config.get("main_in_edges", None)
        self.main_out_edges: Optional[List[str]] = config.get("main_out_edges", None)
        self.ramp_edges: Optional[List[str]] = config.get("ramp_edges", None)
        self.tl_id: Optional[str] = config.get("tl_id", None)

        # Action -> phase mapping
        # Default matches your old design: action 0 -> phase 0, action 1 -> phase 2
        self.action_to_phase: Dict[int, int] = config.get("action_to_phase", {0: 0, 1: 2})

        # Observation/action spaces (keep your original shape: 6-dim state, 2 actions)
        self.observation_space_size = 6
        self.action_space_size = 2

        self.simulation_step = 0

        # Add SUMO tools to PYTHONPATH once
        if "SUMO_HOME" in os.environ:
            tools = os.path.join(os.environ["SUMO_HOME"], "tools")
            if tools not in sys.path:
                sys.path.append(tools)
        else:
            raise EnvironmentError("Please declare environment variable 'SUMO_HOME'")

    # ---------------------------
    # SUMO lifecycle
    # ---------------------------
    def reset(self) -> np.ndarray:
        """Reset the environment and return initial state."""
        self.close()

        sumo_cmd = self._build_sumo_cmd()
        traci.start(sumo_cmd)


        print("TLS:", traci.trafficlight.getIDList())
        print("TL phases:", traci.trafficlight.getPhaseNumber(self.tl_id))

        self.simulation_step = 0

        # Auto-detect IDs if not provided
        self._auto_detect_ids_if_needed()

        return self._get_state()

    def close(self):
        """Close SUMO safely (idempotent)."""
        try:
            if hasattr(traci, "isLoaded") and traci.isLoaded():
                traci.close(False)
                return
        except Exception:
            pass

        try:
            traci.close(False)
        except Exception:
            pass

    def _build_sumo_cmd(self) -> List[str]:
        """Build SUMO command line."""
        cmd = [self.sumo_binary]

        if self.sumocfg is not None:
            cmd += ["-c", self.sumocfg]
        else:
            if not self.net_file or not self.route_file:
                raise ValueError("Provide either 'sumocfg' or both 'net_file' and 'route_file'.")
            cmd += ["-n", self.net_file, "-r", self.route_file]

        # Keep it quiet but stable
        cmd += ["--start", "--no-warnings", "--quit-on-end"]

        # Optional step length
        if self.step_length is not None:
            cmd += ["--step-length", str(self.step_length)]

        # Optional: reduce console spam
        # cmd += ["--no-step-log", "true"]

        return cmd

    def _auto_detect_ids_if_needed(self):
        """
        If user didn't provide IDs, detect them based on lane count and naming heuristics.
        This avoids "Edge is not known" / "Traffic light is not known".
        """
        # Traffic light
        if self.tl_id is None:
            tls = traci.trafficlight.getIDList()
            self.tl_id = tls[0] if len(tls) > 0 else None

        # Edge IDs
        all_edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
        if len(all_edges) == 0:
            raise RuntimeError("No non-internal edges found. Check your network file.")

        # If ramp edges not given: try name contains "ramp" else choose smallest lane count edge
        if self.ramp_edges is None:
            ramp_candidates = [e for e in all_edges if "ramp" in e.lower() or "onramp" in e.lower()]
            if ramp_candidates:
                self.ramp_edges = ramp_candidates
            else:
                # choose edges with minimum lanes (often ramp)
                lane_counts = [(e, traci.edge.getLaneNumber(e)) for e in all_edges]
                min_lanes = min(n for _, n in lane_counts)
                self.ramp_edges = [e for e, n in lane_counts if n == min_lanes]

        # If mainline edges not given: choose edges with maximum lane count that are not ramp edges
        if self.main_in_edges is None or self.main_out_edges is None:
            non_ramp = [e for e in all_edges if e not in set(self.ramp_edges)]
            lane_counts = [(e, traci.edge.getLaneNumber(e)) for e in non_ramp]
            max_lanes = max(n for _, n in lane_counts)
            main_candidates = [e for e, n in lane_counts if n == max_lanes]

            # If we need split into in/out, do a simple heuristic:
            # use names if possible, else take first two
            if self.main_in_edges is None:
                in_like = [e for e in main_candidates if "in" in e.lower() or "up" in e.lower() or "entry" in e.lower()]
                self.main_in_edges = in_like if in_like else [main_candidates[0]]

            if self.main_out_edges is None:
                out_like = [e for e in main_candidates if "out" in e.lower() or "down" in e.lower() or "exit" in e.lower()]
                if out_like:
                    self.main_out_edges = out_like
                else:
                    # choose something different from in if possible
                    remaining = [e for e in main_candidates if e not in set(self.main_in_edges)]
                    self.main_out_edges = remaining if remaining else [main_candidates[0]]

    # ---------------------------
    # RL step
    # ---------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return (state, reward, done, info)."""
        self._apply_action(action)

        traci.simulationStep()
        self.simulation_step += 1

        state = self._get_state()
        reward = self._calculate_reward()

        done = self._is_done()

        info = {
            "waiting_time": self._get_average_waiting_time(),
            "vehicles_passed": self._get_vehicles_passed(),
            "avg_speed": self._get_average_speed(),
        }

        return state, reward, done, info

    def _is_done(self) -> bool:
        # End if no vehicles expected
        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                return True
        except Exception:
            pass

        # End by sim time (recommended)
        if self.episode_seconds is not None:
            try:
                if traci.simulation.getTime() >= float(self.episode_seconds):
                    return True
            except Exception:
                pass

        # End by step count (fallback)
        if self.max_steps is not None:
            if self.simulation_step >= int(self.max_steps):
                return True

        return False

    # ---------------------------
    # State / Action / Reward
    # ---------------------------
    def _get_state(self) -> np.ndarray:
        """
        State = [
          mainline_density_per_lane (veh/km/lane),
          ramp_queue (halting vehicles),
          mainline_avg_speed (km/h),
          ramp_avg_speed (km/h),
          mainline_vehicle_count,
          ramp_vehicle_count
        ]
        """
        try:
            # Vehicle counts
            main_in_n = sum(traci.edge.getLastStepVehicleNumber(e) for e in self.main_in_edges)
            main_out_n = sum(traci.edge.getLastStepVehicleNumber(e) for e in self.main_out_edges)
            ramp_n = sum(traci.edge.getLastStepVehicleNumber(e) for e in self.ramp_edges)

            main_total_n = main_in_n + main_out_n

            # Speeds (m/s -> km/h)
            main_speeds = []
            for e in (self.main_in_edges + self.main_out_edges):
                s = traci.edge.getLastStepMeanSpeed(e)
                if s >= 0:
                    main_speeds.append(s)
            main_speed_kmh = (np.mean(main_speeds) if main_speeds else 0.0) * 3.6

            ramp_speeds = []
            for e in self.ramp_edges:
                s = traci.edge.getLastStepMeanSpeed(e)
                if s >= 0:
                    ramp_speeds.append(s)
            ramp_speed_kmh = (np.mean(ramp_speeds) if ramp_speeds else 0.0) * 3.6

            # Queue: use halting number (speed ~ 0)
            ramp_queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in self.ramp_edges)

            # Density per lane: vehicles / (km * lanes)
            lengths_m = []
            lane_nums = []
            for e in (self.main_in_edges + self.main_out_edges):
                try:
                    # lane length is accessible via lane.getLength; edge length not always
                    lane0 = f"{e}_0"
                    lengths_m.append(traci.lane.getLength(lane0))
                except Exception:
                    # fallback: ignore if not accessible
                    pass
                try:
                    lane_nums.append(traci.edge.getLaneNumber(e))
                except Exception:
                    pass

            total_km = (sum(lengths_m) / 1000.0) if lengths_m else 1.0
            # Use max lane count as representative (mainline)
            lanes = max(lane_nums) if lane_nums else 1
            main_density = main_total_n / max(1e-6, total_km * lanes)

            return np.array(
                [main_density, float(ramp_queue), main_speed_kmh, ramp_speed_kmh, float(main_total_n), float(ramp_n)],
                dtype=np.float32
            )
        except Exception:
            # If anything goes wrong, return zeros but do NOT spam prints every step
            return np.zeros(self.observation_space_size, dtype=np.float32)

    def _apply_action(self, action: int):
        """
        Apply metering action. Default: set TLS phase.
        If tl_id is None (no TLS in scenario), this becomes no-op.
        """
        if self.tl_id is None:
            return

        try:
            phase = self.action_to_phase.get(int(action), 0)
            traci.trafficlight.setPhase(self.tl_id, int(phase))
        except Exception:
            # avoid console spam
            pass

    def _calculate_reward(self) -> float:
        """
        Reward encourages mainline speed and penalizes ramp queue.
        You should tune weights for your scenario.
        """
        try:
            # Mainline speed km/h
            main_speeds = []
            for e in (self.main_in_edges + self.main_out_edges):
                s = traci.edge.getLastStepMeanSpeed(e)
                if s >= 0:
                    main_speeds.append(s)
            main_speed_kmh = (np.mean(main_speeds) if main_speeds else 0.0) * 3.6

            ramp_queue = sum(traci.edge.getLastStepHaltingNumber(e) for e in self.ramp_edges)

            # Normalize by desired speed (adjust to your speed limit)
            desired_speed_kmh = self.config.get("desired_speed_kmh", 100.0)
            speed_reward = main_speed_kmh / max(1e-6, desired_speed_kmh)

            queue_penalty_w = float(self.config.get("queue_penalty_w", 0.1))
            queue_penalty = -queue_penalty_w * ramp_queue

            return float(speed_reward + queue_penalty)
        except Exception:
            return 0.0

    # ---------------------------
    # Diagnostics (optional)
    # ---------------------------
    def _get_average_waiting_time(self) -> float:
        try:
            total_wait = 0.0
            count = 0

            for e in (self.main_in_edges + self.main_out_edges + self.ramp_edges):
                vids = traci.edge.getLastStepVehicleIDs(e)
                for v in vids:
                    total_wait += traci.vehicle.getWaitingTime(v)
                    count += 1
            return total_wait / max(1, count)
        except Exception:
            return 0.0

    def _get_vehicles_passed(self) -> int:
        """Vehicles on downstream edge(s) in this step (proxy)."""
        try:
            return int(sum(traci.edge.getLastStepVehicleNumber(e) for e in self.main_out_edges))
        except Exception:
            return 0

    def _get_average_speed(self) -> float:
        """Average speed over monitored edges (km/h)."""
        try:
            speeds = []
            for e in (self.main_in_edges + self.main_out_edges + self.ramp_edges):
                s = traci.edge.getLastStepMeanSpeed(e)
                if s >= 0:
                    speeds.append(s * 3.6)
            return float(np.mean(speeds)) if speeds else 0.0
        except Exception:
            return 0.0
