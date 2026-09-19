from __future__ import annotations

from collections import deque
from copy import deepcopy
from queue import Empty, Queue
from threading import Lock

from order_system import MENU


class RestaurantOverlay:
    """Thread-safe order + telemetry panel rendered inside the Genesis viewer."""

    def __init__(self) -> None:
        self.selected_table = 1
        self.selected_item_index = 0
        self.selected_quantity = 1
        self._commands: Queue[dict] = Queue()
        self._lock = Lock()
        self._telemetry = {
            "system": "READY",
            "episode": 0,
            "mission": "WAITING FOR ORDER",
            "order_id": "-",
            "table_id": "-",
            "item": "-",
            "quantity": "-",
            "order_status": "IDLE",
            "phase": "IDLE",
            "speed_mps": 0.0,
            "distance": 0.0,
            "human": "NONE",
            "human_clearance": 99.0,
            "ttc": 99.0,
            "planner": "NOMINAL",
            "queue_depth": 0,
            "step": 0,
            "sim_time": 0.0,
            "outcome": "-",
            "people": 4,
        }
        self._history = deque(maxlen=5)
        self._plugin = None

    # ------------------------------------------------------------------
    # Main-thread API
    # ------------------------------------------------------------------
    def next_command(self) -> dict | None:
        try:
            return self._commands.get_nowait()
        except Empty:
            return None

    def update(self, **values) -> None:
        with self._lock:
            self._telemetry.update(values)

    def set_queue_depth(self, value: int) -> None:
        self.update(queue_depth=max(0, int(value)))

    def add_history(self, text: str) -> None:
        with self._lock:
            self._history.appendleft(str(text))

    def snapshot(self) -> tuple[dict, list[str]]:
        with self._lock:
            return deepcopy(self._telemetry), list(self._history)

    # ------------------------------------------------------------------
    # Viewer-thread API
    # ------------------------------------------------------------------
    def attach(self, scene) -> None:
        try:
            from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin
        except Exception as exc:  # pragma: no cover - depends on Genesis install
            raise RuntimeError(
                "Genesis in-viewer GUI is unavailable. Install/upgrade with: "
                'pip install -U "genesis-world[render]"'
            ) from exc

        viewer = getattr(scene, "viewer", None)
        if viewer is None:
            visualizer = getattr(scene, "visualizer", None)
            viewer = getattr(visualizer, "viewer", None)
        if viewer is None:
            raise RuntimeError("Genesis viewer is not available; run with rendering enabled.")

        plugin = next(
            (p for p in viewer.plugins if isinstance(p, ImGuiOverlayPlugin)),
            None,
        )
        if plugin is None:
            # This supports Genesis versions where enable_gui=False but plugins
            # can still be attached directly.
            plugin = viewer.add_plugin(ImGuiOverlayPlugin())
        plugin.register_panel(self._draw_panel)
        self._plugin = plugin

    def _queue_selected_order(self) -> None:
        order = {
            "command": "order",
            "table_id": int(self.selected_table),
            "item": str(MENU[self.selected_item_index]),
            "quantity": int(self.selected_quantity),
        }
        self._commands.put(order)

    def _draw_panel(self, imgui) -> None:
        telemetry, history = self.snapshot()

        imgui.text("ROBO SERVE - RESTAURANT DELIVERY")
        imgui.separator()
        imgui.text("ORDER FOOD")
        imgui.text(f"Target table: {self.selected_table}")
        if imgui.button("Previous table"):
            self.selected_table = 8 if self.selected_table <= 1 else self.selected_table - 1
        if imgui.button("Next table"):
            self.selected_table = 1 if self.selected_table >= 8 else self.selected_table + 1

        imgui.text(f"Food: {MENU[self.selected_item_index]}")
        if imgui.button("Previous food"):
            self.selected_item_index = (self.selected_item_index - 1) % len(MENU)
        if imgui.button("Next food"):
            self.selected_item_index = (self.selected_item_index + 1) % len(MENU)

        imgui.text(f"Quantity: {self.selected_quantity}")
        if imgui.button("Quantity -"):
            self.selected_quantity = max(1, self.selected_quantity - 1)
        if imgui.button("Quantity +"):
            self.selected_quantity = min(3, self.selected_quantity + 1)

        if imgui.button("ADD ORDER / START DELIVERY"):
            self._queue_selected_order()
        if imgui.button("CANCEL CURRENT MISSION"):
            self._commands.put({"command": "cancel"})

        imgui.separator()
        imgui.text("LIVE ROBOT INFORMATION")
        imgui.text(f"System: {telemetry['system']}")
        imgui.text(f"Episode: {telemetry['episode']}")
        imgui.text(f"Mission: {telemetry['mission']}")
        imgui.text(f"Queue: {telemetry['queue_depth']}")
        imgui.text(f"Order: {telemetry['order_id']}")
        imgui.text(f"Target: Table {telemetry['table_id']}")
        imgui.text(f"Food: {telemetry['item']} x{telemetry['quantity']}")
        imgui.text(f"Order status: {telemetry['order_status']}")
        imgui.text(f"Navigation: {telemetry['phase']}")
        imgui.text(f"Robot speed: {float(telemetry['speed_mps']):.2f} m/s")
        imgui.text(f"Distance to target: {float(telemetry['distance']):.2f} m")
        imgui.text(f"Nearest human: {telemetry['human']}")
        clearance = float(telemetry['human_clearance'])
        imgui.text(
            "Human clearance: CLEAR"
            if clearance > 20.0
            else f"Human clearance: {clearance:.2f} m"
        )
        ttc = float(telemetry['ttc'])
        imgui.text("TTC: -" if ttc > 20.0 else f"TTC: {ttc:.2f} s")
        imgui.text(f"Local planner: {telemetry['planner']}")
        imgui.text(f"Step: {telemetry['step']}   Sim time: {float(telemetry['sim_time']):.2f} s")
        imgui.text(f"Outcome: {telemetry['outcome']}")

        imgui.separator()
        imgui.text("RECENT DELIVERIES")
        if history:
            for line in history:
                imgui.text(line)
        else:
            imgui.text("No completed delivery yet.")
