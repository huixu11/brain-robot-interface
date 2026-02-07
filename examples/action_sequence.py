from __future__ import annotations

import time

from bri import Action, Controller


def main() -> None:
    # NOTE: This is intentionally non-interactive and cross-platform.
    # It exists to validate the Action->CmdVel mapping (including BACKWARD) in sim.
    ctrl = Controller(backend="sim", hold_s=10.0)
    ctrl.start()
    try:
        sequence: list[tuple[Action, float]] = [
            (Action.FORWARD, 2.0),
            (Action.BACKWARD, 2.0),
            (Action.LEFT, 1.5),
            (Action.RIGHT, 1.5),
            (Action.STOP, 1.0),
        ]

        for action, duration_s in sequence:
            print(f"[sequence] action={action} duration_s={duration_s}")
            ctrl.set_action(action)
            time.sleep(duration_s)

        # Give a moment to observe STOP in the viewer.
        time.sleep(1.0)
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()

