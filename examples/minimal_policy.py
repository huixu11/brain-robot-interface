from __future__ import annotations

import sys
import time
from select import select
import termios
import threading
import tty

from bri import Action, Controller


def keyboard_policy_loop(ctrl: Controller, stop_event: threading.Event) -> None:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            rlist, _, _ = select([sys.stdin], [], [], 0.05)
            if not rlist:
                continue
            ch = sys.stdin.read(1)
            if ch == "\x03":  # Ctrl-C
                stop_event.set()
                break
            if ch in ("w", "W"):
                ctrl.set_action(Action.FORWARD)
            elif ch in ("a", "A"):
                ctrl.set_action(Action.LEFT)
            elif ch in ("d", "D"):
                ctrl.set_action(Action.RIGHT)
            elif ch in ("s", "S", " "):
                ctrl.set_action(Action.STOP)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main() -> None:
    ctrl = Controller(backend="sim", hold_s=0.3)
    ctrl.start()

    stop_event = threading.Event()
    thread = threading.Thread(target=keyboard_policy_loop, args=(ctrl, stop_event), daemon=True)
    thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()

