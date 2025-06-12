#!/usr/bin/env python3
"""
Crazyflie autonomous flight sequence using the **High-Level Commander (HLC)**
--------------------------------------------------------------------------
This script demonstrates how to control a Crazyflie with the *High-Level* API
(via `cf.high_level_commander`). The sequence:

1. Connect to the Crazyflie using `SyncCrazyflie`.
2. Enable the high-level commander (`commander.enHighLevel = 1`).
3. **Take-off** to 0.4 m.
4. Fly a 0.5 m square (X-Y plane, constant heading).
5. Approximate a circle (0.4 m radius) with 8 smooth `go_to()` segments.
6. **Land** and disconnect.

Usage (example):
    python crazyflie_hlc_sequence.py radio://0/80/2M/E7E7E7E7E7

Requirements:
    pip install cflib

Make sure you have a positioning system (Flow, Lighthouse, Loco, MoCap) since
High-Level Commander works in *position* mode.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Iterable

from cflib.crtp import init_drivers
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
URI_HELP = "Crazyflie URI, e.g. radio://0/80/2M/E7E7E7E7E7"
HEIGHT_M = 0.4         # Take-off height
VEL_M_S = 0.5          # Nominal path velocity (used for timing only)
SQUARE_SIDE_M = 0.5    # Square side length
CIRCLE_RADIUS_M = 0.4  # Circle radius
CIRCLE_SEGMENTS = 8

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def wait(seconds: float) -> None:
    """Busy-sleep helper so Ctrl-C immediately interrupts `time.sleep`."""
    end = time.time() + seconds
    while time.time() < end:
        time.sleep(0.01)


def square_vectors(side: float) -> Iterable[tuple[float, float]]:
    """Generate 4 relative (dx, dy) moves forming a square."""
    return (
        (side, 0.0),
        (0.0, side),
        (-side, 0.0),
        (0.0, -side),
    )


def circle_vectors(radius: float, segments: int) -> Iterable[tuple[float, float]]:
    """Return *segments* small relative vectors approximating a circle."""
    prev_x, prev_y = radius, 0.0  # start on +X axis
    step = 2 * math.pi / segments
    for i in range(1, segments + 1):
        theta = i * step
        x, y = radius * math.cos(theta), radius * math.sin(theta)
        yield x - prev_x, y - prev_y
        prev_x, prev_y = x, y


# ---------------------------------------------------------------------------
# Main flight routine
# ---------------------------------------------------------------------------

def fly(uri: str) -> None:
    print("[INFO] Initialising CRTP drivers …")
    init_drivers(enable_debug_driver=False)

    cf = Crazyflie(rw_cache="./cache")
    print(f"[INFO] Connecting to {uri} …")

    with SyncCrazyflie(uri, cf=cf) as scf:
        # Ensure we are in high-level (position) mode
        scf.cf.param.set_value("commander.enHighLevel", "1")
        wait(0.1)

        hlc = scf.cf.high_level_commander

        # --------------------- TAKE-OFF ---------------------
        print("[INFO] Take-off …")
        hlc.takeoff(HEIGHT_M, 2.0)
        wait(3.0)  # extra stabilisation time

        # # --------------------- SQUARE -----------------------
        # print("[INFO] Flying square …")
        # for dx, dy in square_vectors(SQUARE_SIDE_M):
        #     hlc.go_to(dx, dy, 0.0, yaw=0.0, duration_s=SQUARE_SIDE_M / VEL_M_S, relative=True)
        #     wait(SQUARE_SIDE_M / VEL_M_S + 0.3)

        # # --------------------- CIRCLE -----------------------
        # print("[INFO] Flying circle …")
        # for dx, dy in circle_vectors(CIRCLE_RADIUS_M, CIRCLE_SEGMENTS):
        #     hlc.go_to(dx, dy, 0.0, yaw=0.0, duration_s=(2 * math.pi * CIRCLE_RADIUS_M / VEL_M_S) / CIRCLE_SEGMENTS, relative=True)
        #     wait((2 * math.pi * CIRCLE_RADIUS_M / VEL_M_S) / CIRCLE_SEGMENTS + 0.05)

        # # --------------------- LAND -------------------------
        # print("[INFO] Landing …")
        # hlc.land(0.02, 2.0)  # land to 2 cm above ground
        # wait(3.0)
        # hlc.stop()

    print("[INFO] Flight finished. Disconnected.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Crazyflie HLC autonomous flight")
    parser.add_argument("uri", help=URI_HELP)
    args = parser.parse_args(argv)

    try:
        fly(args.uri)
    except KeyboardInterrupt:
        print("[WARN] Interrupted by user — attempting to stop and land.")
        # If we were still inside the context managers the CF would land already
        sys.exit(1)


if __name__ == "__main__":
    main()
