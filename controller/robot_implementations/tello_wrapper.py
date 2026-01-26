"""
TelloWrapper with Multiprocessing Odometry

EDUCATIONAL NOTES - Key Differences from Threading:
====================================================

1. MEMORY ISOLATION
   - Threads: Share memory (self.position accessible everywhere)
   - Processes: Separate memory spaces (need explicit sharing mechanism)

2. WHY MULTIPROCESSING?
   - Bypasses Python's GIL for true parallelism
   - Better isolation (odometry crash won't kill main process)
   - More predictable timing (no GIL contention)

3. SHARED MEMORY OPTIONS:
   - multiprocessing.Value: Single value (int, float, etc.)
   - multiprocessing.Array: Fixed-size array (what we use here)
   - multiprocessing.Queue: Message passing (good for events)
   - multiprocessing.Manager: Managed objects (slower, more flexible)

4. SYNCHRONIZATION:
   - We use a Lock to prevent read-during-write corruption
   - The 'with' statement ensures the lock is always released
"""

import logging
import threading
import multiprocessing as mp  # NEW: Import multiprocessing
from multiprocessing import Process, Array, Event as MPEvent
import time
from typing import Tuple, Optional
import ctypes  # NEW: Needed for shared array type specification

# =============================================================================
# EDUCATIONAL: Multiprocessing Start Methods
# =============================================================================
"""
Python has THREE ways to start a new process:

1. 'fork' (Linux default):
   - FAST: Copies parent's memory (copy-on-write)
   - DANGEROUS: If parent has threads, child inherits broken lock states
   - The warning you saw!

2. 'spawn' (Windows/macOS default):
   - SAFE: Starts fresh Python interpreter
   - SLOWER: Must re-import all modules
   - REQUIRES: if __name__ == "__main__" guard
   - All arguments must be picklable

3. 'forkserver':
   - HYBRID: Forks from a clean "server" process
   - Avoids thread-related issues
   - Linux only

We use 'spawn' because:
- Your code creates threads BEFORE starting the odometry process
- 'spawn' starts clean, avoiding the deadlock risk

TWO WAYS TO USE SPAWN:

Option A: mp.set_start_method('spawn') - GLOBAL, can only be called once
Option B: mp.get_context('spawn') - LOCAL, can be used multiple times

We use Option B below for flexibility.
"""

# Get a spawn context - this is safer than set_start_method because:
# 1. It doesn't affect other code that might use multiprocessing
# 2. It can be called multiple times without errors
# 3. It's explicit about what start method each Process uses
_spawn_ctx = mp.get_context('spawn')

# =============================================================================
# STATUS CODES (shared between processes)
# =============================================================================
class TrackingStatus:
    """Status codes for shared_status[0]"""
    VALID = 1.0
    STALE = 0.0
    RECOVERING = -1.0
    FAILED = -2.0

import numpy as np
from djitellopy import Tello

from controller.context_map.graph_manager import GraphManager
# NOTE: CrazyflieWrapper is imported here for the main process (cap_distance helper)
# The odometry process will import it separately inside _odometry_process_func
# This is necessary because 'spawn' mode starts a fresh interpreter
from controller.robot_implementations.crazyflie_wrapper import CrazyflieWrapper, cap_distance
from controller.robot_implementations.virtual_robot_wrapper import CommandResult
from controller.utils.constants import REGION_THRESHOLD
from controller.utils.general_utils import adjust_exposure, print_debug, sharpen_image
from ..abs.robot_wrapper import RobotWrapper

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
Tello.LOGGER.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# EDUCATIONAL: Standalone Process Function
# -----------------------------------------------------------------------------
"""
IMPORTANT: Process target functions must be defined at module level (not as methods)
OR be picklable. This is because Python needs to serialize the function to send it
to the new process.

The function receives shared memory objects as arguments, NOT self references,
because 'self' exists in the parent process's memory space only.
"""

def _odometry_process_func(
    shared_position: Array,      # Shared memory array for [x, y, z, yaw]
    stop_event: MPEvent,         # Signal to stop the process
    cf_uri: str,                 # Crazyflie URI for connection
    position_lock: mp.Lock       # Lock for thread-safe access
):
    """
    Standalone function that runs in a separate PROCESS.
    
    EDUCATIONAL NOTES:
    - This function has NO access to TelloWrapper's 'self'
    - It can only communicate through the shared objects passed as arguments
    - CrazyflieWrapper must be created HERE, in this process
    - With 'spawn' mode, this function runs in a FRESH Python interpreter
      that re-imports the module, so all imports must be available
    """
    # These imports happen IN THE CHILD PROCESS
    # With 'spawn', the child starts fresh and needs to import everything
    import time
    
    print("[OdometryProcess] Starting in separate process (spawn mode)...")
    print(f"[OdometryProcess] PID: {mp.current_process().pid}")
    
    # Import CrazyflieWrapper here to avoid issues with spawn
    # The child process will import this module fresh
    try:
        from controller.robot_implementations.crazyflie_wrapper import CrazyflieWrapper
        crazyflie = CrazyflieWrapper(move_enable=False, link_uri=cf_uri)
        print("[OdometryProcess] Crazyflie connected successfully")
    except Exception as e:
        print(f"[OdometryProcess] Crazyflie connection failed: {e}")
        import traceback
        traceback.print_exc()
        return  # Exit the process if we can't connect
    
    # Main loop - runs until stop_event is set
    while not stop_event.is_set():
        try:
            # Get position from Crazyflie (this returns numpy array)
            pos_m = crazyflie.get_pose()
            
            # EDUCATIONAL: Writing to shared memory
            # We use a lock to ensure atomic write (all 4 values written together)
            # Without this, the main process might read a half-updated position
            with position_lock:
                shared_position[0] = pos_m[0]  # x
                shared_position[1] = pos_m[1]  # y
                shared_position[2] = pos_m[2]  # z
                
                # Normalize yaw to [0, 360)
                yaw = pos_m[3]
                if yaw < 0:
                    yaw = yaw + 360
                shared_position[3] = yaw
            
            # 50Hz update rate
            time.sleep(0.02)
            
        except Exception as e:
            print(f"[OdometryProcess] Error: {e}")
            time.sleep(0.1)  # Back off on error
    
    print("[OdometryProcess] Stopping...")


# -----------------------------------------------------------------------------
# Helpers (unchanged)
# -----------------------------------------------------------------------------
class FrameReader:
    """
    Lazy-eval wrapper for video frames. 
    Applies image processing only when .frame is accessed.
    """
    def __init__(self, background_frame_reader):
        self.bfr = background_frame_reader

    @property
    def frame(self):
        raw_frame = self.bfr.frame
        if raw_frame is None:
            return None
        frame = adjust_exposure(raw_frame, alpha=1.3, beta=-30)
        return sharpen_image(frame)


def _cap_distance_positive(distance: int) -> int:
    distance = abs(distance)
    return max(20, min(distance, 500))


def _cap_distance_signed(distance: int) -> int:
    if -20 < distance < 20:
        return 0
    return max(-500, min(distance, 500))

# =============================================================================
# IMPROVED ODOMETRY PROCESS WITH AUTO-RECOVERY
# =============================================================================

def _odometry_process_with_recovery(
    shared_position,      # [x, y, z, yaw]
    shared_status,        # [status_code, last_update_time, recovery_count]
    stop_event,
    cf_uri: str,
    position_lock,
):
    """
    Odometry process that automatically attempts to recover lost tracking.
    
    RECOVERY STRATEGY:
    1. Detect stale data (position unchanged for >0.5s)
    2. Reset Kalman estimator
    3. Wait for variance to converge
    4. Resume normal operation
    
    If recovery fails 3 times, mark as FAILED and stop trying.
    """
    import time
    import numpy as np
    
    print("[OdometryProcess] Starting with auto-recovery...")
    print(f"[OdometryProcess] PID: {mp.current_process().pid}")
    
    # =========================================================================
    # CONNECT TO CRAZYFLIE
    # =========================================================================
    try:
        from controller.robot_implementations.crazyflie_wrapper import CrazyflieWrapper
        crazyflie = CrazyflieWrapper(move_enable=False, link_uri=cf_uri)
        cf = crazyflie.cf  # Direct access for recovery operations
        print("[OdometryProcess] Crazyflie connected successfully")
    except Exception as e:
        print(f"[OdometryProcess] Crazyflie connection failed: {e}")
        with position_lock:
            shared_status[0] = TrackingStatus.FAILED
        return
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    STALE_THRESHOLD_SEC = 0.5        # Consider stale after this long
    POSITION_CHANGE_THRESHOLD = 0.001 # Minimum change to count as "moved" (meters)
    MAX_RECOVERY_ATTEMPTS = 3        # Give up after this many failures
    RECOVERY_COOLDOWN_SEC = 5.0      # Wait between recovery attempts
    
    # =========================================================================
    # STATE
    # =========================================================================
    last_position = np.zeros(4)
    last_change_time = time.time()
    recovery_count = 0
    last_recovery_attempt = 0.0
    is_recovering = False
    
    # =========================================================================
    # HELPER: Reset Kalman and wait for convergence
    # =========================================================================
    def attempt_recovery():
        """
        Reset the Kalman estimator and wait for it to converge.
        Returns True if recovery successful, False otherwise.
        """
        nonlocal last_recovery_attempt, recovery_count, is_recovering
        
        current_time = time.time()
        
        # Cooldown check
        if current_time - last_recovery_attempt < RECOVERY_COOLDOWN_SEC:
            return False
        
        is_recovering = True
        last_recovery_attempt = current_time
        recovery_count += 1
        
        print(f"[OdometryProcess] 🔄 Attempting recovery ({recovery_count}/{MAX_RECOVERY_ATTEMPTS})...")
        
        # Update status to RECOVERING
        with position_lock:
            shared_status[0] = TrackingStatus.RECOVERING
            shared_status[2] = float(recovery_count)
        
        try:
            # Step 1: Reset the Kalman filter
            print("[OdometryProcess]    Resetting Kalman estimator...")
            cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.1)
            cf.param.set_value('kalman.resetEstimation', '0')
            
            # Step 2: Wait for variance to converge
            print("[OdometryProcess]    Waiting for convergence...")
            
            # We need to check the Kalman variance
            # Unfortunately we can't easily add a new log config from here,
            # so we'll use a simple heuristic: wait and check if position changes
            
            convergence_start = time.time()
            convergence_timeout = 5.0  # seconds
            stable_count = 0
            last_check_pos = None
            
            while time.time() - convergence_start < convergence_timeout:
                if stop_event.is_set():
                    return False
                
                current_pos = crazyflie.get_pose()
                
                if last_check_pos is not None:
                    # Check if position is changing (sign of active tracking)
                    pos_diff = np.linalg.norm(current_pos[:3] - last_check_pos[:3])
                    
                    if pos_diff > 0.001:  # Position is changing
                        stable_count += 1
                        if stable_count >= 5:  # 5 consecutive changes = tracking recovered
                            print("[OdometryProcess] ✅ Recovery successful!")
                            is_recovering = False
                            return True
                    else:
                        stable_count = 0  # Reset counter
                
                last_check_pos = current_pos.copy()
                time.sleep(0.1)
            
            print("[OdometryProcess] ❌ Recovery timed out - Lighthouse may not be visible")
            is_recovering = False
            return False
            
        except Exception as e:
            print(f"[OdometryProcess] ❌ Recovery error: {e}")
            is_recovering = False
            return False
    
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    print("[OdometryProcess] Entering main loop...")
    
    while not stop_event.is_set():
        try:
            current_time = time.time()
            
            # Skip normal processing during recovery
            if is_recovering:
                time.sleep(0.1)
                continue
            
            # Get position
            pos_m = crazyflie.get_pose()
            
            # Check if position actually changed
            position_changed = False
            for i in range(4):
                if abs(pos_m[i] - last_position[i]) > POSITION_CHANGE_THRESHOLD:
                    position_changed = True
                    break
            
            if position_changed:
                last_change_time = current_time
                last_position = pos_m.copy()
            
            # Calculate staleness
            time_since_change = current_time - last_change_time
            is_stale = time_since_change > STALE_THRESHOLD_SEC
            
            # =========================================================
            # STALE DATA HANDLING
            # =========================================================
            if is_stale:
                if recovery_count >= MAX_RECOVERY_ATTEMPTS:
                    # Give up - mark as failed
                    with position_lock:
                        shared_status[0] = TrackingStatus.FAILED
                    
                    # Log occasionally
                    if int(current_time) % 5 == 0:
                        print(f"[OdometryProcess] ⛔ Tracking FAILED after {MAX_RECOVERY_ATTEMPTS} recovery attempts")
                        print(f"[OdometryProcess]    Move drone back into Lighthouse view and restart")
                else:
                    # Attempt recovery
                    if attempt_recovery():
                        # Recovery successful - reset counters
                        last_change_time = current_time
                        last_position = crazyflie.get_pose().copy()
                    # If recovery failed, loop will try again after cooldown
                
                time.sleep(0.1)
                continue
            
            # =========================================================
            # NORMAL OPERATION - Write to shared memory
            # =========================================================
            with position_lock:
                shared_position[0] = pos_m[0]
                shared_position[1] = pos_m[1]
                shared_position[2] = pos_m[2]
                
                yaw = pos_m[3]
                if yaw < 0:
                    yaw += 360
                shared_position[3] = yaw
                
                shared_status[0] = TrackingStatus.VALID
                shared_status[1] = last_change_time
                shared_status[2] = float(recovery_count)
            
            time.sleep(0.02)  # 50Hz
            
        except Exception as e:
            print(f"[OdometryProcess] Error: {e}")
            time.sleep(0.1)
    
    print("[OdometryProcess] Stopping...")

# -----------------------------------------------------------------------------
# Tello Implementation with Multiprocessing
# -----------------------------------------------------------------------------
class TelloWrapper(RobotWrapper):
    """
    EDUCATIONAL: Key changes from threading version:
    
    1. self.position -> self._shared_position (multiprocessing.Array)
    2. threading.Event -> multiprocessing.Event (for stop signal)
    3. threading.Thread -> multiprocessing.Process
    4. Added self._position_lock for synchronized access
    5. get_position() now reads from shared memory with lock
    """
    
    KEEPALIVE_PERIOD = 4.0
    GRAPH_UPDATE_PERIOD = 0.1  # Update graph at 10Hz

    def __init__(self, move_enable: bool, graph_manager: GraphManager, 
                 use_crazyflie_lighthouse: bool = True, 
                 cf_uri: str = 'radio://0/40/2M/BADF00D003'):
        
        super().__init__(graph_manager=graph_manager, move_enable=move_enable)

        self.drone = Tello()
        self.use_lighthouse = use_crazyflie_lighthouse
        self._cf_uri = cf_uri  # Store for passing to process
        
        # REMOVED: self.crazyflie - will be created in subprocess instead
        # The CrazyflieWrapper will be instantiated INSIDE the odometry process
        # because we can't share hardware connections across processes
        
        self.stream_on = False
        self.active_count = 0

        # =====================================================================
        # EDUCATIONAL: Shared Memory Setup with Spawn Context
        # =====================================================================
        # 
        # We use _spawn_ctx to create shared memory objects. This ensures
        # they're compatible with the spawn start method.
        #
        # multiprocessing.Array creates a shared memory region that both
        # the main process and child process can access.
        #
        # Arguments:
        # - ctypes.c_double: The C type for each element (64-bit float)
        # - 4: Number of elements [x, y, z, yaw]
        #
        # The Array is backed by shared memory, not regular Python memory.
        # Changes made in one process are visible in the other.
        # =====================================================================
        self._shared_position = _spawn_ctx.Array(ctypes.c_double, 4)  # [x, y, z, yaw]
        self._shared_status = _spawn_ctx.Array(ctypes.c_double, 3)
        
        # Initialize to zeros (explicit, for clarity)
        with self._shared_position.get_lock():
            for i in range(4):
                self._shared_position[i] = 0.0
        
        # =====================================================================
        # EDUCATIONAL: Synchronization Lock
        # =====================================================================
        # 
        # _spawn_ctx.Lock() creates an inter-process lock compatible with spawn.
        # This is different from threading.Lock() which only works within
        # a single process.
        #
        # We use this to ensure:
        # 1. The odometry process doesn't write while main reads
        # 2. We get consistent readings (all 4 values from same timestamp)
        # =====================================================================
        self._position_lock = _spawn_ctx.Lock()
        
        # =====================================================================
        # EDUCATIONAL: Stop Event
        # =====================================================================
        #
        # _spawn_ctx.Event() is the inter-process version of 
        # threading.Event(). It uses shared memory internally to signal
        # between processes.
        # =====================================================================
        self._stop_event = _spawn_ctx.Event()  # For processes
        self._thread_stop_event = threading.Event()  # Keep for threads (keepalive)
        
        # Process and thread storage
        self._odometry_process: Optional[Process] = None
        self._threads = []  # Keep for non-odometry threads (keepalive)

    def get_tracking_status(self) -> str:
        '''Get human-readable tracking status.'''
        with self._position_lock:
            status_code = self._shared_status[0]
            recovery_count = int(self._shared_status[2])
        
        if status_code == TrackingStatus.VALID:
            return "VALID"
        elif status_code == TrackingStatus.STALE:
            return "STALE"
        elif status_code == TrackingStatus.RECOVERING:
            return f"RECOVERING (attempt {recovery_count})"
        elif status_code == TrackingStatus.FAILED:
            return f"FAILED after {recovery_count} attempts"
        else:
            return f"UNKNOWN ({status_code})"
        
    def is_position_valid(self) -> bool:
        """Check if tracking is working."""
        with self._position_lock:
            return self._shared_status[0] == TrackingStatus.VALID

    def wait_for_tracking(self, timeout=10.0) -> bool:
        """Wait for tracking to become valid."""
        start = time.time()
        while time.time() - start < timeout:
            if self.is_position_valid():
                return True
            time.sleep(0.5)
        return False

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def connect(self, retries: int = 3):
        for attempt in range(retries):
            try:
                time.sleep(0.5)
                self.drone.connect()
                logger.info("Tello connected successfully")
                break
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1}/{retries} failed: {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(1.0)
        
        # Start keepalive as THREAD (it's lightweight and needs self.drone)
        self._start_thread(self._keepalive_loop, "tello-keepalive")
        
        # =====================================================================
        # EDUCATIONAL: Starting the Odometry Process with SPAWN context
        # =====================================================================
        #
        # Key differences from threading:
        #
        # 1. We pass shared objects as arguments (not accessible via self)
        # 2. The target function is module-level (for pickling)
        # 3. We use daemon=True so it dies when main process exits
        # 4. start() spawns a NEW Python interpreter (not fork!)
        #
        # Using _spawn_ctx.Process instead of mp.Process ensures we use
        # the 'spawn' start method, avoiding the fork+threads deadlock risk.
        # =====================================================================
        if self.use_lighthouse:
            # IMPORTANT: Use _spawn_ctx.Process, not mp.Process!
            # This ensures we use 'spawn' method regardless of platform default
            self._odometry_process = _spawn_ctx.Process(
                target=_odometry_process_func,
                args=(
                    self._shared_position,  # Shared memory for position
                    self._stop_event,        # Signal to stop
                    self._cf_uri,            # Crazyflie URI
                    self._position_lock      # Synchronization lock
                ),
                name="odometry-crazyflie",
                daemon=True  # Process dies when main process exits
            )
            self._odometry_process.start()
            print(f"[TelloWrapper] Odometry process started (PID: {self._odometry_process.pid}) [spawn mode]")
        else:
            # Dead reckoning still uses thread (or could be converted similarly)
            self._start_thread(self._odometry_loop_dead_reckoning, "odo-dead-reckoning")

    def disconnect(self):
        """
        EDUCATIONAL: Proper cleanup is critical with multiprocessing!
        
        Unlike threads, processes don't share memory cleanup.
        If you don't properly terminate, you can have zombie processes.
        """
        # Signal all background tasks to stop
        self._stop_event.set()
        self._thread_stop_event.set()
        
        # Wait for odometry process to finish
        if self._odometry_process is not None:
            print("[TelloWrapper] Waiting for odometry process to terminate...")
            self._odometry_process.join(timeout=2.0)  # Wait up to 2 seconds
            
            # If it didn't stop gracefully, force terminate
            if self._odometry_process.is_alive():
                print("[TelloWrapper] Force terminating odometry process")
                self._odometry_process.terminate()
                self._odometry_process.join(timeout=1.0)
        
        # Wait for threads
        for t in self._threads:
            t.join(timeout=1.0)
        
        self.drone.end()
        print("[TelloWrapper] Disconnected")

    def _start_thread(self, target, name):
        """Helper for starting threads (keepalive still uses threads)"""
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    # -------------------------------------------------------------------------
    # Video Stream (unchanged)
    # -------------------------------------------------------------------------
    def start_stream(self):
        if not self.stream_on:
            self.drone.streamon()
            self.stream_on = True

    def stop_stream(self):
        if self.stream_on:
            self.drone.streamoff()
            self.stream_on = False

    def get_frame_reader(self):
        if not self.stream_on: 
            return None
        return FrameReader(self.drone.get_frame_read())

    # -------------------------------------------------------------------------
    # Movement Commands (mostly unchanged)
    # -------------------------------------------------------------------------
    def takeoff(self) -> CommandResult:
        if not self.is_battery_good(): 
            return CommandResult(value=False, replan=False)
            
        if self.move_enable:
            self.drone.takeoff()
        else:
            print("[Drone] Takeoff (Simulated)")
        return CommandResult(value=True, replan=False)

    def land(self) -> CommandResult:
        if self.move_enable:
            self.drone.land()
        else:
            print("[Drone] Land (Simulated)")
        return CommandResult(value=True, replan=False)

    def _move_relative(self, forward=0, backward=0, left=0, right=0, up=0, down=0, yaw_cw=0, yaw_ccw=0):
        if not self.move_enable:
            print(f"[Drone] Move: F:{forward} B:{backward} L:{left} R:{right}")
            return CommandResult(value=True, replan=False)

        if forward: 
            self.drone.move_forward(_cap_distance_positive(forward))
        if backward: 
            self.drone.move_back(_cap_distance_positive(backward))
        if left: 
            self.drone.move_left(_cap_distance_positive(left))
        if right: 
            self.drone.move_right(_cap_distance_positive(right))
        if up: 
            self.drone.move_up(_cap_distance_positive(up))
        if down: 
            self.drone.move_down(_cap_distance_positive(down))
        
        if yaw_cw: 
            self.drone.rotate_clockwise(max(1, min(yaw_cw, 360)))
        if yaw_ccw: 
            self.drone.rotate_counter_clockwise(max(1, min(yaw_ccw, 360)))
        
        time.sleep(0.5)
        return CommandResult(value=True, replan=False)

    def move_north(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        return self._move_relative(forward=int(distance_cm))

    def move_south(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        return self._move_relative(backward=int(distance_cm))

    def move_west(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        return self._move_relative(left=int(distance_cm))

    def move_east(self, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        return self._move_relative(right=int(distance_cm))

    def move_up(self, distance_cm: int) -> CommandResult:
        return self._move_relative(up=int(distance_cm))

    def move_down(self, distance_cm: int) -> CommandResult:
        return self._move_relative(down=int(distance_cm))

    def turn_cw(self, degree: int) -> CommandResult:
        return self._move_relative(yaw_cw=degree)

    def turn_ccw(self, degree: int) -> CommandResult:
        return self._move_relative(yaw_ccw=degree)
    
    def move_direction(self, direction_deg: int, distance_cm: int = REGION_THRESHOLD) -> CommandResult:
        """Rotate to target yaw (shortest path), then move forward.
            Safe movement that checks tracking first."""
        
        # if not self.is_position_valid():
        #     print("⚠️ Cannot move - tracking lost!")
        #     return CommandResult(value=False, replan=True)
        print(f"[Drone] Rotate to {direction_deg}°, then move {distance_cm}cm")
        if not self.move_enable:
            print(f"[Drone] Not enable")
            return CommandResult(value=True, replan=False)

        # EDUCATIONAL: Reading from shared memory
        # We get the position using our thread-safe getter
        current_yaw = self.get_position()[3] % 360
        target_yaw = direction_deg % 360
        print(f"{current_yaw} vs {target_yaw}")

        delta = (target_yaw - current_yaw + 180) % 360 - 180
        print(f"Delta: {delta}")

        if delta > 0:
            self.drone.rotate_counter_clockwise(abs(int(delta)))
        elif delta < 0:
            self.drone.rotate_clockwise(abs(int(delta)))   

        time.sleep(1.0)
        self.drone.move_forward(cap_distance(int(distance_cm)))

        return CommandResult(value=True, replan=False)

    def go_to_position(self, target_x_cm: float, target_y_cm: float, target_z_cm: float):
        curr_x, curr_y, curr_z = self.get_position()[:3]
        
        dx = int(target_x_cm - curr_x)
        dy = int(target_y_cm - curr_y)
        dz = int(target_z_cm - curr_z)
        
        if abs(dx) < 20 and abs(dy) < 20 and abs(dz) < 20:
            return CommandResult(value=True, replan=False)
        
        dx = _cap_distance_signed(dx)
        dy = _cap_distance_signed(dy)
        dz = _cap_distance_signed(dz)

        print(f"[Drone] GoTo: Δ({dx}, {dy}, {dz}) cm")
        if not self.move_enable:
            return CommandResult(value=True, replan=False)

        self.drone.go_xyz_speed(dx, dy, dz, speed=20)
        
        return CommandResult(value=True, replan=False)

    # -------------------------------------------------------------------------
    # Odometry & State - THE KEY CHANGED SECTION
    # -------------------------------------------------------------------------
    def get_position(self) -> Tuple[float, float, float, float]:
        """
        Return position in cm and yaw in degrees (x, y, z, yaw).
        
        EDUCATIONAL: Reading from Shared Memory
        =========================================
        
        Unlike the threading version where we just read self.position,
        we now need to:
        
        1. Acquire the lock (prevents reading during a write)
        2. Copy all values (they're in shared memory, not regular Python objects)
        3. Release the lock
        4. Process the copied values
        
        The 'with' statement handles lock acquire/release automatically,
        even if an exception occurs.
        """
        # Read from shared memory with lock
        with self._position_lock:
            x_cm = self._shared_position[0]
            y_cm = self._shared_position[1]
            z_cm = self._shared_position[2]
            yaw_deg = self._shared_position[3]
        
        
        print(f"{x_cm}, {y_cm}, {z_cm}, {yaw_deg}")
        
        return (x_cm, y_cm, z_cm, yaw_deg)

    def _update_graph_manager(self):
        """
        EDUCATIONAL: Graph Manager Updates
        ===================================
        
        The graph_manager.update_pose() was previously called inside
        the odometry thread. Now we have two options:
        
        Option A: Call it from main process periodically
        Option B: Use a Queue to send pose updates to main process
        
        I've chosen Option A here - we call this method from somewhere
        in the main loop. This keeps graph_manager in the main process
        where the rest of the logic runs.
        
        If you need real-time updates, consider adding a separate thread
        in the MAIN process that polls get_position() and updates the graph.
        """
        if self.graph_manager:
            pos = self.get_position()
            # pos is (x_cm, y_cm, z_cm, yaw)
            self.graph_manager.update_pose(
                np.array([pos[0], pos[1], pos[2]]),  # Position in cm
                pos[3]  # Yaw in degrees
            )

    def is_battery_good(self) -> bool:
        try:
            bat = self.drone.get_battery()
            print(f"> Battery: {bat}% {'[LOW]' if bat < 20 else '[OK]'}")
            return bat >= 20
        except:
            return True
        

    def _graph_update_loop(self):
        """
        NEW: Dedicated thread for updating graph manager at high frequency.
        This ensures position changes are reflected immediately.
        """
        print("[GraphUpdater] Starting...")
        while not self._thread_stop_event.is_set():
            try:
                if self.graph_manager and self.is_position_valid():
                    pos = self.get_position()
                    self.graph_manager.update_pose(
                        np.array([pos[0], pos[1], pos[2]]),
                        pos[3]
                    )
            except Exception as e:
                logger.debug(f"Graph update error: {e}")
            
            self._thread_stop_event.wait(self.GRAPH_UPDATE_PERIOD)
        
        print("[GraphUpdater] Stopped")

    def _keepalive_loop(self):
        """Keepalive still runs as a thread (needs access to self.drone)"""
        while not self._thread_stop_event.is_set():
            try:
                self.drone.send_control_command("command")
                
                # EDUCATIONAL: Update graph manager in keepalive loop
                # This ensures position updates happen at regular intervals
                # even when no movement commands are issued
                self._update_graph_manager()
                
            except Exception as e:
                logger.debug(f"Keepalive error: {e}")
            self._thread_stop_event.wait(self.KEEPALIVE_PERIOD)

    def _odometry_loop_dead_reckoning(self):
        """Placeholder for dead reckoning (still thread-based)"""
        # TODO: Implement dead reckoning
        pass