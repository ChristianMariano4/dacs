# Encode the image in base64
import base64
import csv
from io import BytesIO
import math


def encode_image(self, image):
    """Convert an image (PIL or numpy) to base64 string"""
    if isinstance(image, np.ndarray):
        # If it's an OpenCV image
        import cv2
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Could not encode numpy image")
        return base64.b64encode(buffer).decode("utf-8")
    elif isinstance(image, Image.Image):
        # If it's a PIL image
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise TypeError("Unsupported image type")
    
# ─────────────────────── Tajectories utils ───────────────────────

def save_waypoints_csv(self, waypoints, filename):
        """Save waypoints in format expected by trajectory optimizer"""
        try:
            with open(filename, 'w', newline='') as csvfile:
                csvfile.write('t,x,y,z,yaw\n')
                for wp in waypoints:
                    csvfile.write(f"{wp['t']:.3f},{wp['x']:.3f},{wp['y']:.3f},{wp['z']:.3f},{math.degrees(wp['yaw']):.1f}\n")
        except Exception as e:
            print(f"Failed to save waypoints CSV: {e}")


def process_trajectory(self, raw_waypoints):
    """Process raw waypoints using polynomial optimization"""
    if len(raw_waypoints) < 2:
        return raw_waypoints
    
    # Save raw waypoints to CSV for trajectory optimization
    save_waypoints_csv(raw_waypoints, 'my_timed_waypoints_yaw.csv')

    # Run trajectory optimization (similar to original Crazyflie code)
    import subprocess
    import os
    
    try:
        # Run the trajectory generation script
        cmd = ['python3', 'controller/generate_trajectory.py', 
            'my_timed_waypoints_yaw.csv', 'traj.csv', '--pieces', '5']
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Trajectory optimization failed: {result.stderr}")
            return fallback_processing(raw_waypoints)
        
        # Load optimized trajectory
        optimized_trajectory = load_optimized_trajectory('traj.csv')
        print(f"Trajectory optimized: {len(raw_waypoints)} waypoints -> polynomial trajectory")
        return optimized_trajectory
    except Exception as e:
        print(f"Trajectory optimization error: {e}")
        return fallback_processing(raw_waypoints)
    
def fallback_processing(raw_waypoints):
    """Fallback to simple processing if optimization fails"""
    processed = [raw_waypoints[0]] # Always keep first point

    for i in range(1, len(raw_waypoints) - 1):
        current = raw_waypoints[i]
        last_kept = processed[-1]

        # Calculate distance from last kept point
        dx = current['x'] - last_kept['x']
        dy = current['y'] - last_kept['y']
        dz = current['z'] - last_kept['z']
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Keep point if it's far enough from last kept point
        if distance > 0.1:  # 10cm minimum distance
            processed.append(current)

    # Always keep last point
    if len(raw_waypoints) > 1:
        processed.append(raw_waypoints[-1])

    print(f"Fallback processing: {len(raw_waypoints)} -> {len(processed)} waypoints")
    return processed

def load_optimized_trajectory(filename):
    """Load polynomial trajectory from CSV"""
    try:
        # Import the trajectory classes
        import controller.uav_trajectory as uav_trajectory

        # Load the trajectory
        traj = uav_trajectory.Trajectory()
        traj.loadcsv(filename)

        return traj
    except Exception as e:
        print(f"Failed to load optimized trajectory: {e}")
        return None
    
def save_trajectory_csv(trajectory, filename):
        """Save trajectory to CSV file"""
        if not trajectory:
            return
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['t', 'x', 'y', 'z', 'yaw']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for waypoint in trajectory:
                    writer.writerow(waypoint)
            print(f"Trajectory saved to {filename}")
        except Exception as e:
            print(f"Failed to save trajectory: {e}")