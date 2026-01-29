
import math
import sys
import time, cv2
import logging
import numpy as np
from typing import Tuple

import controller.utils.general_utils as general_utils
from ..abs.robot_wrapper import RobotWrapper



import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.commander import Commander

from cflib.utils import uri_helper

# object detection screen dimension. note: (0,0) is the top-left screen corner
CENTER_SCREEN_X = 162.5
CENTER_SCREEN_Y = 122.5

# import logging
# Tello.LOGGER.setLevel(logging.WARNING)


MOVEMENT_MIN = 20
MOVEMENT_MAX = 300

SCENE_CHANGE_DISTANCE = 120
SCENE_CHANGE_ANGLE = 90

def adjust_exposure(img, alpha=1.0, beta=0):
    """
    Adjust the exposure of an image.
    
    :param img: Input image
    :param alpha: Contrast control (1.0-3.0). Higher values increase exposure.
    :param beta: Brightness control (0-100). Higher values add brightness.
    :return: Exposure adjusted image
    """
    # Apply exposure adjustment using the formula: new_img = img * alpha + beta
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new_img


import socket
import struct
import time
import numpy as np
import cv2

class AIDeckClient:
    def __init__(self, ip="192.168.4.1", port=5000):
        self.ip = ip
        self.port = port
        self.socket = None
        self.connect()

    def connect(self):
        # Open a TCP connection to the AI-deck, which streams image data continuously.
        print(f"Connecting to {self.ip}:{self.port}...")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        client_socket.connect((self.ip, self.port))
        print("Connection established.")

    # Ensure exactly size bytes are received from the socket
    def rx_bytes(self, size):
        data = bytearray()
        while len(data) < size:
            data.extend(self.socket.recv(size - len(data)))
        return data
    
    def receive_image_packet(self):
        # Get image packet header
        header = self.rx_bytes(4)
        #print(packetInfoRaw)

        length, routing, function = struct.unpack('<HBB', header)
        #print("Length is {}".format(length))
        #print("Route is 0x{:02X}->0x{:02X}".format(routing & 0xF, routing >> 4))
        #print("Function is 0x{:02X}".format(function))

        img_header = self.rx_bytes(length - 2)
        #print(img_header)
        #print("Length of data is {}".format(len(imgHeader)))
        magic, width, height, depth, img_format, size = struct.unpack('<BHHBBI', img_header)

        if magic != 0xBC:
            return None, None, None
        
        #print("Magic is good")
        #print("Resolution is {}x{} with depth of {} byte(s)".format(width, height, depth))
        #print("Image format is {}".format(format))
        #print("Image size is {} bytes".format(size))

        # Start receiving the image, packet by packet
        img_stream = bytearray()
        while len(img_stream) < size:
            packet_header = self.rx_bytes(4)
            chunk_length, dst, src = struct.unpack('<HBB', packet_header)
            #print("Chunk size is {} ({:02X}->{:02X})".format(length, src, dst))
            chunk = self.rx_bytes(chunk_length - 2)
            img_stream.extend(chunk)
        
        return img_stream, img_format, (width, height)
    
    def close(self):
        if self.socket:
            self.socket.close()
            print("Connection closed.")

class AIDeckViewer:
    def __init__(self, client: AIDeckClient, output_queue1, output_queue2):
        self.client = client
        self.output_queue1 = output_queue1
        self.output_queue2 = output_queue2
        self.fps_avg_frame_count = 10
    
    def display_loop(self):
        frame_count = 0
        start = time.time()

        try:
            while True:
                img_stream, img_format, _ = self.client.receive_image_packet()
                if img_stream is None:
                    continue

                if img_format == 0: # Image is in Bayer format (raw sensor data)
                    bayer_img = np.frombuffer(img_stream, dtype=np.uint8).reshape((244, 324))
                    color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2RGB)
                    cv2.imshow('Raw', bayer_img)
                    cv2.imshow('Color', color_img)
                    if False:
                        cv2.imwrite(f"stream_out/raw/img_{count:06d}.png", bayer_img)
                        cv2.imwrite(f"stream_out/debayer/img_{count:06d}.png", color_img)
                else: # Image is jpeg
                    # with open("img.jpeg", "wb") as f:
                    #     f.write(img_stream)
                    nparr = np.frombuffer(img_stream, np.uint8)
                    color_img = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
                    cv2.imshow('JPEG', color_img)
                
                cv2.waitKey(1)

                self.output_queue1.put(color_img)
                self.output_queue2.put(color_img)

                frame_count += 1

                # Compute average FPS every 10 frames
                if frame_count % self.fps_avg_frame_count == 0:
                    elapsed = time.time() - start
                    fps = self.fps_avg_frame_count / elapsed
                    start = time.time()
                    
                    fps_text = 'FPS = {:.1f}'.format(fps)
                    #print(fps_text)
    
        finally:
            print("Viewer stopped.")
            self.client.close()
            cv2.destroyAllWindows()

def sharpen_image(img):
    """
    Apply a sharpening filter to an image.
    
    :param img: Input image
    :return: Sharpened image
    """
    # Define a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    # Apply the sharpening filter
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

class FrameReader:
    def __init__(self, fr):
        # Initialize the video capture
        self.fr = fr

    @property
    def frame(self):
        # Read a frame from the video capture
        frame = self.fr.frame
        frame = adjust_exposure(frame, alpha=1.3, beta=-30)
        return sharpen_image(frame)
        
def cap_distance(distance: int) -> int:
    """
    Cap distance to Tello's limits, preserving sign.
    Tello go command: x,y,z = -500 to 500
    Cannot be between -20 and 20 simultaneously.
    """
    if -20 < distance < 20:
        return 0  # Too small, skip
    
    # Clamp to [-500, 500]
    if distance > 500:
        return 500
    if distance < -500:
        return -500
    
    return distance

class CrazyflieWrapper():
    def __init__(self, move_enable: bool = False, link_uri: str = 'radio://0/40/2M/BADF00D003'):
        # super().__init__(move_enable=move_enable)
        self.move_enable = False
        self.cf=Crazyflie(rw_cache='./cache')
        self.stream_on = False
        self.connected = False
        self.is_flying = False
        self.gesture_trajectory_mapping = {}
        self.link_uri = link_uri
        self.z_target = 0.5  # Default target height in meters
        self.pose = np.zeros(4)
        self.connect()
        self.odometry_file = open("odometry.txt", "a")

    def _log_stab_callback(self, timestamp, data, logconf):
        # print(f"[POSE CALLBACK] Received pose data at timestamp {timestamp}")
        x = data['stateEstimate.x']
        y = data['stateEstimate.y']
        z = data['stateEstimate.z']
        yaw = data['stateEstimate.yaw']
        
        self.pose[0] = x * 100
        self.pose[1] = y * 100
        self.pose[2] = (z - 0.05) * 100
        self.pose[3]= yaw % 360
        
        # print(f"[{timestamp}] x={self.pose[0]:.2f}, y={self.pose[1]:.2f}, z={self.pose[2]:.2f}, yaw={self.pose[3]:.2f}")

    def _track_position(self):
        '''Continuosly track drone position through lighthouse'''
        # Configure logging
        logconf = LogConfig(name='Position', period_in_ms=100)
        logconf.add_variable('stateEstimate.x', 'float')
        logconf.add_variable('stateEstimate.y', 'float')
        logconf.add_variable('stateEstimate.z', 'float')
        logconf.add_variable('stateEstimate.yaw', 'float')

        self.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(self._log_stab_callback)
        logconf.start()
    
    def get_pose(self):
        self.odometry_file.write(f"[{time.time()}] x={self.pose[0]:.2f}, y={self.pose[1]:.2f}, z={self.pose[2]:.2f}, yaw={self.pose[3]:.2f}\n")
        return self.pose

    def connect(self):
        cflib.crtp.init_drivers()
        self.scf = SyncCrazyflie(self.link_uri, cf=self.cf)
        
        try:
            self.scf.open_link()
            
            # Aspetta che la connessione sia stabile
            time.sleep(2.0)
            
            # Abilita high level commander
            self.scf.cf.param.set_value("commander.enHighLevel", "1")
            time.sleep(0.5)
            
            # Assegna il commander DOPO la connessione
            self.commander = self.scf.cf.high_level_commander
            
            # Reset e aspetta la stabilizzazione del Kalman
            self.reset_kalman_estimator()
            
                
            self.connected = True
            print("Connected to Crazyflie - Ready for flight")
            
            self._track_position()
            
        except Exception as e:
            print(f"Connection failed: {e}")
            if hasattr(self, 'scf'):
                self.scf.close_link()
            raise

    def takeoff(self, height=0.5, duration=3):
        if self.move_enable:
            self.commander.takeoff(height, duration)
            time.sleep(duration)
            self.is_flying = True

    def land(self, height=0.0, duration=4.5):
        if self.move_enable:
            self.commander.land(height, duration)
            time.sleep(duration)
            self.is_flying = False

    def go_to(self, x, y, z, yaw, duration, relative=True):
        self.commander.go_to(x, y, z, yaw, duration, relative)

    def start_stream(self):
        pass
        # self.stream_on = True
        # self.ai_deck_client.connect()

    def stop_stream(self):
        pass
        # self.stream_on = False
        # if self.ai_deck_client.socket:
        #     self.ai_deck_client.close()
        # if self.ai_deck_client.socket:
        #     self.ai_deck_client.close()

    def get_move_enable(self):
        return True
    
    def get_is_flying(self):
        return self.is_flying
    
    def set_is_flying(self, value):
        self.is_flying = value
    
    def create_new_trajectory(self, gesture, duration_s=15)-> Tuple[bool, bool]:
        general_utils.print_t(f"New trajectory creation ... associated with gesture {gesture}")
        if self.get_move_enable() and self.get_is_flying():
            self.land(height=0.0, duration=4.5)
            self.set_is_flying(False)
        time.sleep(6.0)
        self._record_trajectory(duration_s)
        general_utils.run_command('python3 controller/generate_trajectory.py my_timed_waypoints_yaw.csv traj.csv --pieces 5')
        print('New trajectory recorded')
        print('uploading trajectory...')
        header, dataTraj = general_utils.import_csv('traj.csv')
        self.gesture_trajectory_mapping.update({gesture : dataTraj})
        print(f'DICTIONARY: {self.gesture_trajectory_mapping}')
        self.sound_effect(7)
        return True, False

    def _record_trajectory(self, duration_s):
        from cflib.crazyflie.log import LogConfig

        lg_stab = LogConfig(name='Stabilizer', period_in_ms=500)
        for v in ['x', 'y', 'z', 'yaw']:
            lg_stab.add_variable(f'stateEstimate.{v}', 'float')

        timed_waypoints = {'t': [], 'x': [], 'y': [], 'z': [], 'yaw': []}

        self.sound_effect(3)
        for count in ['3', '2', '1']:
            print(count)
            time.sleep(1)
        print('Start recording...')
        self.sound_effect(6)

        with SyncLogger(self.scf, lg_stab) as logger:
            start_time = time.time()
            end_time = start_time + duration_s

            for log_entry in logger:
                ts, data, _ = log_entry
                t = time.time() - start_time
                for key in timed_waypoints:
                    if key != 't':
                        timed_waypoints[key].append(data[f'stateEstimate.{key}'])
                timed_waypoints['t'].append(t)
                if time.time() > end_time:
                    general_utils.save_to_csv(timed_waypoints, 'my_timed_waypoints_yaw.csv')
                    print("CSV saved.")
                    break

    def _upload_trajectory(self, trajectory_id, trajectory):
        
        trajectory_mem = self.cf.mem.get_mems(MemoryElement.TYPE_TRAJ)[0]
        trajectory_mem.trajectory = []
        total_duration = 0
        for row in trajectory:
            duration = float(row[0])
            x  = Poly4D.Poly(list(map(float, row[1:9])) )
            y  = Poly4D.Poly(list(map(float, row[9:17])))
            z  = Poly4D.Poly(list(map(float, row[17:25])))
            # yaw= Poly4D.Poly(list(map(float, row[25:33])))
            yaw_deg = list(map(float, row[25:33]))
            yaw_rad = [math.radians(c) for c in yaw_deg]
            yaw = Poly4D.Poly(yaw_rad)
            trajectory_mem.trajectory.append(Poly4D(duration, x, y, z, yaw))
            total_duration += duration
        if not trajectory_mem.write_data_sync():
            print('Upload failed.')
            sys.exit(1)
        self._define_trajectory(trajectory_id, trajectory_mem.trajectory)
        return total_duration

    def _define_trajectory(self, trajectory_id, traj_data):
        self.cf.high_level_commander.define_trajectory(trajectory_id, 0, len(traj_data))

    def _start_trajectory(self, trajectory_id: int, time_scale=1.0, relative=True):
        self.commander.start_trajectory(trajectory_id, time_scale, relative=relative)

    def _run_sequence(self, trajectory_id, duration):
        if not self.is_flying:
            general_utils.print_t("_run_sequence function before taking off")
            self.takeoff(height=0.4, duration=2.0)
            general_utils.print_t("_run_sequence function after taking off")
            time.sleep(2.0)
            self.is_flying = True
        self._start_trajectory(trajectory_id, time_scale=1.0, relative=True)
        # Wait for trajectory to complete
        time.sleep(duration + 1.0)  # Add buffer time

    def start_trajectory(self, gesture) -> Tuple[bool, bool]:
        print('[BASIC_COMMANDS] Start trajectory command received')
        traj = self.gesture_trajectory_mapping.get(gesture)
        print(traj)
        trajectory_id = 1 #for now we overwrite the trajectory every time, so there is no more than 1 trajectory on the crazyflie.
        duration = self._upload_trajectory(trajectory_id, traj) #duration=around 15s
        print('Done! duration: ', duration)
        self._run_sequence(trajectory_id=1, duration=duration)
        # time.sleep(duration)
        return True, False

    def reset_kalman_estimator(self):
        self.cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.5)  # Aumenta il tempo
        self.cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(1.0)  # Tempo aggiuntivo
        print('Waiting for estimator to find position...')

        log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
        log_config.add_variable('kalman.varPX', 'float')
        log_config.add_variable('kalman.varPY', 'float')
        log_config.add_variable('kalman.varPZ', 'float')

        var_y_history = [1000] * 10
        var_x_history = [1000] * 10
        var_z_history = [1000] * 10

        threshold = 0.001

        with SyncLogger(self.scf, log_config) as logger:
            for log_entry in logger:
                data = log_entry[1]

                var_x_history.append(data['kalman.varPX'])
                var_x_history.pop(0)
                var_y_history.append(data['kalman.varPY'])
                var_y_history.pop(0)
                var_z_history.append(data['kalman.varPZ'])
                var_z_history.pop(0)

                min_x = min(var_x_history)
                max_x = max(var_x_history)
                min_y = min(var_y_history)
                max_y = max(var_y_history)
                min_z = min(var_z_history)
                max_z = max(var_z_history)

                # print("{} {} {}".
                #       format(max_x - min_x, max_y - min_y, max_z - min_z))

                if (max_x - min_x) < threshold and (
                        max_y - min_y) < threshold and (
                        max_z - min_z) < threshold:
                    break

    def sound_effect(self, value):
        self.cf.param.set_value("sound.effect", value)

    def no_motion(self, duration=0.4):
        self.go_to(0, 0, 0, -math.pi / 5, duration)
        time.sleep(duration)
        self.go_to(0, 0, 0, 2 * math.pi / 5, duration * 2)
        time.sleep(duration * 2)
        self.go_to(0, 0, 0, -math.pi / 5, duration)
        time.sleep(duration)

    def rotate_360(self, duration=2.0):
        self.go_to(0, 0, 0, math.pi, duration / 2)
        time.sleep(duration)
        self.go_to(0, 0, 0, -math.pi, duration / 2)
        time.sleep(duration)

    def victory_motion(self, duration=0.5):
        for _ in range(2):
            self.go_to(0, 0, 0.2, 0, duration)
            time.sleep(duration)
            self.go_to(0, 0, -0.2, 0, duration)
            time.sleep(duration)


    def move_north(self, distance: int) -> bool:
        pass
    
    def move_south(self, distance: int) -> bool:
        pass
    
    def move_west(self, distance: int) -> bool:
        pass

    def move_east(self, distance: int) -> bool:
        pass
    
    def move_up(self, distance: int) -> bool:
        pass
    
    def move_down(self, distance: int) -> bool:
        pass

    def turn_ccw(self, degree: int) -> bool:
        pass

    def turn_cw(self, degree: int) -> bool:
        pass

    def keep_active(self):
        pass

    def get_frame_reader(self):
        pass

    def go_to_position(self, current_pos, target_pos, speed=50):
        pass

    def move_direction(self, direction, distance):
        pass
