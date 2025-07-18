
import math
import sys
import time, cv2
import logging
import numpy as np
from typing import Tuple

import controller.utils as utils
from .abs.robot_wrapper import RobotWrapper



import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.commander import Commander

from cflib.utils import uri_helper

class CrazyflieWrapper(RobotWrapper):
    def __init__(self, move_enable: bool = False):
        super().__init__(move_enable=move_enable)
        self.cf=Crazyflie(rw_cache='./cache')
        self.stream_on = False
        self.connected = False
        self.link_uri = 'radio://0/80/2M/E7E7E7E7E7'
        self.z_target = 0.5  # Default target height in meters

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
            
            # Verifica batteria
            if not self.check_battery():
                raise Exception("Battery level too low")
                
            self.connected = True
            print("Connected to Crazyflie - Ready for flight")
            
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
    
    def create_new_trajectory(self, gesture, duration_s=15)-> Tuple[bool, bool]:
        utils.print_t(f"New trajectory creation ... associated with gesture {gesture}")
        if self.get_move_enable() and self.get_is_flying():
            self.land(height=0.0, duration=4.5)
            self.set_is_flying(False)
        time.sleep(6.0)
        self._record_trajectory(duration_s)
        utils.run_command('python3 controller/generate_trajectory.py my_timed_waypoints_yaw.csv traj.csv --pieces 5')
        print('New trajectory recorded')
        print('uploading trajectory...')
        header, dataTraj = utils.import_csv('traj.csv')
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
                    utils.save_to_csv(timed_waypoints, 'my_timed_waypoints_yaw.csv')
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
            utils.print_t("_run_sequence function before taking off")
            self.takeoff(height=0.4, duration=2.0)
            utils.print_t("_run_sequence function after taking off")
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