import keyboard
import time
import cv2 as cv
from threading import Thread, Event
from djitellopy import tello


######################################################################
width = 640  # WIDTH OF THE IMAGE
height = 480  # HEIGHT OF THE IMAGE
deadZone =100
#####################################################################


class TelloController:
    def __init__(self):
        # Initialize drone and setup keyboard listener
        self.tello_drone = tello.Tello()
        self.tello_drone.connect()
        self.tello_drone.streamon()
        self.stop_controller = Event() # Event to signal thread termination
        self.start_keyboard_listener() # Setup keyboard listener thread
        self.data = pd.DataFrame(columns=['speed_X', 'speed_Y', 'speed_Z','accel_X', 'accel_Y', 'accel_Z', 'roll', 'pitch', 'yaw']) # Initialize DataFrame to store data
        self.start_data_collection()
        

    def start_keyboard_listener(self): # Start a thread for keyboard
        Thread(target=self.keyboard_listener).start()

    def keyboard_listener(self):

        while not self.stop_controller.is_set():
            key = keyboard.read_key()

            if key == 't':
                self.takeoff()
                
            elif key == 'q':
                self.land()
                self.stop_controller.set()
                print('Quitting, Please Wait...')
                print('Landing Drone...')


    def takeoff(self): 
        print('Drone taking off, Please Wait...')
        self.tello_drone.takeoff()
        print('Drone took off.')
        time.sleep(10)

    def land(self):
        print('Drone landing, Please Wait...')
        self.tello_drone.land()
        print('Drone landed')

    def start_data_collection(self): # Start a thread for data collection
        Thread(target=self.collect_data).start()

    def collect_data(self): # Append current row data to the DataFrame
        while not self.stop_controller.is_set(): # Only if drone is flying.
            # Capture frame from drone
            frame = self.tello_drone.get_frame_read().frame

            # Display frame on live stream
            cv.imshow("Drone Camera", frame)
            cv.waitkey(0) # Continuously update the stream

            new_row = {
                'speed_X': self.tello_drone.get_speed_x(),
                'speed_Y': self.tello_drone.get_speed_y(),
                'speed_Z': self.tello_drone.get_speed_z(),
                'accel_X': self.tello_drone.get_acceleration_x(),
                'accel_Y': self.tello_drone.get_acceleration_y(),
                'accel_Z': self.tello_drone.get_acceleration_z(),
                'roll': self.tello_drone.get_roll(),
                'pitch': self.tello_drone.get_pitch(),
                'yaw': self.tello_drone.get_yaw()
            }
            self.data = self.data.append(new_row, ignore_index=True) ## Index to infinity for now. maybe we only need Index[0]?
            time.sleep(1) ## To be adjusted..

if __name__ == '__main__':
    tc = TelloController()
