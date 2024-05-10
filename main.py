import keyboard
import time
from threading import Thread, Event
from djitellopy import tello

class TelloController:
    def __init__(self):
        # Initialize drone and setup keyboard listener
        self.tello_drone = tello.Tello()
        self.tello_drone.connect()
        self.tello_drone.streamon()
        self.stop_controller = Event() # Event to signal thread termination
        self.thread_keyboard_listener() # Setup keyboard listener thread

    def thread_keyboard_listener(self): # Start a thread for keyboard
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

if __name__ == '__main__':
    tc = TelloController()
