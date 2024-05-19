import cv2
import time
import pandas as pd
from threading import Thread, Event
from djitellopy import Tello
from pretrained_yolov4_realtime_nms import ObjectDetector

class TelloController:
    def __init__(self):
        self.tello_drone = Tello()
        self.tello_drone.connect()
        self.tello_drone.streamon()
        self.stop_controller = Event()
        self.data = pd.DataFrame(columns=['Speed_X', 'Speed_Y', 'Speed_Z',
                                          'accel_X', 'accel_Y', 'accel_Z',
                                          'roll', 'pitch', 'yaw'])
        self.object_detector = ObjectDetector('yolov4-tiny.cfg', 'yolov4-tiny.weights', 'coco.names')
        self.setup_keyboard_listener()
        self.start_data_collection()

    def setup_keyboard_listener(self):
        Thread(target=self.keyboard_listener).start()

    def keyboard_listener(self):
        while not self.stop_controller.is_set():
            key = input()  # Change to input() for continuous input from console
            if key == 't':
                self.takeoff()
            elif key == 'q':
                self.land()
                self.stop_controller.set()
                print('Quitting... Please Wait')
                print('Landing Drone...')

    def takeoff(self):
        self.tello_drone.takeoff()
        print('Drone took off')
        time.sleep(10)  # Wait for 10 seconds

    def land(self):
        self.tello_drone.land()
        print('Drone landed')
        # Save data to CSV when landing
        self.save_data_to_csv()

    def save_data_to_csv(self):
        # Save DataFrame to CSV
        self.data.to_csv('drone_data.csv', index=False)
        print('Data saved to CSV: drone_data.csv')

    def start_data_collection(self):
        Thread(target=self.collect_data).start()

    def collect_data(self):
        while not self.stop_controller.is_set():
            # Read frame from drone's camera
            frame = self.tello_drone.get_frame_read().frame

            # Perform object detection
            self.object_detector.img_to_detect = cv2.resize(frame, (416, 416))
            detections = self.object_detector.process_detections(frame)

            # Process detections to move the drone
            self.process_detections(detections, frame)

            # Display frame on live stream
            self.object_detector.draw_boxes(frame, *detections)
            cv2.imshow("Drone Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Append current telemetry data to the DataFrame
            new_row = {
                'Speed_X': self.tello_drone.get_speed_x(),
                'Speed_Y': self.tello_drone.get_speed_y(),
                'Speed_Z': self.tello_drone.get_speed_z(),
                'accel_X': self.tello_drone.get_acceleration_x(),
                'accel_Y': self.tello_drone.get_acceleration_y(),
                'accel_Z': self.tello_drone.get_acceleration_z(),
                'roll': self.tello_drone.get_roll(),
                'pitch': self.tello_drone.get_pitch(),
                'yaw': self.tello_drone.get_yaw()
            }
            self.data = self.data.append(new_row, ignore_index=True)

    def process_detections(self, detections, frame):
        img_height, img_width = frame.shape[:2]

        for (class_id, confidence, box) in detections:
            x, y, w, h = box
            label = str(self.object_detector.classes[class_id])

            if label == "cell phone":
                frame_center_x = img_width / 2
                object_center_x = x + (w / 2)

                # Calculate error
                error_x = object_center_x - frame_center_x

                # Move drone left or right
                if abs(error_x) > 20:  # Threshold to prevent jitter
                    if error_x > 0:
                        self.tello_drone.send_rc_control(20, 0, 0, 0)  # Move right
                    else:
                        self.tello_drone.send_rc_control(-20, 0, 0, 0)  # Move left
                else:
                    self.tello_drone.send_rc_control(0, 0, 0, 0)  # Stay still

if __name__ == '__main__':
    tc = TelloController()
