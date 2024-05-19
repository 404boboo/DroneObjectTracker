import numpy as np
import cv2
import threading
import time

class ObjectDetector:
    def __init__(self, config_path, weights_path, labels_path, confidence_threshold=0.5, nms_threshold=0.4):
        self.yolo_model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.output_layer_names = [self.yolo_model.getLayerNames()[i - 1] for i in self.yolo_model.getUnconnectedOutLayers()]

        # If you have an NVIDIA GPU, use CUDA
        self.yolo_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.yolo_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        with open(labels_path, 'r') as f:
            self.classes = f.read().strip().split("\n")

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.stop_thread = False
        self.img_to_detect = None
        self.detections = None
        self.boxes_list = []
        self.confidences_list = []
        self.class_ids_list = []
        self.thread = threading.Thread(target=self.detect_objects)
        self.thread.start()

    def detect_objects(self):
        while not self.stop_thread:
            if self.img_to_detect is not None:
                blob = cv2.dnn.blobFromImage(self.img_to_detect, 0.003922, (416, 416), swapRB=True, crop=False)
                self.yolo_model.setInput(blob)
                self.detections = self.yolo_model.forward(self.output_layer_names)

    def process_detections(self, frame):
        img_height, img_width = frame.shape[:2]
        self.boxes_list.clear()
        self.confidences_list.clear()
        self.class_ids_list.clear()

        if self.detections is not None:
            for detection in self.detections:
                for object_detection in detection:
                    all_scores = object_detection[5:]
                    predicted_class_id = np.argmax(all_scores)
                    prediction_confidence = all_scores[predicted_class_id]

                    if prediction_confidence > self.confidence_threshold and self.classes[predicted_class_id] == "cell phone":
                        bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                        (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                        start_x_pt = int(box_center_x_pt - (box_width / 2))
                        start_y_pt = int(box_center_y_pt - (box_height / 2))

                        self.class_ids_list.append(predicted_class_id)
                        self.confidences_list.append(float(prediction_confidence))
                        self.boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])

            max_value_ids = cv2.dnn.NMSBoxes(self.boxes_list, self.confidences_list, self.confidence_threshold, self.nms_threshold)

            final_boxes_list = []
            final_class_ids_list = []
            final_confidences_list = []

            if len(max_value_ids) > 0:
                for i in max_value_ids.flatten():
                    box = self.boxes_list[i]
                    final_boxes_list.append(box)
                    final_class_ids_list.append(self.class_ids_list[i])
                    final_confidences_list.append(self.confidences_list[i])

            return final_boxes_list, final_class_ids_list, final_confidences_list
        return [], [], []

    def draw_boxes(self, frame, boxes, class_ids, confidences):
        class_colors = ["0,255,0", "0,0,255", "255,0,0", "255,255,0", "0,255,255"]
        class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
        class_colors = np.array(class_colors)
        class_colors = np.tile(class_colors, (16, 1))

        for i in range(len(boxes)):
            box = boxes[i]
            start_x_pt = box[0]
            start_y_pt = box[1]
            box_width = box[2]
            box_height = box[3]

            predicted_class_id = class_ids[i]
            prediction_confidence = confidences[i]

            end_x_pt = start_x_pt + box_width
            end_y_pt = start_y_pt + box_height

            box_color = class_colors[predicted_class_id]
            box_color = [int(c) for c in box_color]

            predicted_class_label = "{}: {:.2f}%".format(self.classes[predicted_class_id], prediction_confidence * 100)
            print("predicted object {}".format(predicted_class_label))

            cv2.rectangle(frame, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
            cv2.putText(frame, predicted_class_label, (start_x_pt, start_y_pt - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

# Main loop
webcam_video_stream = cv2.VideoCapture(0)
object_detector = ObjectDetector(
    'C:\\Users\\ahmed\\OneDrive\\Desktop\\Projects\\DroneObjectTracker\\tello-ai\\dataset\\yolov4-tiny.cfg', 
    'C:\\Users\\ahmed\\OneDrive\\Desktop\\Projects\\DroneObjectTracker\\tello-ai\\dataset\\yolov4-tiny.weights', 
    'C:\\Users\\ahmed\\OneDrive\\Desktop\\Projects\\DroneObjectTracker\\tello-ai\\dataset\\coco.names',
    confidence_threshold=0.5,  # Higher confidence threshold
    nms_threshold=0.4          # NMS threshold
)

frame_count = 0
frame_skip = 1  # Process every frame

final_boxes_list = []
final_class_ids_list = []
final_confidences_list = []

while True:
    ret, current_frame = webcam_video_stream.read()
    frame_count += 1

    if frame_count % frame_skip == 0:
        object_detector.img_to_detect = cv2.resize(current_frame, (416, 416))
        final_boxes_list, final_class_ids_list, final_confidences_list = object_detector.process_detections(current_frame)

    object_detector.draw_boxes(current_frame, final_boxes_list, final_class_ids_list, final_confidences_list)
    cv2.imshow("Detection Output", current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        object_detector.stop_thread = True
        break

webcam_video_stream.release()
cv2.destroyAllWindows()
object_detector.thread.join()
