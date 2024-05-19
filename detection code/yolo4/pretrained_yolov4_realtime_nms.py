import numpy as np
import cv2
import threading

def detect_objects():
    global img_to_detect, stop_thread, detections
    while not stop_thread:
        if img_to_detect is not None:
            blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (320, 320), swapRB=True, crop=False)
            yolo_model.setInput(blob)
            detections = yolo_model.forward(output_layer_names)

def process_detections():
    global detections, img_height_orig, img_width_orig, img_height, img_width, boxes_list, confidences_list, class_ids_list

    class_ids_list.clear()
    boxes_list.clear()
    confidences_list.clear()

    if detections is not None:
        for detection in detections:
            for object_detection in detection:
                all_scores = object_detection[5:]
                predicted_class_id = np.argmax(all_scores)
                prediction_confidence = all_scores[predicted_class_id]

                if prediction_confidence > 0.5:
                    bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                    (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                    start_x_pt = int(box_center_x_pt - (box_width / 2))
                    start_y_pt = int(box_center_y_pt - (box_height / 2))

                    class_ids_list.append(predicted_class_id)
                    confidences_list.append(float(prediction_confidence))
                    boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])

        max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

        final_boxes_list = []
        final_class_ids_list = []
        final_confidences_list = []

        if len(max_value_ids) > 0:
            for i in max_value_ids.flatten():
                box = boxes_list[i]
                final_boxes_list.append(box)
                final_class_ids_list.append(class_ids_list[i])
                final_confidences_list.append(confidences_list[i])

        return final_boxes_list, final_class_ids_list, final_confidences_list
    return [], [], []

def draw_boxes(frame, boxes, class_ids, confidences):
    global class_colors, class_labels
    for i in range(len(boxes)):
        box = boxes[i]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]

        # Scale bounding box coordinates back to the size of the original frame
        scale_x = img_width_orig / img_width
        scale_y = img_height_orig / img_height
        start_x_pt = int(start_x_pt * scale_x)
        start_y_pt = int(start_y_pt * scale_y)
        box_width = int(box_width * scale_x)
        box_height = int(box_height * scale_y)

        predicted_class_id = class_ids[i]
        predicted_class_label = class_labels[predicted_class_id]
        prediction_confidence = confidences[i]

        end_x_pt = start_x_pt + box_width
        end_y_pt = start_y_pt + box_height

        box_color = class_colors[predicted_class_id]
        box_color = [int(c) for c in box_color]

        predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
        print("predicted object {}".format(predicted_class_label))

        cv2.rectangle(frame, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
        cv2.putText(frame, predicted_class_label, (start_x_pt, start_y_pt - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

webcam_video_stream = cv2.VideoCapture(0)

# Load YOLO model
yolo_model = cv2.dnn.readNetFromDarknet('D:\\python\\dataset\\yolov4.cfg', 'D:\\python\\dataset\\yolov4.weights')
output_layer_names = [yolo_model.getLayerNames()[i - 1] for i in yolo_model.getUnconnectedOutLayers()]

# If you have an NVIDIA GPU, use CUDA
yolo_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yolo_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

stop_thread = False
img_to_detect = None
current_frame = None
detections = None
boxes_list = []
confidences_list = []
class_ids_list = []
thread = threading.Thread(target=detect_objects)
thread.start()

frame_count = 0
frame_skip = 5  # Process every 5th frame

class_labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddy bear", "hair dryer", "toothbrush"]

class_colors = ["0,255,0", "0,0,255", "255,0,0", "255,255,0", "0,255,255"]
class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
class_colors = np.array(class_colors)
class_colors = np.tile(class_colors, (16, 1))

final_boxes_list = []
final_class_ids_list = []
final_confidences_list = []

while True:
    ret, current_frame = webcam_video_stream.read()
    frame_count += 1

    if frame_count % frame_skip == 0:
        img_height_orig, img_width_orig = current_frame.shape[:2]
        img_to_detect = cv2.resize(current_frame, (320, 320))
        img_height = img_to_detect.shape[0]
        img_width = img_to_detect.shape[1]
        final_boxes_list, final_class_ids_list, final_confidences_list = process_detections()

    draw_boxes(current_frame, final_boxes_list, final_class_ids_list, final_confidences_list)
    cv2.imshow("Detection Output", current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_thread = True
        break

webcam_video_stream.release()
cv2.destroyAllWindows()
thread.join()
