from math import floor
import cv2
import json
from datetime import datetime
import face_recognition
import numpy as np
from ultralytics import YOLO
import os
import time

employee = "visitors/employee"
unknown_folder = "visitors/unregistered"
report = "report/report.json"
yolo = YOLO("yolov8l.pt") 
known_face_encodings = []  
known_face_names = []  
were_tracked = {}
tracked_objects = {}

# Load known faces
def load_known_faces(face_folder):
    for person_folder in os.listdir(face_folder):
        for filename in os.listdir(face_folder + "/" + person_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = f"./{face_folder}/{person_folder}/{filename}"
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    known_face_encodings.append(encoding[0])
            known_face_names.append(person_folder)

# Update and clear JSON log file
def clear_json_file(file_path):
    with open(file_path, 'w') as f:
        json.dump([], f) 

def clear_unknown(obj_id, video_name):
    file_name = f"{obj_id}__{video_name}.jpg"
    folder_with_obj_id = os.path.join(unknown_folder, str(obj_id))
    file_path = os.path.join(folder_with_obj_id, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)


def clean_log(video_name):
    rec_ids = []
    filtered_events = []

    with open(report, "r") as f:
        events = json.load(f)

        for event in events:
            if event['event_type'] == 'recognized':
                rec_ids.append(event['object_id'])
                filtered_events.append(event)

        for id in rec_ids:
            ent = None
            diss = None
            for event in events:
                if event['object_id'] == id:
                    if event['event_type'] == 'entry':
                        ent = event
                    elif event['event_type'] == 'disappear':
                        diss = event
                        break  

            if ent is not None:
                filtered_events.append(ent)
            if diss is not None:
                filtered_events.append(diss)

            clear_unknown(id, video_name)

        id_list = []
        for folder_name in os.listdir(unknown_folder):
            full_path = os.path.join(unknown_folder, folder_name)
            if os.path.isdir(full_path):
                for file_name in os.listdir(full_path):
                    if file_name.endswith('.jpg'):
                        obj_id = file_name.split('__')[0] 
                        id_list.append(obj_id)

        for event in events:
            if str(event['object_id']) in id_list:
                filtered_events.append(event)
        
    filtered_events = sorted(filtered_events, key=lambda x: x['timestamp'])

    clear_json_file(report)
    with open(report, "w") as f:
        json.dump(filtered_events, f, indent=4)

def write_log(log_entry):
    with open(report, "r") as f:
        data = json.load(f)
        data.append(log_entry)
    with open(report, "w") as f:
        json.dump(data, f, indent=4)

def log_event(timestamp, event_type, obj_id, name=None):
    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "object_id": obj_id,
        "name": name if name else "Unknown"
    }
    write_log(log_entry)

# Improved tracking logic
def match_face_to_known(roi, obj_id):
    face_locations = face_recognition.face_locations(roi) 
    if face_locations:
        face_encodings = face_recognition.face_encodings(roi, face_locations)
        name = ""
        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(distances)
            if distances[best_match_index] < 0.48:
                name = known_face_names[best_match_index]
                tracked_objects[obj_id]["name"] = name
                tracked_objects[obj_id]["recognized"] = True
                return [name, True]
        return[None, True]
    return [None, False]

# Object tracking with consistency check
def run_recognition(video_name):
    count = 0
    frame_skip = 3
    curr_frame_time = 0 
    video_capture = cv2.VideoCapture(f"video/{video_name}.mp4")
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_name}.mp4")
        return
    
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    load_known_faces(employee)
    clear_json_file(report)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        count += 1

        if count % frame_skip != 0:
            continue
        
        curr_frame_time = round(count / fps, 2)

        detected_id = set()
        results = yolo.track(source=frame, persist=True, conf=0.5, iou=0.25)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                if box.id is None:
                    continue
                obj_id = int(box.id.item()) 

                detected_id.add(obj_id)
                cls = int(box.cls)  
                conf = float(box.conf)

                if cls == 0:  # Person class
                    if obj_id not in tracked_objects:
                        if obj_id not in were_tracked:
                            tracked_objects[obj_id] = {"name": "Unknown", "recognized": False}
                            log_event(curr_frame_time ,"entry", obj_id, tracked_objects[obj_id]["name"])

                    if not tracked_objects[obj_id]["recognized"]: 
                        roi = frame[y1:y2, x1:x2].copy()
                        name, seen_face = match_face_to_known(roi, obj_id)
                        if name:
                            log_event(curr_frame_time, "recognized", obj_id, name)
                        else:
                            if seen_face:
                                folder_with_obj_id = unknown_folder + "/" + str(obj_id)
                                unknown_image_path = folder_with_obj_id + "/" + f"{obj_id}__{video_name}.jpg"
                                os.makedirs(folder_with_obj_id, exist_ok=True)
                                cv2.imwrite(unknown_image_path, frame[y1:y2, x1:x2])
                    label = f'ID: {tracked_objects[obj_id]["name"]}, Conf: {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Check for missing tracked objects
        for obj_id in list(tracked_objects.keys()):
            if obj_id not in detected_id:
                log_event(curr_frame_time, "disappear", obj_id, tracked_objects[obj_id]["name"])
                del tracked_objects[obj_id]

        cv2.imshow("YOLOv8 Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    clean_log(video_name)
    video_capture.release()
    cv2.destroyAllWindows()

run_recognition("video_2024-12-17_12-39-06")
