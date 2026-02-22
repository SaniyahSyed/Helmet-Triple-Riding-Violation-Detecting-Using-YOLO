import cv2
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1[:4]  
    x1_b, y1_b, x2_b, y2_b = box2[:4]  
    
    xi1 = max(x1, x1_b)
    yi1 = max(y1, y1_b)
    xi2 = min(x2, x2_b)
    yi2 = min(y2, y2_b)
    
    if xi2 < xi1 or yi2 < yi1:
        return 0.0

    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_b - x1_b) * (y2_b - y1_b)

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

helmet_model = YOLO('runs/detect/HelmetVioltionDetection-YOLOV8m/weights/best.pt')
triple_riding_model = YOLO('yolov8m.pt')

def helmet_and_triple_riding_detection_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return
    frame_counter = 0 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        frame_counter += 1
        if frame_counter % 6 != 0: 
            continue

        helmet_results = helmet_model.predict(frame, conf=0.6)

        annotated_image = helmet_results[0].plot()

        if helmet_results[0].boxes is not None:
            for box in helmet_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) 
                
                if int(box.cls[0]) == 1:
                    cv2.putText(annotated_image, "Helmet Detected", (x1-50, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if int(box.cls[0]) == 2:
                    cv2.putText(annotated_image, "No Helmet Detected", (x1-50, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        triple_riding_results = triple_riding_model.predict(frame, conf=0.5)
        bikes = []
        people = []
        triple_riding_violation_count = 0

        for box in triple_riding_results[0].boxes:
            class_id = int(box.cls) 
            if class_id == 0: 
                people.append(box.xyxy[0].cpu().numpy())  
            elif class_id == 3:  
                bikes.append(box.xyxy[0].cpu().numpy())  

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf.item()
            label = f"{triple_riding_model.names[class_id]}: {confidence:.2f}"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
        violation_detected = False
        for bike in bikes:
            bike_box = bike.astype(int)
            people_count = 0
            for person in people:
                person_box = person.astype(int)
                iou = calculate_iou(bike_box, person_box)
                if iou > 0.1: 
                    people_count += 1

            if people_count > 2: 
                violation_detected = True
                triple_riding_violation_count += 1
                cv2.rectangle(annotated_image, (bike_box[0], bike_box[1]), (bike_box[2], bike_box[3]), (0, 0, 255), 2)
                violation_text = f"Violation: {people_count} riders"
                cv2.putText(annotated_image, violation_text, (bike_box[0], bike_box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if violation_detected:
            cv2.putText(annotated_image, "Triple Riding Violation Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1)
        else:
            cv2.putText(annotated_image, "No Triple Riding Violation", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 1)

        cv2.imshow("Helmet and Triple Riding Violation Detection", annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


video_path = input("Enter video path: ")
helmet_and_triple_riding_detection_video(video_path)