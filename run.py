import cv2
import json
import numpy as np
import os
import uuid
from ultralytics import YOLO
from plate_validator import PlateRecognitionValidator
from fast_plate_ocr import LicensePlateRecognizer
from fuzzywuzzy import fuzz
from datetime import datetime
import mysql.connector
import threading

# Veritabanı yapılandırması
DB_CONFIG = {
    'host': "localhost",
    'user': "root",
    'password': "",
    'database': "proje"
}

# Yardımcı fonksiyonlar
def get_db():
    try:
        mydb = mysql.connector.connect(**DB_CONFIG)
        cursor = mydb.cursor()
        return mydb, cursor
    except mysql.connector.Error as err:
        print(f"Veritabanı bağlantı hatası: {err}")
        return None, None

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(image, -1, kernel)
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def point_in_roi(point, roi_polygon_np):
    return cv2.pointPolygonTest(roi_polygon_np, point, False) >= 0

def box_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])  # Burası düzeltildi
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    return interArea / float(boxAArea)

def box_is_valid(x1, y1, x2, y2, min_area=800, max_area=35000, min_ratio=1.2, max_ratio=5.5):
    w, h = x2 - x1, y2 - y1
    area = w * h
    if area < min_area or area > max_area:
        return False
    ratio = w / h if h != 0 else 0
    return min_ratio <= ratio <= max_ratio

def process_video(video_path, video_id, output_dir):
    """
    Video işleme ve analizini gerçekleştirir.
    `web_app.py`'den `output_dir` parametresini alır.
    """
    try:
        mydb, cursor = get_db()
        if not mydb:
            print("Veritabanı bağlantısı kurulamadı.")
            return

        cursor.execute("SELECT id FROM videos WHERE id = %s", (video_id,))
        if not cursor.fetchone():
            print(f"Hata: video_id {video_id} 'videos' tablosunda bulunamadı. Analiz durduruldu.")
            cursor.close()
            mydb.close()
            return

        plate_model_path = "model_weights/best.pt"
        vehicle_model_path = "yolo11m.pt"
        plate_conf_threshold = 0.70
        vehicle_conf_threshold = 0.70
        plate_class_id = 0
        vehicle_class_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        fuzzy_match_threshold = 85
        cooldown_frames = 50

        os.makedirs(output_dir, exist_ok=True)
        saved_plates = {}
        roi_file = os.path.join("static", "uploads", f"roi_points_{video_id}.json")
        if not os.path.exists(roi_file):
            print(f"ROI dosyası bulunamadı: {roi_file}")
            cursor.close()
            mydb.close()
            return
            
        with open(roi_file, 'r') as f:
            roi_points_list = json.load(f)
        roi_polygon_np = np.array(roi_points_list, dtype=np.int32)

        plate_model = YOLO(plate_model_path)
        vehicle_model = YOLO(vehicle_model_path)
        plate_ocr_model = LicensePlateRecognizer('cct-s-v1-global-model')
        plate_validator = PlateRecognitionValidator(plate_ocr_model)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = sharpen_image(frame)
            plate_results = plate_model(frame, verbose=False)[0]
            vehicle_results = vehicle_model(frame, verbose=False)[0]

            plate_boxes = []
            for box in plate_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if cls == plate_class_id and conf >= plate_conf_threshold:
                    if box_is_valid(x1, y1, x2, y2) and point_in_roi(((x1 + x2) // 2, (y1 + y2) // 2), roi_polygon_np):
                        plate_boxes.append({'box': (x1, y1, x2, y2), 'conf': conf})

            vehicle_boxes = []
            for box in vehicle_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls in vehicle_class_ids and conf >= vehicle_conf_threshold:
                    if point_in_roi(((x1 + x2) // 2, (y1 + y2) // 2), roi_polygon_np):
                        vehicle_boxes.append({'box': (x1, y1, x2, y2), 'cls': cls, 'conf': conf})

            for plate_data in plate_boxes:
                plate_box = plate_data['box']
                px1, py1, px2, py2 = plate_box
                
                h, w, _ = frame.shape
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(w, px2), min(h, py2)

                plate_img = frame[py1:py2, px1:px2]
                
                temp_path = os.path.join(output_dir, f"temp_plate_{video_id}_{frame_count}.jpg")
                cv2.imwrite(temp_path, plate_img)
                ocr_result = plate_validator.run_detection(temp_path)
                os.remove(temp_path)

                if ocr_result["status"] != "success":
                    continue

                plate_text = ocr_result.get("plate", "")
                if not plate_text:
                    continue
                    
                confidence_score = ocr_result.get("confidence", 0.0)
                best_match_plate = None
                best_match_score = 0
                
                for saved_plate_text in saved_plates.keys():
                    similarity = fuzz.ratio(plate_text, saved_plate_text)
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_plate = saved_plate_text
                
                is_new_plate = True
                if best_match_plate and best_match_score >= fuzzy_match_threshold:
                    last_seen_frame = saved_plates[best_match_plate]
                    if (frame_count - last_seen_frame) < cooldown_frames:
                        print(f"⏭️ Zaten kaydedilmiş plaka veya benzeri: {plate_text} (orijinal: {best_match_plate})")
                        saved_plates[best_match_plate] = frame_count
                        is_new_plate = False
                
                if is_new_plate:
                    matched_vehicle_type = None
                    vehicle_box = None
                    for veh_data in vehicle_boxes:
                        veh_box = veh_data['box']
                        if box_overlap(plate_box, veh_box) > 0.2:
                            matched_vehicle_type = class_names.get(veh_data['cls'], 'Unknown')
                            vehicle_box = veh_box 
                            break
                    
                    if matched_vehicle_type:
                        timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
                        unique_id = uuid.uuid4()
                        
                        plate_filename = f"plaka_{plate_text}_{timestamp_str}_{unique_id}.jpg"
                        final_plate_image_path = os.path.join(output_dir, plate_filename)
                        cv2.imwrite(final_plate_image_path, plate_img)
                        
                        vehicle_filename = None
                        final_vehicle_image_path = None
                        if vehicle_box:
                            vx1, vy1, vx2, vy2 = vehicle_box
                            vehicle_img = frame[vy1:vy2, vx1:vx2]
                            vehicle_filename = f"arac_{plate_text}_{timestamp_str}_{unique_id}.jpg"
                            final_vehicle_image_path = os.path.join(output_dir, vehicle_filename)
                            cv2.imwrite(final_vehicle_image_path, vehicle_img)
                            print(f"✅ Araç görseli kaydedildi: {final_vehicle_image_path}")
                        
                        insert_query = "INSERT INTO plate_detections (video_id, plate_text, vehicle_type, confidence_score, detection_timestamp, plate_image_path, vehicle_image_path) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                        data = (video_id, plate_text, matched_vehicle_type, confidence_score, datetime.now(), final_plate_image_path, final_vehicle_image_path)
                        
                        try:
                            cursor.execute(insert_query, data)
                            mydb.commit()
                            print(f"✅ Yeni plaka veritabanına kaydedildi: {plate_text} (Araç Tipi: {matched_vehicle_type})")
                        except Exception as e:
                            print(f"Veritabanına kayıt hatası: {e}")
                        
                        saved_plates[plate_text] = frame_count

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        cursor.close()
        mydb.close()
        print(f"Video {video_id} analizi tamamlandı.")
        
    except Exception as e:
        print(f"process_video fonksiyonunda beklenmedik bir hata oluştu: {e}")


        #bu 