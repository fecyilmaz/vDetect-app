import cv2
import json
import numpy as np
import os
from ultralytics import YOLO

def sharpen_image(image):
    """Görüntüyü keskinleştirir ve kontrast artırır."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(image, -1, kernel)
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def point_distance_to_roi_edge(point, roi_polygon_np):
    return cv2.pointPolygonTest(roi_polygon_np, point, True)

def box_is_valid(x1, y1, x2, y2, min_area=800, max_area=35000, min_ratio=1.2, max_ratio=5.5):
    """Kutu oranı ve alanına göre saçma kutuları filtrele"""
    w, h = x2 - x1, y2 - y1
    area = w * h
    if area < min_area or area > max_area:
        return False
    ratio = w / h if h != 0 else 0
    return min_ratio <= ratio <= max_ratio

def main():
    CONFIG = {
        "video_path": "YolKamerası.mp4",
        "model_path": "model_weights/best.pt",
        "confidence_threshold": 0.70,
        "roi_file": "roi_points.json",
        "plate_class_id": 0,
        "roi_edge_tolerance": 25,
        "output_dir": "output_plates",
        "camera_text_filter": {
            "x_start_from_right": 400,
            "y_start_from_bottom": 150
        }
    }

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    saved_plate_ids = set()

    try:
        with open(CONFIG["roi_file"], 'r') as f:
            roi_points_list = json.load(f)
        roi_polygon_np = np.array(roi_points_list, dtype=np.int32)
        if len(roi_points_list) < 3:
            print("Hata: ROI dosyasında en az 3 nokta olmalı.")
            return
    except Exception as e:
        print(f"ROI dosyası okunamadı: {e}")
        return

    try:
        model = YOLO(CONFIG["model_path"])
    except Exception as e:
        print(f"Model yüklenemedi: {e}")
        return

    cap = cv2.VideoCapture(CONFIG["video_path"])
    if not cap.isOpened():
        print(f"Video açılamadı: {CONFIG['video_path']}")
        return

    print("Video işleniyor... 'q' ile çıkabilirsiniz.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video bitti veya okunamadı.")
            break

        frame_count += 1
        frame = sharpen_image(frame)
        results = model(frame, verbose=False)[0]
        frame_display = frame.copy()
        cv2.polylines(frame_display, [roi_polygon_np], isClosed=True, color=(0, 255, 255), thickness=2)

        frame_height, frame_width = frame.shape[:2]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == CONFIG["plate_class_id"] and conf >= CONFIG["confidence_threshold"]:

                # Saçma (çok küçük, çok büyük veya orantısız) kutuları atla
                if not box_is_valid(x1, y1, x2, y2):
                    continue

                dist_to_edge = point_distance_to_roi_edge((x_center, y_center), roi_polygon_np)

                if dist_to_edge >= 0:
                    box_color = (0, 255, 0)
                    show_box = True
                elif dist_to_edge >= -CONFIG["roi_edge_tolerance"]:
                    box_color = (0, 255, 255)
                    show_box = True
                else:
                    show_box = False

                # Kamera yazısı olan bölgeden gelen kutuları atla
                if show_box:
                    if (x1 > frame_width - CONFIG["camera_text_filter"]["x_start_from_right"] and
                        y1 > frame_height - CONFIG["camera_text_filter"]["y_start_from_bottom"]):
                        continue

                    # Kutuyu çiz
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), box_color, 2)
                    cv2.circle(frame_display, (x_center, y_center), 5, (0, 0, 255), -1)
                    label = model.names[cls] if hasattr(model, "names") else "Plate"
                    text = f"{label} {conf:.2f}"
                    cv2.putText(frame_display, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                    print(f"Plaka tespit: {text} (Kenara mesafe: {dist_to_edge:.1f})")

                    # Tekrar kaydetme kontrolü
                    plate_id = f"{x1}_{y1}_{x2}_{y2}"
                    if plate_id not in saved_plate_ids:
                        saved_plate_ids.add(plate_id)
                        plate_img = frame[y1:y2, x1:x2]
                        save_path = os.path.join(CONFIG["output_dir"], f"plate_{frame_count}_{x1}_{y1}.jpg")
                        cv2.imwrite(save_path, plate_img)
                        print(f"[✓] Plaka resmi kaydedildi: {save_path}")

        cv2.imshow("Plaka ROI Tespiti", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Çıkış yapıldı.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()