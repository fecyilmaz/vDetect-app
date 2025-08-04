import cv2
import json
import numpy as np
from ultralytics import YOLO

# --- 1. Yardımcı Fonksiyon: Noktanın ROI İçinde Olup Olmadığını Kontrol Etme ---
def point_in_roi(point, roi_polygon_np):
    """
    Belirtilen noktanın verilen ROI (Region of Interest) çokgeninin içinde olup olmadığını kontrol eder.
    Args:
        point (tuple): (x, y) formatında kontrol edilecek nokta.
        roi_polygon_np (numpy.ndarray): ROI'yi tanımlayan çokgenin köşe noktaları.
    Returns:
        bool: Nokta ROI içindeyse True, değilse False.
    """
    # cv2.pointPolygonTest, noktanın çokgen içinde olup olmadığını kontrol eder.
    # Negatif: Dışında, Sıfır: Kenarında, Pozitif: İçinde
    return cv2.pointPolygonTest(roi_polygon_np, point, False) >= 0

def main():
    """
    Belirli bir video akışında tanımlı bir ROI içinde araçları algılar ve görselleştirir.
    Parametreler bir CONFIG sözlüğünden yüklenir.
    """
    # --- 2. Yapılandırma ve Parametreler ---
    # Bu parametreleri harici bir config dosyasından (örn. config.json) okumak daha iyi bir pratiktir.
    # Şimdilik, kod içinde tanımlıyoruz.
    CONFIG = {
        "video_path": 'YolKamerası.mp4',
        "model_name": 'yolo11m.pt', # 'yolov8n.pt' (nano) hızlıdır, daha doğru için 'yolov8s.pt' deneyin.
                                    # -> 'yolov8m.pt' olarak değiştirildi
        "confidence_threshold": 0.7, # Güven skoru eşiği (0.0 ile 1.0 arasında, genellikle 0.5 veya üzeri kullanılır)
        "vehicle_class_ids": [ # Sadece araçları filtrelemek için COCO veri seti sınıf ID'leri
            2,  # car
            3,  # motorcycle
            5,  # bus
            7,  # truck
        ],
        "roi_file": "roi_points.json"
    }

    # --- 3. ROI Noktalarını JSON Dosyasından Yükle ---
    try:
        with open(CONFIG["roi_file"], 'r') as f:
            roi_points_list = json.load(f)
        
        # roi_points_list'i bir kez numpy array'e dönüştür. Bu, her çağrıda tekrar dönüşümü engeller.
        roi_polygon_np = np.array(roi_points_list, dtype=np.int32)
        
        if len(roi_points_list) < 3:
            print(f"Hata: '{CONFIG['roi_file']}' dosyasında bir ROI tanımlamak için en az 3 nokta bulunmalıdır. Lütfen '{CONFIG['roi_file']}' dosyasını kontrol edin.")
            return

    except FileNotFoundError:
        print(f"Hata: '{CONFIG['roi_file']}' dosyası bulunamadı. Lütfen önce ROI belirleme aracını çalıştırın ve bir '{CONFIG['roi_file']}' dosyası oluşturun.")
        return
    except json.JSONDecodeError:
        print(f"Hata: '{CONFIG['roi_file']}' dosyası geçerli bir JSON formatında değil. Lütfen dosyanın içeriğini kontrol edin.")
        return
    except Exception as e:
        print(f"ROI noktaları okunurken beklenmeyen bir hata oluştu: {e}")
        return

    # --- 4. YOLO Modelini Yükle ---
    try:
        model = YOLO(CONFIG["model_name"])
    except FileNotFoundError:
        print(f"Hata: YOLO modeli '{CONFIG['model_name']}' bulunamadı. Model dosyasının doğru yolda olduğundan ve indirilmiş olduğundan emin olun.")
        return
    except Exception as e:
        print(f"YOLO modeli yüklenirken bir hata oluştu: {e}")
        return

    # --- 5. Video İşleme ---
    cap = cv2.VideoCapture(CONFIG["video_path"])

    if not cap.isOpened():
        print(f"Hata: Video açılamadı veya bulunamadı: {CONFIG['video_path']}. Lütfen video dosyasının yolunu ve erişilebilirliğini kontrol edin.")
        return

    print(f"Video işleniyor: {CONFIG['video_path']}...")
    print(f"Güven eşiği: {CONFIG['confidence_threshold']}.")
    print(f"Tespit edilecek araç sınıfları: {[model.names[i] for i in CONFIG['vehicle_class_ids']]}.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video sona erdi veya okunamadı. Program sonlandırılıyor.")
            break

        # YOLO ile nesne tespiti yap (verbose=False ile konsol çıktısını kapat)
        results = model(frame, verbose=False)[0]
        frame_display = frame.copy() # Çizimleri yapmak için karenin bir kopyasını al

        # --- 6. ROI Çizimi ---
        # roi_polygon_np zaten numpy array olduğu için doğrudan kullanılabilir.
        cv2.polylines(frame_display, [roi_polygon_np], isClosed=True, color=(0, 255, 255), thickness=3) # Sarı ROI

        # Tespit edilen her bir nesne kutusu için döngü
        for box in results.boxes:
            # Bounding box koordinatlarını al
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Bounding box'ın merkezi noktasını hesapla
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            # Sınıf ID'si ve güven skoru
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # --- 7. Araç Filtreleme ve ROI Kontrolü ---
            # Sadece araç sınıflarını ve belirli bir güven skorunun üzerindekileri dikkate al
            if cls in CONFIG["vehicle_class_ids"] and conf >= CONFIG["confidence_threshold"]:
                # Aracın orta noktasının ROI içinde olup olmadığını kontrol et
                if point_in_roi((x_center, y_center), roi_polygon_np):
                    # ROI içinde olan aracı yeşil renkte kutu ile çiz
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Aracın merkezini kırmızı nokta ile işaretle
                    cv2.circle(frame_display, (x_center, y_center), 5, (0, 0, 255), -1)
                    
                    # Sınıf adını ve güven skorunu etiket olarak ekle
                    label = model.names[cls]
                    display_text = f"{label} {conf:.2f}"
                    cv2.putText(frame_display, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    print(f"ROI içinde tespit edildi: {label} (Güven: {conf:.2f}) at ({x_center}, {y_center})")

        # Sonuçları ekranda göster
        cv2.imshow('ROI Araç Tespiti', frame_display)

        # 'q' tuşuna basıldığında döngüden çık
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("'q' tuşuna basılarak çıkış yapıldı.")
            break

    # --- 8. Kaynakları Serbest Bırak ---
    cap.release()
    cv2.destroyAllWindows()
    print("Program başarıyla sonlandırıldı.")

if __name__ == "__main__":
    main()