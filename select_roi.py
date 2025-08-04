import cv2
import json
import numpy as np

# Global değişken olarak ROI noktalarını saklamak için liste
roi_points = []

# Fare olaylarını işleyen geri çağırım fonksiyonu
def mouse_handler(event, x, y, flags, param):
    """
    Fare olaylarını işler ve ROI noktalarını global listeye ekler veya siler.
    """
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        print(f"Nokta seçildi: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN: # Sağ tıklama ile son noktayı silme
        if roi_points:
            removed_point = roi_points.pop()
            print(f"Son nokta silindi: {removed_point}")
        else:
            print("Silinecek nokta yok.")

def main():
    """
    Kullanıcının bir video karesi üzerinden ROI noktalarını seçmesini sağlayan aracı çalıştırır.
    Seçilen noktaları 'roi_points.json' dosyasına kaydeder.
    """
    video_path = 'YolKamerası.mp4'
    cap = cv2.VideoCapture(video_path)

    # Videodan sadece ilk kareyi alıyoruz. ROI seçimi için yeterlidir.
    ret, frame = cap.read()
    cap.release() # İlk kareyi aldıktan sonra videoyu kapatabiliriz, bellekte yer kaplamaz

    if not ret:
        print(f"Hata: Video açılamadı veya ilk kare alınamadı: {video_path}. Lütfen video dosyasının yolunu ve erişilebilirliğini kontrol edin.")
        return

    # ROI seçim penceresini oluştur ve fare geri çağırımını ayarla
    window_name = "ROI Belirleme Aracı - Sol Tıkla Nokta Ekle, Sağ Tıkla Nokta Sil, 'q' ile Bitir"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_handler)

    print("\nYolun/bölgenin köşelerini fare ile sol tıklayarak belirleyin.")
    print("Yanlış nokta seçerseniz, fare ile sağ tıklayarak son noktayı silebilirsiniz.")
    print("İşiniz bittiğinde pencereye odaklanıp 'q' tuşuna basın.")
    print("Her tıklamada yeni bir köşe noktası eklenecektir.")

    while True:
        # Görüntüye noktaları ve çizgileri çizmek için geçici bir kopya oluştur
        temp_frame = frame.copy()

        if len(roi_points) > 0:
            # Belirlenen noktaları küçük daireler halinde çiz
            for point in roi_points:
                cv2.circle(temp_frame, point, 5, (0, 255, 0), -1) # Yeşil daireler

            # Noktalar arasında çizgiler çiz
            if len(roi_points) > 1:
                # isClosed=True: İlk ve son noktayı otomatik olarak birleştirerek kapalı bir çokgen çizer
                cv2.polylines(temp_frame, [np.array(roi_points)], isClosed=True, color=(0,255,0), thickness=2) # Yeşil çizgiler

        cv2.imshow(window_name, temp_frame)

        # 'q' tuşuna basılana kadar bekle
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows() # Tüm OpenCV pencerelerini kapat

    # --- Noktaları Kaydetme ---
    # Eğer 2'den az nokta varsa (yani bir çizgi veya tek nokta), bir çokgen oluşturamaz
    if len(roi_points) < 3:
        print("\nUyarı: ROI için en az 3 nokta gereklidir. Noktalar kaydedilmedi.")
        return

    try:
        # Hata düzeltmesi: Dosyayı yazma modunda ('w') aç
        with open('roi_points.json', 'w') as f:
            json.dump(roi_points, f, indent=4) # indent=4 ile daha okunaklı JSON çıktısı
        print(f"\nROI noktaları 'roi_points.json' dosyasına başarıyla kaydedildi: {roi_points}")
    except IOError as e:
        print(f"\nHata: 'roi_points.json' dosyasına yazılırken bir sorun oluştu: {e}")
    except Exception as e:
        print(f"Noktalar kaydedilirken beklenmeyen bir hata oluştu: {e}")

if __name__ == "__main__":
    main()