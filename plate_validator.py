import cv2
import numpy as np
import re
from typing import List, Tuple, Optional, Dict
import os
import csv
from fast_plate_ocr import LicensePlateRecognizer




class PlateRecognitionValidator:
    def __init__(self, model):
        self.model = model
        self.min_confidence_threshold = 0.7  # Minimum güven eşiği
        self.max_plates_allowed = 1  # Tek plaka bekleniyor
        
        # Türkiye plaka formatları (örnek olarak, ihtiyacınıza göre genişletebilirsiniz)
        self.valid_patterns = [
            r'^[0-9]{2}[A-Z]{1,3}[0-9]{2,4}$',  # 34ABC123 formatı
            r'^[0-9]{2}[A-Z]{2}[0-9]{3,4}$',    # 34AB123 formatı
            r'^[0-9]{2}[A-Z]{3}[0-9]{2}$',      # 34ABC12 formatı
        ]
        
        # Global plaka formatları (diğer ülkeler için)
        self.global_patterns = [
            r'^[A-Z0-9]{5,8}$',  # Genel format
            r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$',  # Karma format
        ]
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Görüntüyü ön işleme tabi tutar"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Görüntü yüklenemedi: {image_path}")
        
        # Görüntü kalitesini kontrol et
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return image, gray
    
    def detect_plate_regions(self, gray_image: np.ndarray) -> List[Tuple]:
        """Potansiyel plaka bölgelerini tespit eder"""
        # Kenar tespiti
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # Konturları bul
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plate_regions = []
        
        for contour in contours:
            # Dikdörtgen yaklaşımı
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            
            if len(approx) == 4:  # Dikdörtgen
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                area = cv2.contourArea(contour)
                
                # Plaka benzeri özellikler
                if 2.0 <= aspect_ratio <= 5.0 and area > 1000:
                    plate_regions.append((x, y, w, h, area))
        
        return sorted(plate_regions, key=lambda x: x[4], reverse=True)  # Alana göre sırala
    
    def validate_plate_format(self, plate_text: str) -> Tuple[bool, float]:
        """Plaka formatının geçerliliğini kontrol eder"""
        if not plate_text or len(plate_text) < 5:
            return False, 0.0
        
        # Temizleme
        cleaned_plate = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
        
        confidence = 0.0
        
        # Türkiye formatlarını kontrol et
        for pattern in self.valid_patterns:
            if re.match(pattern, cleaned_plate):
                confidence = 0.9
                return True, confidence
        
        # Global formatları kontrol et
        for pattern in self.global_patterns:
            if re.match(pattern, cleaned_plate):
                confidence = 0.7
                return True, confidence
        
        return False, 0.0
    
    def calculate_text_quality(self, text: str) -> float:
        """Metin kalitesini değerlendirir"""
        if not text:
            return 0.0
        
        quality_score = 1.0
        
        # Özel karakterler varsa kaliteyi düşür
        special_chars = len(re.findall(r'[^A-Z0-9]', text))
        quality_score -= (special_chars * 0.1)
        
        # Çok kısa veya çok uzun metinleri cezalandır
        if len(text) < 5 or len(text) > 10:
            quality_score -= 0.3
        
        # Alt çizgi veya belirsiz karakterler
        if '_' in text or '?' in text:
            quality_score -= 0.5
        
        return max(0.0, quality_score)
    
    def analyze_multiple_detections(self, detections: List[str]) -> Dict:
        """Çoklu tespit durumunu analiz eder"""
        if len(detections) == 0:
            return {
                'status': 'no_plate',
                'message': 'Plaka bulunamadı',
                'confidence': 0.0
            }
        
        if len(detections) > self.max_plates_allowed:
            return {
                'status': 'multiple_plates',
                'message': f'Çoklu plaka tespit edildi ({len(detections)} adet)',
                'detections': detections,
                'confidence': 0.0
            }
        
        return {
            'status': 'single_plate',
            'detections': detections,
            'confidence': 1.0
        }
    
    def run_detection(self, image_path: str) -> Dict:
        """Ana tespit fonksiyonu"""
        try:
            # Görüntüyü yükle ve ön işle
            image, gray = self.preprocess_image(image_path)
            
            # Modeli çalıştır
            raw_results = self.model.run(image_path)
            
            # Sonuçları temizle
            cleaned_results = []
            for result in raw_results:
                if result and isinstance(result, str):
                    cleaned = result.strip().replace('_', '')
                    if cleaned:
                        cleaned_results.append(cleaned)
            
            # Çoklu tespit kontrolü
            multi_analysis = self.analyze_multiple_detections(cleaned_results)
            
            if multi_analysis['status'] != 'single_plate':
                return multi_analysis
            
            # Tek plaka için detaylı analiz
            plate_text = cleaned_results[0]
            
            # Format kontrolü
            is_valid_format, format_confidence = self.validate_plate_format(plate_text)
            
            # Metin kalitesi
            text_quality = self.calculate_text_quality(plate_text)
            
            # Genel güven skoru
            overall_confidence = (format_confidence * 0.6) + (text_quality * 0.4)
            
            # Plaka bölgesi tespiti (ek doğrulama)
            plate_regions = self.detect_plate_regions(gray)
            region_confidence = min(1.0, len(plate_regions) / 3.0) if plate_regions else 0.0
            
            # Final güven skoru
            final_confidence = (overall_confidence * 0.8) + (region_confidence * 0.2)
            
            # Karar verme
            if final_confidence < self.min_confidence_threshold or not is_valid_format:
                return {
                    'status': 'low_confidence',
                    'message': 'Plaka bulunamadı - düşük güven skoru',
                    'detected_text': plate_text,
                    'confidence': final_confidence,
                    'details': {
                        'format_valid': is_valid_format,
                        'format_confidence': format_confidence,
                        'text_quality': text_quality,
                        'region_confidence': region_confidence
                    }
                }
            
            return {
                'status': 'success',
                'message': 'Plaka başarıyla tanındı',
                'plate': plate_text,
                'confidence': final_confidence,
                'details': {
                    'format_valid': is_valid_format,
                    'format_confidence': format_confidence,
                    'text_quality': text_quality,
                    'region_confidence': region_confidence
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Hata oluştu: {str(e)}',
                'confidence': 0.0
            }

def main():
    model = LicensePlateRecognizer('cct-s-v1-global-model')
    validator = PlateRecognitionValidator(model)
    image_folder = "./output_plates"

    csv_success_path = "./plaka_kayitlari.csv"
    csv_lowconf_path = "./dusuk_guvenli_kayitlar.csv"

    seen_plates = set()

    with open(csv_success_path, mode="w", newline="", encoding="utf-8") as success_file, \
         open(csv_lowconf_path, mode="w", newline="", encoding="utf-8") as lowconf_file:
        
        success_writer = csv.writer(success_file)
        lowconf_writer = csv.writer(lowconf_file)

        # Başlıklar
        success_writer.writerow(["Görsel Adı", "Plaka", "Başarı Oranı"])
        lowconf_writer.writerow(["Görsel Adı", "Tespit Edilen", "Başarı Oranı"])

        for filename in os.listdir(image_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(image_folder, filename)
                result = validator.run_detection(image_path)

                status = result.get("status", "")
                confidence = round(result.get("confidence", 0.0), 2)

                if status == "success":
                    plate = result["plate"]

                    if plate not in seen_plates:
                        seen_plates.add(plate)
                        success_writer.writerow([filename, plate, confidence])
                        print(f"✅ Kaydedildi: {filename} ➜ {plate} | Güven: {confidence}")
                    else:
                        print(f"⏭️ Zaten var: {plate} ➜ Atlandı")
                
                elif status == "low_confidence":
                    detected_text = result.get("detected_text", "-")
                    lowconf_writer.writerow([filename, detected_text, confidence])
                    print(f"⚠️ Düşük güvenli ➜ {filename}: {detected_text} | {confidence}")
                
                else:
                    print(f"❌ {filename} ➜ {result.get('message', 'Tanımsız hata')}")


# Basit kullanım fonksiyonu
def detect_plate_with_validation(model, image_path: str) -> str:
    """Basit kullanım için wrapper fonksiyon"""
    validator = PlateRecognitionValidator(model)
    result = validator.run_detection(image_path)
    
    if result['status'] == 'success':
        return f"Plaka: {result['plate']}"
    else:
        return result['message']

if __name__ == "__main__":
    main()