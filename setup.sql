
-- 1. Veritabanı oluştur
DROP DATABASE IF EXISTS proje;
CREATE DATABASE IF NOT EXISTS proje;
USE proje;

-- 2. Videolar tablosu
CREATE TABLE IF NOT EXISTS videos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    filepath VARCHAR(255) NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Plaka tespitleri tablosu
CREATE TABLE IF NOT EXISTS plate_detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    video_id INT NOT NULL,
    plate_text VARCHAR(255),
    vehicle_type VARCHAR(255),
    confidence_score FLOAT,
    plate_image_path VARCHAR(255), 
    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

-- 4. İndeksler
CREATE INDEX idx_plate_text ON plate_detections (plate_text);
CREATE INDEX idx_vehicle_type ON plate_detections (vehicle_type);