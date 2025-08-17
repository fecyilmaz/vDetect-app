import os
import threading
import json
import cv2
import mysql.connector
import csv

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename

# Kendi video işleme modülünüz
# run.py içindeki process_video fonksiyonunu içeri aktarıyoruz.
from run import process_video

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sizin_cok_gizli_anahtariniz_burada'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['PLATE_OUTPUT_FOLDER'] = os.path.join('static', 'plate_vehicle_outputs')

# Klasörler yoksa oluştur
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['PLATE_OUTPUT_FOLDER']):
    os.makedirs(app.config['PLATE_OUTPUT_FOLDER'])

DB_CONFIG = {
    'host': "localhost",
    'user': "root",
    'password': "",
    'database': "proje"
}

def get_db():
    try:
        mydb = mysql.connector.connect(**DB_CONFIG)
        cursor = mydb.cursor()
        return mydb, cursor
    except mysql.connector.Error as err:
        print(f"Veritabanı bağlantı hatası: {err}")
        return None, None

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/plate_outputs/<path:filename>')
def plate_output_file(filename):
    """
    Bu fonksiyon, plate_vehicle_outputs klasöründeki dosyaları sunar.
    `path:filename` kullanılarak alt dizinleri de işleyebilir.
    """
    # Dosya yolunu güvence altına alıyoruz, ancak `run.py` zaten tam yolu kaydettiği için 
    # `send_from_directory` doğru çalışacaktır.
    return send_from_directory(app.config['PLATE_OUTPUT_FOLDER'], filename)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    mydb, cursor = get_db()
    if not mydb:
        flash("Veritabanına bağlanılamadı! Lütfen MySQL servisinin çalıştığını kontrol edin.")
        return redirect(url_for('home'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Dosya bulunamadı')
            cursor.close()
            mydb.close()
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Dosya seçilmedi')
            cursor.close()
            mydb.close()
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            cursor.execute("INSERT INTO videos (filename, filepath) VALUES (%s, %s)", (filename, filepath))
            mydb.commit()
            video_id = cursor.lastrowid
            
            flash('Video başarıyla yüklendi!')
            cursor.close()
            mydb.close()
            return redirect(url_for('select_roi', video_id=video_id))
    
    cursor.execute("SELECT id, filename, upload_date FROM videos ORDER BY id DESC")
    videos = cursor.fetchall()
    cursor.close()
    mydb.close()
    return render_template('upload_video.html', videos=videos)

@app.route('/select_roi/<int:video_id>')
def select_roi(video_id):
    mydb, cursor = get_db()
    if not mydb:
        flash("Veritabanına bağlanılamadı!")
        return redirect(url_for('home'))
    
    cursor.execute("SELECT filepath FROM videos WHERE id = %s", (video_id,))
    result = cursor.fetchone()
    cursor.close()
    mydb.close()
    
    if not result:
        flash("Video bulunamadı!")
        return redirect(url_for('upload_file'))

    video_path = result[0]
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        flash("Video karesi alınamadı!")
        return redirect(url_for('upload_file'))

    frame_filename = f"roi_frame_{video_id}.jpg"
    frame_path_full = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
    cv2.imwrite(frame_path_full, frame)

    return render_template("select_roi.html", video_id=video_id, frame_path=frame_filename)

@app.route('/save_roi/<int:video_id>', methods=['POST'])
def save_roi(video_id):
    data = request.get_json()
    points = data.get("points", [])

    if len(points) < 3:
        return jsonify({"message": "En az 3 nokta gerekli."}), 400

    roi_path = os.path.join(app.config['UPLOAD_FOLDER'], f"roi_points_{video_id}.json")
    with open(roi_path, "w") as f:
        json.dump(points, f)

    return jsonify({"message": "ROI başarıyla kaydedildi."})

@app.route('/analyze_video/<int:video_id>')
def analyze_video(video_id):
    mydb, cursor = get_db()
    if not mydb:
        flash("Veritabanına bağlanılamadı!")
        return redirect(url_for('home'))

    cursor.execute("SELECT filepath FROM videos WHERE id = %s", (video_id,))
    result = cursor.fetchone()
    cursor.close()
    mydb.close()
    
    if not result:
        flash("Video bulunamadı!")
        return redirect(url_for('upload_file'))

    video_path = result[0]
    
    roi_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"roi_points_{video_id}.json")
    if not os.path.exists(roi_file_path):
        flash("Lütfen önce ROI (İlgili Bölge) belirleyin!")
        return redirect(url_for('select_roi', video_id=video_id))
    
    # process_video fonksiyonuna `PLATE_OUTPUT_FOLDER` yolunu gönderiyoruz.
    analysis_thread = threading.Thread(target=process_video, args=(video_path, video_id, app.config['PLATE_OUTPUT_FOLDER']))
    analysis_thread.start()

    flash("Video analizi başlatıldı. Sonuçlar birazdan görüntülenebilir.")
    return redirect(url_for('results', video_id=video_id))

@app.route('/results/<int:video_id>')
def results(video_id):
    mydb, cursor = get_db()
    if not mydb:
        flash("Veritabanına bağlanılamadı!")
        return redirect(url_for('home'))

    cursor.execute("SELECT filename FROM videos WHERE id = %s", (video_id,))
    video_result = cursor.fetchone()
    
    if not video_result:
        cursor.close()
        mydb.close()
        flash("İstenen video bulunamadı.")
        return redirect(url_for('upload_file'))
        
    video_filename = video_result[0]
    
    # NOTE: Sorgu, results.html'in ihtiyaç duyduğu tüm sütunları çekmeli.
    cursor.execute("SELECT id, video_id, plate_text, vehicle_type, confidence_score, detection_timestamp, plate_image_path FROM plate_detections WHERE video_id = %s ORDER BY detection_timestamp DESC", (video_id,))
    detections = cursor.fetchall()
    
    cursor.close()
    mydb.close()
    
    return render_template("results.html", detections=detections, video_filename=video_filename, video_id=video_id)

@app.route('/api/results/<int:video_id>')
def api_results(video_id):
    mydb, cursor = get_db()
    if not mydb:
        return jsonify({'error': 'DB bağlantı hatası'}), 500
    
    cursor.execute("SELECT plate_text, confidence_score FROM plate_detections WHERE video_id = %s ORDER BY detection_timestamp DESC", (video_id,))
    detections = cursor.fetchall()
    cursor.close()
    mydb.close()
    
    detections_list = [{'plate_text': d[0], 'confidence_score': f"{d[1]:.2f}"} for d in detections]
    return jsonify(detections_list)

@app.route('/plate/<string:plate_text>')
def plate_detail(plate_text):
    mydb, cursor = get_db()
    if not mydb:
        flash("Veritabanına bağlanılamadı!")
        return redirect(url_for('home'))

    try:
        # Düzeltildi: Sadece plaka metni ile değil, tüm kayıtları çekiyoruz.
        # Bu, benzersiz dosya yollarını (UUID içeren) almamızı sağlar.
        query = "SELECT id, video_id, plate_text, vehicle_type, confidence_score, detection_timestamp, plate_image_path, vehicle_image_path FROM plate_detections WHERE plate_text = %s ORDER BY detection_timestamp DESC"
        cursor.execute(query, (plate_text,))
        detections = cursor.fetchall()
        
        if not detections:
            flash(f"'{plate_text}' plakasına ait detaylı bilgi bulunamadı.")
            return redirect(url_for('upload_file'))

        # Düzeltildi: `render_template`'e doğru verileri gönderiyoruz.
        return render_template("plate_detail.html",
                               plate_text=plate_text,
                               detections=detections)
    
    except Exception as e:
        flash(f"Plaka detaylarını alırken bir hata oluştu: {e}")
        return redirect(url_for('upload_file'))
    finally:
        if cursor:
            cursor.close()
        if mydb:
            mydb.close()


@app.route('/export_csv/<int:video_id>')
def export_csv(video_id):
    mydb, cursor = get_db()
    if not mydb:
        flash("Veritabanına bağlanılamadı!")
        return redirect(url_for('home'))

    cursor.execute(
        "SELECT plate_text, vehicle_type, confidence_score, detection_timestamp FROM plate_detections WHERE video_id = %s ORDER BY detection_timestamp DESC",
        (video_id,))
    rows = cursor.fetchall()
    cursor.close()
    mydb.close()

    def generate():
        yield ','.join(['Plaka', 'Araç Tipi', 'Güven Skoru', 'Tespit Zamanı']) + '\n'
        for row in rows:
            yield ','.join(map(str, row)) + '\n'

    return Response(generate(), mimetype='text/csv',
                    headers={"Content-Disposition": f"attachment;filename=video_{video_id}_detections.csv"})


# ... diğer kodlar ...

@app.route('/delete_video/<int:video_id>', methods=['POST'])
def delete_video(video_id):
    mydb, cursor = get_db()
    if not mydb:
        flash("Veritabanına bağlanılamadı!")
        return redirect(url_for('upload_file'))

    try:
        # Veritabanından ilgili video dosya yolunu al
        cursor.execute("SELECT filepath FROM videos WHERE id = %s", (video_id,))
        video_info = cursor.fetchone()
        
        if video_info:
            video_path = video_info[0]

            # İlişkili plaka ve araç görsellerinin tam yollarını veritabanından al
            # NOT: Bu sorgu, run.py içindeki kayıt mantığıyla eşleşmelidir.
            cursor.execute("SELECT plate_image_path, vehicle_image_path FROM plate_detections WHERE video_id = %s", (video_id,))
            image_paths = cursor.fetchall()
            
            # Görselleri dosya sisteminden sil
            for (plate_img_path, vehicle_img_path) in image_paths:
                if plate_img_path and os.path.exists(plate_img_path):
                    os.remove(plate_img_path)
                    print(f"Silinen plaka görseli: {plate_img_path}")
                if vehicle_img_path and os.path.exists(vehicle_img_path):
                    os.remove(vehicle_img_path)
                    print(f"Silinen araç görseli: {vehicle_img_path}")
            
            # Veritabanı kayıtlarını sil
            cursor.execute("DELETE FROM plate_detections WHERE video_id = %s", (video_id,))
            cursor.execute("DELETE FROM videos WHERE id = %s", (video_id,))
            mydb.commit()

            # Video dosyasını sil
            if os.path.exists(video_path):
                os.remove(video_path)
            
            # ROI dosyalarını sil
            roi_json = os.path.join(app.config['UPLOAD_FOLDER'], f"roi_points_{video_id}.json")
            if os.path.exists(roi_json):
                os.remove(roi_json)
            
            roi_frame = os.path.join(app.config['UPLOAD_FOLDER'], f"roi_frame_{video_id}.jpg")
            if os.path.exists(roi_frame):
                os.remove(roi_frame)

            flash("Video ve ilgili tüm veriler başarıyla silindi.")
        else:
            flash("Silinecek video bulunamadı.")
            
    except Exception as e:
        mydb.rollback()
        print(f"Silme işlemi sırasında hata oluştu: {e}")
        flash(f"Silme işleminde bir hata oluştu: {e}")
    finally:
        if cursor:
            cursor.close()
        if mydb:
            mydb.close()
    
    return redirect(url_for('upload_file'))

# ... diğer kodlar ...

@app.route('/reset_results/<int:video_id>')
def reset_results(video_id):
    mydb, cursor = get_db()
    if not mydb:
        flash("Veritabanı bağlantı hatası!")
        return redirect(url_for('results', video_id=video_id))
    
    try:
        # İlişkili plaka ve araç görsellerinin tam yollarını veritabanından al
        # NOT: Bu sorgu, run.py içindeki kayıt mantığıyla eşleşmelidir.
        cursor.execute("SELECT plate_image_path, vehicle_image_path FROM plate_detections WHERE video_id = %s", (video_id,))
        image_paths = cursor.fetchall()
        
        # Görselleri dosya sisteminden sil
        for (plate_img_path, vehicle_img_path) in image_paths:
            if plate_img_path and os.path.exists(plate_img_path):
                os.remove(plate_img_path)
                print(f"Silinen plaka görseli: {plate_img_path}")
            if vehicle_img_path and os.path.exists(vehicle_img_path):
                os.remove(vehicle_img_path)
                print(f"Silinen araç görseli: {vehicle_img_path}")
        
        # Sadece plaka tespit verilerini veritabanından sil
        cursor.execute("DELETE FROM plate_detections WHERE video_id = %s", (video_id,))
        mydb.commit()
        
        flash(f"Video {video_id} için analiz sonuçları başarıyla sıfırlandı ve görseller temizlendi.")
    except Exception as e:
        mydb.rollback()
        flash(f"Sonuçları sıfırlama sırasında bir hata oluştu: {e}")
        print(f"Sonuçları sıfırlama hatası: {e}")
    finally:
        if cursor:
            cursor.close()
        if mydb:
            mydb.close()

    return redirect(url_for('results', video_id=video_id))

if __name__ == '__main__':
    app.run(debug=True)
# Bu kod, Flask uygulamasını başlatır.
# debug=True ile hata ayıklama modu aktif edilir.