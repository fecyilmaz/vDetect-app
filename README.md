"# vDetect-ocr-app" 

Çalıştırmak için ise oluşturduğunuz sanal ortamda;
    * ultralytics, (yolo11m.pt)
    * flask
    * fast-plate-ocr[onnx-gpu] veya fast-plate-ocr[onnx-openvino] 
kütüphanelerini pip komutu ile indirmeniz gerekmektedir. Buradaki işlemlerde onnx-openvino kullanılmıştır. 
modelin ise cct-s-v1-global-model s boyutunda olanı kullanılmıştır.

YolKamerası.mp4 isimli video'yu indirip dosya dizinine eklemeniz gerekmektedir:
https://khenda-my.sharepoint.com/personal/necati_er_khenda_com/_layouts/15/stream.aspx?id=%2Fpersonal%2Fnecati%5Fer%5Fkhenda%5Fcom%2FDocuments%2FStaj2025%2FYolKameras%C4%B1%2Emp4&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2Eebe924c3%2Dc626%2D475a%2D8681%2D93803a3ea54c


Çalıştırmak için sadece uygun sql server'ını oluşturup bağlantısını yapınız yukarıdaki gereklilikleri de kurduktan sonra yalnızca python web_app.py'ı çalıştırınız. 



Aşağıdaki kısım eski versiyonu içindir:
İlk önce select_roi.py isimli dosyayı çalıştırınız, ve ilgili alanı seçiniz.
Devamında roi_car_detection.py isimli dosyayı çalıştırınız. Bu kısım videodaki araçları tespit etmektedir.
Ardından roi_plate.py dosyasını çalıştırarak araçların plakalarının tespitini yapıp bir dosyaya kaydedilmesini sağlayacaksınız. Videoda tespit edilen her plaka output_plates klasörüne kaydedilmektedir.
En sonunda ise app.py dosyasını çalıştırarak plate_validator.py dosyasını çalıştırmış olacaksınız. Burada ise output_plate klasörüne kaydedilen .img uzantılı plaka fotoğrafları için OCR işlemi yapmış olacaksınız.
Tüm işlemlerin bitiminde tespit edilen plakalar'dan tekrarlanmayanlar "./plaka_kayitlari.csv" bu isimdeki csv dosyasına, plaka bulunup confidence rate'i düşükse "./dusuk_guvenli_kayitlar.csv klasörüne kaydedilecektir. Tekrarlananlar ise hiçbir şekilde kaydedilmeyecektir. En son terminalde atlandı olarak görebilmektesiniz.