import mysql.connector
from mysql.connector import errorcode

# Veritabanı bağlantı bilgilerini buraya girin.
# Bu bilgileri kendi MySQL sunucunuza göre düzenlemelisiniz.
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',       # MySQL kullanıcı adınız
    'password': '',       # MySQL şifreniz (varsayılan olarak boş)
    'database': 'proje'
}

def get_db_connection():
    """
    Veritabanı bağlantısını kurar ve bağlantı nesnesini döndürür.
    Bağlantı hatası durumunda hata mesajı yazdırır ve None döner.
    """
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Kullanıcı adı veya şifre hatalı.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Veritabanı mevcut değil.")
        else:
            print(f"Bağlantı hatası: {err}")
        return None

if __name__ == '__main__':
    # Dosyanın bağımsız olarak çalışıp çalışmadığını test etmek için
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        print("Bağlantı başarılı. Test için veritabanı adı getiriliyor...")
        
        cursor.execute("SELECT DATABASE()")
        db_name = cursor.fetchone()[0]
        print(f"Bağlanılan veritabanı: {db_name}")

        cursor.close()
        conn.close()
    else:
        print("Veritabanına bağlanılamadı. Lütfen ayarlarınızı kontrol edin.")