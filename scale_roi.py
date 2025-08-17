import json

def scale_roi_points(points, scale_factor):
    """
    Verilen noktaları (x, y) bir ölçek faktörü ile çarpar ve tam sayıya yuvarlar.
    """
    return [(int(x * scale_factor), int(y * scale_factor)) for (x, y) in points]

def main():
    input_file = 'roi_points.json'
    output_file = 'roi_points_scaled.json'

    # ROI noktalarını JSON dosyasından oku
    try:
        with open(input_file, 'r') as f:
            points = json.load(f)
        if not points or not isinstance(points, list):
            print(f"Hata: '{input_file}' dosyasında geçerli nokta listesi yok.")
            return
        print(f"'{input_file}' dosyasından {len(points)} nokta okundu.")
    except FileNotFoundError:
        print(f"Hata: '{input_file}' bulunamadı.")
        return
    except json.JSONDecodeError:
        print(f"Hata: '{input_file}' geçerli bir JSON formatında değil.")
        return
    except Exception as e:
        print(f"Noktalar okunurken bir hata oluştu: {e}")
        return

    # Ölçek faktörünü kullanıcıdan al
    while True:
        try:
            scale_factor_str = input("Ölçek faktörünü girin (örn. 1.2 büyütmek için, 0.8 küçültmek için): ")
            scale_factor = float(scale_factor_str)
            if scale_factor <= 0:
                print("Hata: Ölçek faktörü sıfırdan büyük olmalıdır.")
            else:
                break
        except ValueError:
            print("Hata: Geçersiz giriş. Lütfen sayı girin (örn. 1.25).")
        except Exception as e:
            print(f"Ölçek faktörü alınırken bir hata oluştu: {e}")

    # Noktaları ölçeklendir
    scaled_points = scale_roi_points(points, scale_factor)

    # Ölçeklendirilmiş noktaları yeni dosyaya kaydet
    try:
        with open(output_file, 'w') as f:
            json.dump(scaled_points, f, indent=4)
        print(f"{len(scaled_points)} nokta '{output_file}' dosyasına kaydedildi.")
        for p in scaled_points:
            print(p)
    except IOError as e:
        print(f"Hata: '{output_file}' dosyasına yazılırken bir sorun oluştu: {e}")
    except Exception as e:
        print(f"Noktalar kaydedilirken bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
