"""
Regresyon Modelleri için Hata Değerlendirme
--------------------------------------------
Miuul Data Scientist Bootcamp - Ödev Çözümü

Problem:
    Çalışanların deneyim yılı (x) ve maaş (y) bilgileri verilmiştir.
    1) Verilen bias (b=275) ve weight (w=90) değerlerine göre doğrusal
       regresyon model denklemi oluşturulur: y' = b + w*x
    2) Bu denklem ile tüm gözlemler için maaş tahmini yapılır.
    3) Modelin başarısı MSE, RMSE ve MAE metrikleri ile ölçülür.

Kullanım:
    python src/main.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Sabitler ve yol tanımları
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "calisan_maas.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

BIAS = 275      # b
WEIGHT = 90     # w


# --------------------------------------------------------------------------- #
# Adım 1: Veri setini yükle
# --------------------------------------------------------------------------- #
def veri_setini_yukle(yol: Path = DATA_PATH) -> pd.DataFrame:
    """Deneyim yılı - maaş veri setini CSV dosyasından okur."""
    return pd.read_csv(yol)


# --------------------------------------------------------------------------- #
# Adım 2: Model denklemi ile tahmin yap ve hata sütunlarını üret
# --------------------------------------------------------------------------- #
def maas_tahmini_yap(df: pd.DataFrame, bias: float, weight: float) -> pd.DataFrame:
    """
    y' = b + w * x doğrusal regresyon denklemine göre maaş tahmini yapar
    ve her gözlem için hata, hata karesi ile mutlak hata değerlerini ekler.
    """
    sonuc = df.copy()
    sonuc["maas_tahmini"] = bias + weight * sonuc["deneyim_yili"]
    sonuc["hata"] = sonuc["maas"] - sonuc["maas_tahmini"]
    sonuc["hata_karesi"] = sonuc["hata"] ** 2
    sonuc["mutlak_hata"] = sonuc["hata"].abs()
    return sonuc


# --------------------------------------------------------------------------- #
# Adım 3: Hata değerlendirme metriklerini hesapla (MSE, RMSE, MAE)
# --------------------------------------------------------------------------- #
def hata_metriklerini_hesapla(df: pd.DataFrame) -> dict:
    """
    MSE  = (1/n) * sum((y - y')^2)
    RMSE = sqrt(MSE)
    MAE  = (1/n) * sum(|y - y'|)
    """
    n = len(df)
    mse = df["hata_karesi"].sum() / n
    rmse = np.sqrt(mse)
    mae = df["mutlak_hata"].sum() / n
    return {"gozlem_sayisi": n, "MSE": mse, "RMSE": rmse, "MAE": mae}


# --------------------------------------------------------------------------- #
# Adım 4: Sonuçları görselleştir
# --------------------------------------------------------------------------- #
def gorsellestir(df: pd.DataFrame, bias: float, weight: float, kayit_yolu: Path) -> None:
    """Regresyon doğrusunu ve gerçek/tahmin karşılaştırmasını çizip kaydeder."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sol grafik: veri noktaları + regresyon doğrusu
    x_sirali = np.linspace(df["deneyim_yili"].min(), df["deneyim_yili"].max(), 100)
    y_sirali = bias + weight * x_sirali

    axes[0].scatter(df["deneyim_yili"], df["maas"], color="#2563eb", label="Gerçek Değerler")
    axes[0].plot(x_sirali, y_sirali, color="#dc2626", linewidth=2,
                 label=f"y' = {bias} + {weight}x")
    axes[0].set_xlabel("Deneyim Yılı (x)")
    axes[0].set_ylabel("Maaş (y)")
    axes[0].set_title("Doğrusal Regresyon Modeli")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Sağ grafik: gerçek vs tahmin (y=x referans doğrusu ile)
    min_v = min(df["maas"].min(), df["maas_tahmini"].min())
    max_v = max(df["maas"].max(), df["maas_tahmini"].max())
    axes[1].scatter(df["maas"], df["maas_tahmini"], color="#16a34a")
    axes[1].plot([min_v, max_v], [min_v, max_v], color="#dc2626", linestyle="--",
                 label="Mükemmel Tahmin (y=y')")
    axes[1].set_xlabel("Gerçek Maaş (y)")
    axes[1].set_ylabel("Tahmini Maaş (y')")
    axes[1].set_title("Gerçek vs. Tahmin Edilen Değerler")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    kayit_yolu.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(kayit_yolu, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def main() -> None:
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)

    df = veri_setini_yukle()

    print("=" * 72)
    print(f"REGRESYON MODEL DENKLEMİ:  y' = {BIAS} + {WEIGHT} * x")
    print("=" * 72)

    sonuc_df = maas_tahmini_yap(df, BIAS, WEIGHT)
    print("\nTahmin Tablosu:\n")
    print(sonuc_df.to_string(index=False))

    metrikler = hata_metriklerini_hesapla(sonuc_df)
    print("\n" + "-" * 72)
    print("HATA DEĞERLENDİRME METRİKLERİ")
    print("-" * 72)
    print(f"Gözlem Sayısı (n) : {metrikler['gozlem_sayisi']}")
    print(f"MSE               : {metrikler['MSE']:.4f}")
    print(f"RMSE              : {metrikler['RMSE']:.4f}")
    print(f"MAE               : {metrikler['MAE']:.4f}")

    # Çıktıları kaydet
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sonuc_df.to_csv(OUTPUT_DIR / "tahmin_sonuclari.csv", index=False)
    gorsellestir(sonuc_df, BIAS, WEIGHT, OUTPUT_DIR / "regresyon_grafik.png")

    # scikit-learn ile doğrulama (kontrol amaçlı)
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    sk_mse = mean_squared_error(sonuc_df["maas"], sonuc_df["maas_tahmini"])
    sk_mae = mean_absolute_error(sonuc_df["maas"], sonuc_df["maas_tahmini"])
    print("\n[Doğrulama - scikit-learn]")
    print(f"MSE  (sklearn) : {sk_mse:.4f}")
    print(f"RMSE (sklearn) : {np.sqrt(sk_mse):.4f}")
    print(f"MAE  (sklearn) : {sk_mae:.4f}")

    print(f"\nSonuçlar '{OUTPUT_DIR}' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
