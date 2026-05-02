# Deprem Analiz

AFAD verileri kullanilarak makine ogrenmesi modelleriyle deprem oncu (foreshock) olasiligi tahmin eden otomatik analiz sistemi.

GitHub Actions her 6 saatte bir otomatik calisir ve HTML ciktilarini `docs/` klasorunde gunceller.

## Kurulum (Setup)

```bash
pip install -r requirements.txt
```

## Calistirma (Run)

```bash
python DTS_66_AFAD_LIVE.py
```

## Repo'da Yer Alan Veri Dosyalari (Tracked Data Files)

| Dosya | Aciklama |
|---|---|
| `earthquakes_3_5_plus_scientific_v5.db` | SQLite katalog veritabani (her calistirmada AFAD apiv2 ile guncellenir) |
| `ridgecrest_catalog.csv` | Tarihi referans katalog (ilk seed) |

## Uretilen Dosyalar (Generated Artifacts — repoda tutulmaz)

Asagidaki dosyalar `.gitignore` ile haric tutulur ve ilk calistirmada otomatik olarak uretilir:

| Dosya | Aciklama |
|---|---|
| `xgb_v5.joblib` | Egitilmis XGBoost modeli (kalibre edilmis) |
| `rf_v5.joblib` | Egitilmis Random Forest modeli (kalibre edilmis) |
| `lstm_v5.keras` | Egitilmis LSTM agi |
| `lstm_scaler_v5.joblib` | LSTM ozellik olcekleyicisi |
| `deprem_analiz_raporu_sade.html` | HTML rapor |
| `deprem_haritasi_tip.html` | Deprem tipi haritasi |
| `deprem_haritasi_olasilik.html` | Risk olasiligi haritasi |
| `molchan_*.png` | Molchan diyagramlari |

> **Not:** Ilk calistirma uzun surebilir (model egitimi). Sonraki calistirmalarda mevcut modeller yeniden kullanilir; yalnizca yeni veri geldiginde yeniden egitim tetiklenir.

## Veri Kaynaklari (Data Sources)

- **AFAD apiv2** — Canli sismik katalog (`https://deprem.afad.gov.tr/apiv2/event/filter`)
- **`ridgecrest_catalog.csv`** — Tarihi referans katalog (repoda mevcut)

## Modeller (Models)

- **XGBoost** + **Random Forest** (Optuna ile hiperparametre optimizasyonu, Platt sigmoid kalibrasyon)
- **LSTM** (Monte Carlo Dropout ile epistemik belirsizlik tahmini)
- Ensemble: belirsizlik agirlikli ortalama (XGB:0.40, RF:0.20, LSTM:0.40·exp(-σ/10))

## Performans Metrikleri (Performance Metrics)

ROC-AUC, Brier skoru, ECE (kalibrasyon hatasi), Molchan beceri skoru, prospektif simulasyon dogrulugu.
