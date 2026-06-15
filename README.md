# 🌍 Deprem Analiz Sistemi (Earthquake Precursor Prediction)

AFAD verileri ve tarihsel sismik kayıtlar kullanılarak, makine öğrenmesi modelleriyle **öncü deprem (foreshock) olasılığını hesaplayan** otomatik bir analitik sistemidir. 

Sistem, GitHub Actions aracılığıyla her 6 saatte bir otomatik olarak çalışır, canlı verileri çeker ve elde ettiği güncel analizleri HTML raporları halinde `docs/` klasöründe sunar.

## 🚀 Proje Hakkında
Sismik hareketlerin doğası gereği, öncü deprem verileri oldukça seyrek ve dengesiz bir dağılım gösterir. Bu proje, model eğitiminde karşılaşılan bu dengesiz veri seti problemini **SMOTE (Synthetic Minority Over-sampling Technique)** algoritması ile çözerek istikrarlı bir öğrenme ortamı sağlar. Tarihsel veriler üzerinden öncü deprem olasılıkları hesaplanırken, algoritmaların parametreleri en yüksek **F1-score** ve **ROC-AUC** değerlerini verecek şekilde sürekli olarak optimize edilmektedir.

## ✨ Temel Özellikler

* **Otomatize Edilmiş Veri Akışı:** AFAD apiv2 (`https://deprem.afad.gov.tr/apiv2/event/filter`) üzerinden çekilen canlı sismik veri ile `earthquakes_3_5_plus_scientific_v5.db` veri tabanının sürekli güncel tutulması.
* **Gelişmiş Makine Öğrenmesi Modelleri:** * XGBoost ve Random Forest (Platt sigmoid kalibrasyonu ile).
  * Zaman serisi analizi için LSTM ağları (Monte Carlo Dropout ile epistemik belirsizlik tahmini).
  * Modellerin belirsizlik ağırlıklı ortalaması (XGB:0.40, RF:0.20, LSTM:0.40·exp(-σ/10)) ile oluşturulan güçlü Ensemble yapı.
* **Dengesiz Veri Çözümü & Optimizasyon:** Seyrek veriler için SMOTE entegrasyonu ve Optuna ile uçtan uca hiperparametre optimizasyonu.
* **Görsel Raporlama:** Molchan diyagramları, risk olasılık haritaları ve detaylı analitik HTML çıktıları üretimi.

## ⚙️ Kurulum (Setup)

Projeyi kendi bilgisayarınızda çalıştırmak için öncelikle gerekli kütüphaneleri yüklemelisiniz:

```bash
pip install -r requirements.txt
