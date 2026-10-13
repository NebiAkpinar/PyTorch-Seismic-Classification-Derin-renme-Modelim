# PyTorch ile Sismik Aktivite Sınıflandırması

Bu proje, derin öğrenme (Deep Learning) alanına attığım ilk adımı temsil etmektedir. Projenin amacı, temel yapay sinir ağı konseptlerini pratik ederek öğrenmek ve ikili sınıflandırma (binary classification) problemlerine PyTorch ile çözüm üretmektir.

## Proje Özeti

Model, **08-seismic_activity_svm.csv** veri setindeki bilgileri kullanarak sismik bir olayın gerçekleşip gerçekleşmediğini tahmin etmektedir. Bağımsız değişkenler olarak `underground_wave_energy` (yeraltı dalga enerjisi) ve `vibration_axis_variation` (titreşim ekseni değişimi) kullanılmış; hedef değişken olan `seismic_event_detected` (0 veya 1) tahmin edilmeye çalışılmıştır.

## Kullanılan Teknolojiler ve Araçlar

- **Derin Öğrenme Çerçevesi:** PyTorch (`torch.nn`)
- **Veri İşleme ve Makine Öğrenmesi:** Pandas, Scikit-learn (`train_test_split`)
- **Veri Görselleştirme:** Matplotlib, Seaborn

## Model Mimarisi ve Eğitim Detayları

- **Ağ Yapısı:** 2 giriş özelliğine sahip (`in_features=2`), 5 nöronlu tek bir gizli katmandan (ReLU aktivasyonu ile) ve 1 çıkıştan (`out_features=1`) oluşan ileri beslemeli bir yapay sinir ağı (`ClassificationModel`) tasarlanmıştır.
- **Kayıp Fonksiyonu (Loss Function):** İkili sınıflandırma için logit değerleriyle çalışan `BCEWithLogitsLoss` tercih edilmiştir.
- **Optimizasyon:** Adam optimizer kullanılmış olup öğrenme oranı (learning rate) **0.001** olarak ayarlanmıştır.
- **Eğitim Döngüsü:** Veri seti %80 eğitim, %20 test olarak ayrılmış; model toplam **120 epoch** boyunca eğitilerek kayıp (loss) ve doğruluk (accuracy) metrikleri gözlemlenmiştir.
