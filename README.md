# Finans RAG Chatbot


Bu proje, açık kaynak modeller kullanılarak hazırlanmış bir RAG (Retrieval-Augmented Generation) chatbottur.


## Amaç
Finans konularında (yatırım, enflasyon, faiz, döviz, kripto vb.) kullanıcı sorularını, küçük bir doküman koleksiyonundan yararlanarak cevaplamak.


## Veri Seti
Proje örnek amaçlı küçük bir metin koleksiyonu kullanır. İstendiğinde daha büyük bir dataset (HuggingFace, web scraping vb.) ile genişletilebilir.


## Kullanılan Yöntemler
- Embedding: sentence-transformers/all-MiniLM-L6-v2
- Vektör Veri Tabanı: FAISS (IndexFlatL2)
- Yaratım (Generator): google/flan-t5-base (transformers pipeline)
- Arayüz: Gradio


## Kurulum & Çalıştırma
1. GitHub'dan klonlayın veya dosyaları indirin.
2. `pip install -r requirements.txt`
3. `python <script>.py` veya `jupyter/colab` üzerinde notebook'u çalıştırın.


## Kullanım
- Colab üzerinde tüm hücreleri çalıştırın.
- Gradio arayüzü açıldığında soru girip test edin.


## Deploy
- İsterseniz bu Gradio uygulamasını HuggingFace Spaces veya başka bir ortamda deploy edebilirsiniz.
