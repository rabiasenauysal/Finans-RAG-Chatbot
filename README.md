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
  
## Deploy & Web Arayüzü
-Gradio demo linki: https://9c63a6ef8548b55e9a.gradio.live
<img width="1917" height="966" alt="Screenshot 2025-10-29 153550" src="https://github.com/user-attachments/assets/8281027e-7cf2-4855-b76f-c7d3adfcf254" />


# DÜZELTME!!!!!!!!!!!!!!!!
-Streamlit demo linki: https://finans-rag-chatbot-gpxgexdxs49bpfuydotrd8.streamlit.app/
<img width="1919" height="1017" alt="Screenshot 2025-10-29 160105" src="https://github.com/user-attachments/assets/789a431f-8a60-4f26-9ab2-d47f275d4fc7" />





