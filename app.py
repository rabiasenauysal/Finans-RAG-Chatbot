import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

# =============================================================================
# SAYFA AYARLARI
# =============================================================================

st.set_page_config(
    page_title="📚 Ekonomi Terimleri Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL YÜKLEME (SESSION STATE İLE CACHE)
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model_and_chain():
    """Model ve RAG chain'i yükle (bir kez)"""
    
    # API Key
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    
    if not GOOGLE_API_KEY:
        st.error("⚠️ GOOGLE_API_KEY bulunamadı! Streamlit Cloud > Settings > Secrets'e ekleyin.")
        st.stop()
    
    # PDF yükleme
    pdf_path = "sozluk.pdf"
    
    if not os.path.exists(pdf_path):
        st.error(f"❌ '{pdf_path}' dosyası bulunamadı! PDF'i repo'ya ekleyin.")
        st.stop()
    
    with st.spinner("📄 PDF yükleniyor..."):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
    
    # Chunking
    with st.spinner("✂️ Metin parçalara ayrılıyor..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
    
    # Embedding
    with st.spinner("🧠 Embedding modeli yükleniyor..."):
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Vector Database
    with st.spinner("🔍 Vector Database oluşturuluyor (1-2 dakika)..."):
        vector_store = FAISS.from_documents(chunks, embeddings)
    
    # LLM ve Chain
    with st.spinner("🤖 RAG Chain kuruluyor..."):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    
    return qa_chain, len(chunks), len(docs)

# =============================================================================
# SESSION STATE BAŞLATMA
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    with st.spinner("🚀 Sistem başlatılıyor..."):
        qa_chain, chunk_count, page_count = load_model_and_chain()
        st.session_state.qa_chain = qa_chain
        st.session_state.chunk_count = chunk_count
        st.session_state.page_count = page_count

# =============================================================================
# SIDEBAR - BİLGİ PANELİ
# =============================================================================

with st.sidebar:
    st.title("📚 Ekonomi Terimleri")
    st.markdown("---")
    
    st.subheader("💡 Örnek Sorular")
    st.markdown("""
    - Arbitraj nedir?
    - Enflasyon ne demek?
    - Deflasyon açıkla
    - Merkez bankası ne iş yapar?
    - Devalüasyon nedir?
    - Revalüasyon ile devalüasyon farkı?
    - Döviz kuru nasıl belirlenir?
    """)
    
    st.markdown("---")
    st.subheader("ℹ️ Sistem Bilgisi")
    st.info(f"""
    **Model:** Gemini 2.0 Flash  
    **Embedding:** Multilingual MiniLM  
    **Vector DB:** FAISS  
    **Toplam Sayfa:** {st.session_state.page_count}  
    **Toplam Chunk:** {st.session_state.chunk_count}  
    **Retrieval:** Top-10 Similarity
    """)
    
    st.markdown("---")
    st.subheader("📊 Durum")
    st.success("""
    ✅ RAG Chain Aktif  
    🔍 Semantic Search Etkin  
    🌐 Türkçe Destek  
    📄 Kaynak Referanslı
    """)
    
    st.markdown("---")
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
    Made with ❤️ using<br>
    LangChain & Streamlit
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# ANA SAYFA - BAŞLIK
# =============================================================================

st.title("📚 Ekonomi Terimleri RAG Chatbot")
st.markdown("### Powered by Google Gemini 2.0 Flash & LangChain")
st.markdown("""
Bu chatbot, ekonomi sözlüğü PDF'inden bilgi çekerek sorularınızı yanıtlar.
**RAG (Retrieval Augmented Generation)** teknolojisi kullanır.
""")
st.markdown("---")

# =============================================================================
# CHAT MESAJLARINI GÖSTER
# =============================================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =============================================================================
# KULLANICI GİRİŞİ VE CEVAP
# =============================================================================

if prompt := st.chat_input("Sorunuzu yazın (Örn: Arbitraj nedir?)"):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Bot cevabı
    with st.chat_message("assistant"):
        with st.spinner("🤔 Düşünüyorum..."):
            try:
                # RAG chain ile cevap al
                result = st.session_state.qa_chain.invoke({"query": prompt})
                answer = result['result']
                
                # Kaynak ekleme
                if 'source_documents' in result and len(result['source_documents']) > 0:
                    sources = set()
                    for doc in result['source_documents'][:3]:
                        page = doc.metadata.get('page', 'N/A')
                        sources.add(f"Sayfa {page}")
                    
                    if sources:
                        answer += f"\n\n📚 **Kaynaklar:** {', '.join(sorted(sources))}"
                
                st.markdown(answer)
                
                # Mesajı kaydet
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"❌ Bir hata oluştu: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; padding: 2rem;'>
<p>📚 Bu chatbot, PDF belgelerinden bilgi çekmek için RAG (Retrieval Augmented Generation) teknolojisini kullanır.</p>
<p>Sorularınız için en alakalı bilgileri bulur ve doğal dilde cevaplar.</p>
</div>
""", unsafe_allow_html=True)