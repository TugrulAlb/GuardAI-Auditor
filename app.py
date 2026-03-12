import streamlit as st
import os
import tempfile
import shutil
from pypdf import PdfReader

# auditor.py içerisinden gerekli yapıları import edelim
from auditor import app as workflow_app, AuditState, setup_chroma

# ---------------------------------------------------------
# Sayfa Konfigürasyonu
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Compliance & Security Auditor",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------
# Yardımcı Fonksiyonlar
# ---------------------------------------------------------
def extract_text_from_pdf(file) -> str:
    """PDF BytesIO nesnesinden düz metin çıkarır."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\\n"
    return text

# ---------------------------------------------------------
# Sidebar (Yan Panel) - RAG / Politikalar
# ---------------------------------------------------------
with st.sidebar:
    st.header("📚 Kurumsal Politikalar (RAG)")
    st.write("Şirketin KVKK ve Güvenlik politikalarını (PDF) buraya yükleyin.")
    
    policy_files = st.file_uploader(
        "Politika PDF'lerini Seçin", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Politikaları Hafızaya Al (RAG)"):
        if policy_files:
            with st.spinner("Politikalar yükleniyor ve ChromaDB'ye kaydediliyor..."):
                # Önceki ChromaDB'yi temizleyerek yepyeni bir bilgi tabanı oluşturalım
                if os.path.exists("./chroma_db"):
                    shutil.rmtree("./chroma_db")
                    
                # Yüklenen dosyaları ana dizine kaydedelim ki setup_chroma() bulabilsin
                for file in policy_files:
                    with open(file.name, "wb") as f:
                        f.write(file.getbuffer())
                
                try:
                    setup_chroma()
                    st.success("Politikalar başarıyla eklendi! RAG katmanı hazır.")
                except Exception as e:
                    st.error(f"Hata oluştu: {str(e)}")
        else:
            st.warning("Lütfen en az bir PDF politikası yükleyin.")

# ---------------------------------------------------------
# Ana Ekran
# ---------------------------------------------------------
st.title("🛡️ AI Compliance & Security Auditor")
st.markdown("Dokümanlarınızı yükleyin veya metin girin, KVKK ve güvenlik kurallarına göre analiz edelim.")

# Düzen için iki sütun oluşturuyoruz
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📄 Denetlenecek Doküman")
    
    input_method = st.radio("Veri Giriş Yöntemi", ["Metin Gir", "PDF Yükle"], horizontal=True)
    
    text_to_audit = ""
    
    if input_method == "Metin Gir":
        text_to_audit = st.text_area(
            "Denetlenecek metni buraya yapıştırın",
            height=300,
            placeholder="Müşteri Ahmet Yılmaz'ın sipariş teyidi alındı. İletişim: 05551234567..."
        )
    else:
        uploaded_doc = st.file_uploader("Denetlenecek PDF Dosyası", type=["pdf"])
        if uploaded_doc:
            with st.spinner("PDF okunuyor..."):
                text_to_audit = extract_text_from_pdf(uploaded_doc)
                st.success("PDF başarıyla okundu!")
                with st.expander("Okunan Orijinal Metin"):
                    st.write(text_to_audit)
                
    start_audit = st.button("Denetimi Başlat 🚀", use_container_width=True, type="primary")

# ---------------------------------------------------------
# Analiz Sonuçları / Rapor
# ---------------------------------------------------------
with col_results:
    st.subheader("🔍 Analiz Çıktısı")
    
    if start_audit:
        if not text_to_audit.strip():
            st.warning("Lütfen denetlenecek bir metin girin veya PDF yükleyin.")
        else:
            with st.spinner("LangGraph Agent çalıştırılıyor... Analiz ediliyor..."):
                initial_state = AuditState(
                    original_text=text_to_audit,
                    scrubbed_text="",
                    audit_report={},
                    confidence_score=0.0,
                    iteration=0,
                    error_message=""
                )
                
                # Langgraph'ı çalıştır
                try:
                    result = workflow_app.invoke(initial_state)
                    
                    # Verileri çekelim
                    scrubbed_text = result.get("scrubbed_text", "")
                    report_data = result.get("audit_report", {})
                    confidence = result.get("confidence_score", 0.0)
                    error_msg = result.get("error_message", "")
                    
                    if error_msg:
                        st.error(f"Denetim sırasında bir hata oluştu: {error_msg}")
                    elif not report_data:
                        st.error("LLM'den sonuç alınamadı.")
                    else:
                        is_compliant = report_data.get("is_compliant", False)
                        risk_level = report_data.get("risk_level", "Bilinmiyor")
                        violations = report_data.get("violations", [])
                        
                        # Güven Skoru
                        st.metric(label="AI Güven Skoru", value=f"%{int(confidence*100)}")
                        
                        # Uyum & Risk
                        if is_compliant:
                            st.success(f"✅ Uyumlu | Risk Seviyesi: {risk_level}")
                        else:
                            # Risk seviyesine göre renk 
                            if risk_level.lower() in ["kritik", "critical"]:
                                st.error(f"❌ Uyumsuz | Risk Seviyesi: {risk_level}")
                            elif risk_level.lower() in ["yüksek", "high"]:
                                st.warning(f"❌ Uyumsuz | Risk Seviyesi: {risk_level}", icon="⚠️")
                            else:
                                st.info(f"❌ Uyumsuz | Risk Seviyesi: {risk_level}")
                        
                        # İhlaller (Violations)
                        if violations:
                            st.markdown("#### Bulunan İhlaller:")
                            for vio in violations:
                                st.error(f"- {vio}")
                        elif is_compliant:
                            st.info("Herhangi bir ihlal bulunamadı.")
                            
                        # Maskelenmiş metin
                        with st.expander("Görüntüle: Maskelenmiş (Scrubbed) Metin", expanded=False):
                            st.markdown(f"*{scrubbed_text}*")
                            
                except Exception as e:
                    st.error(f"Uygulama Hatası: {str(e)}")
    else:
        st.info("Sonuçları görmek için sol taraftan veri girin ve işlemi başlatın.")
