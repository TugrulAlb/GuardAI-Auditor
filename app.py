import streamlit as st
import os
import tempfile
import shutil
from pypdf import PdfReader
from datetime import datetime, timezone
import json

from PIL import Image

import pytesseract

# auditor.py içerisinden gerekli yapıları import edelim
from auditor import (
    app as workflow_app,
    AuditState,
    setup_chroma,
    generate_pdf_report,
    save_audit_history,
    get_history,
    redact_pdf
)

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

def extract_text_from_image(image_bytes) -> str:
    """Resim dosyasından OCR ile metin çıkarır."""
    try:
        img = Image.open(image_bytes)
        # Türkçe dil desteği varsa, 'tur' dil kodunu kullan (Tesseract kurulumu gerektirir)
        try:
            text = pytesseract.image_to_string(img, lang="tur")
        except Exception:
            # Eğer Türkçe dili yüklü değilse varsayılan dili kullan
            text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"OCR sırasında hata oluştu: {e}")
        return ""


def save_uploaded_file(uploaded_file, dst_path: str) -> None:
    """UploadedFile objesini belirtilen path'e kaydeder."""
    with open(dst_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
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
# Ana Ekran & Sekmeler
# ---------------------------------------------------------
st.title("🛡️ AI Compliance \u0026 Security Auditor")

tab_new, tab_history = st.tabs(["Yeni Denetim","Eski Denetimler"])

with tab_new:
    st.markdown("Dokümanlarınızı yükleyin veya metin girin, KVKK ve güvenlik kurallarına göre analiz edelim.")

    # Düzen için iki sütun oluşturuyoruz
    col_input, col_results = st.columns([1, 1], gap="large")

with tab_history:
    st.subheader("📜 Geçmiş Denetimler")
    history = get_history()
    if history:
        df = []
        for rec in history:
            df.append({
                "id": rec.id,
                "date": rec.date,
                "file": rec.original_filename,
                "risk": rec.risk_level,
                "confidence": rec.confidence_score,
                "compliant": rec.compliance_status
            })
        st.dataframe(df)
        sel = st.selectbox("Detayını görmek için kayıt seçin", [rec.id for rec in history])
        if sel:
            rec = next(r for r in history if r.id == sel)
            st.markdown(f"**Tarih:** {rec.date}")
            st.markdown(f"**Dosya:** {rec.original_filename}")
            st.markdown(f"**Risk Seviyesi:** {rec.risk_level}")
            st.markdown(f"**Güven Skoru:** %{int(float(rec.confidence_score)*100)}")
            st.markdown(f"**Uyumlu mu:** {rec.compliance_status}")
            st.markdown("**Rapor JSON:**")
            data = json.loads(rec.json_report)
            st.json(data)
            if rec.pdf_path and os.path.exists(rec.pdf_path):
                with open(rec.pdf_path, "rb") as f:
                    st.download_button("PDF Raporu İndir", f, file_name=os.path.basename(rec.pdf_path), mime="application/pdf")
            else:
                # regenerate PDF from stored JSON
                if st.button("PDF'yi Yeniden Oluştur"):
                    temp_pdf = os.path.join(tempfile.gettempdir(), f"audit_regen_{rec.id}.pdf")
                    # build a fake state
                    fake_state = AuditState(
                        original_text=data.get('original_text',''),
                        scrubbed_text=data.get('scrubbed_text',''),
                        audit_report=data,
                        confidence_score=float(rec.confidence_score),
                        iteration=0,
                        error_message=""
                    )
                    generate_pdf_report(fake_state, temp_pdf)
                    with open(temp_pdf, "rb") as f2:
                        st.download_button("Yeni PDF İndir", f2, file_name=os.path.basename(temp_pdf), mime="application/pdf")
    else:
        st.info("Henüz herhangi bir denetim kaydı yok.")

with col_input:
    st.subheader("📄 Denetlenecek Doküman")
    
    input_method = st.radio("Veri Giriş Yöntemi", ["Metin Gir", "PDF Yükle", "Resim Yükle"], horizontal=True)
    
    text_to_audit = ""
    redacted_pdf_path = None

    if input_method == "Metin Gir":
        text_to_audit = st.text_area(
            "Denetlenecek metni buraya yapıştırın",
            height=300,
            placeholder="Müşteri Ahmet Yılmaz'ın sipariş teyidi alındı. İletişim: 05551234567..."
        )

    elif input_method == "PDF Yükle":
        uploaded_doc = st.file_uploader("Denetlenecek PDF Dosyası", type=["pdf"])
        if uploaded_doc:
            with st.spinner("PDF okunuyor..."):
                temp_pdf_path = os.path.join(tempfile.gettempdir(), os.path.basename(uploaded_doc.name))
                save_uploaded_file(uploaded_doc, temp_pdf_path)

                text_to_audit = extract_text_from_pdf(temp_pdf_path)
                st.success("PDF başarıyla okundu!")
                with st.expander("Okunan Orijinal Metin"):
                    st.write(text_to_audit)

                # PDF üzerinde hassas verileri kırparak redakte et
                redacted_pdf_path = os.path.join(
                    tempfile.gettempdir(),
                    f"redacted_{os.path.basename(uploaded_doc.name)}"
                )
                patterns = [
                    r"\b[1-9][0-9]{10}\b",  # TCKN
                    r"TR\d{2}(?:\s?\d{4}){5}\s?\d{2}",  # IBAN
                    r"(?:\+90|0)?5\d{2}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}",  # Telefon
                    r"\b05\d{9}\b"
                ]
                try:
                    redact_pdf(temp_pdf_path, redacted_pdf_path, patterns)
                    st.info("PDF başarıyla redakte edildi. (Siyah bantlı versiyon hazır)")
                except Exception as e:
                    st.warning(f"PDF redaksiyonu sırasında hata oluştu: {e}")

    else:  # Resim Yükle
        uploaded_img = st.file_uploader("Denetlenecek Resim Dosyası", type=["png", "jpg", "jpeg"])
        if uploaded_img:
            with st.spinner("OCR yapılıyor..."):
                text_to_audit = extract_text_from_image(uploaded_img)
                if text_to_audit.strip():
                    st.success("OCR başarıyla tamamlandı!")
                else:
                    st.warning("OCR'dan metin alınamadı. Lütfen daha net bir resim deneyin.")
                with st.expander("OCR ile çıkarılan metin"):
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
                    error_message="",
                    source_docs=[],
                    redacted_pdf_path=redacted_pdf_path
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
                        violations = report_data.get("violations_with_reference") or report_data.get("violations", [])
                        source_docs = report_data.get("source_documents") or result.get("source_docs", [])
                        redacted_pdf_path = result.get("redacted_pdf_path")
                        
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
                                if isinstance(vio, dict):
                                    text = vio.get("text")
                                    ref = vio.get("reference")
                                    if ref:
                                        st.error(f"- {text}  \n  (Referans: {ref})")
                                    else:
                                        st.error(f"- {text}")
                                else:
                                    st.error(f"- {vio}")
                        elif is_compliant:
                            st.info("Herhangi bir ihlal bulunamadı.")
                            
                        # Maskelenmiş metin
                        with st.expander("Görüntüle: Maskelenmiş (Scrubbed) Metin", expanded=False):
                            st.markdown(f"*{scrubbed_text}*")
                        
                        # Bu karara nasıl varıldı? (Kaynak Dokümanlar)
                        if source_docs:
                            st.markdown("#### Bu karara nasıl varıldı?")
                            for src in source_docs:
                                src_name = src.get("source") or "Bilinmeyen kaynak"
                                page = src.get("page")
                                if page:
                                    st.markdown(f"- {src_name} (Sayfa {page})")
                                else:
                                    st.markdown(f"- {src_name}")

                        # Redakte edilmiş Orijinal PDF (Siyah Bantlı)
                        if redacted_pdf_path and os.path.exists(redacted_pdf_path):
                            with open(redacted_pdf_path, "rb") as f:
                                st.download_button(
                                    "Redakte Edilmiş PDF İndir",
                                    f,
                                    file_name=os.path.basename(redacted_pdf_path),
                                    mime="application/pdf"
                                )

                        # PDF raporu oluştur ve indirme tuşu
                        pdf_name = f"audit_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf"
                        pdf_path = os.path.join(tempfile.gettempdir(), pdf_name)
                        # create report file
                        generate_pdf_report(result, pdf_path)
                        # ensure file exists before offering download
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as f:
                                st.download_button("Denetim Raporunu PDF Olarak İndir", f, file_name=pdf_name, mime="application/pdf")
                        else:
                            st.error("PDF raporu oluşturulamadı.")

                        # Veritabanına kaydet
                        # Tarihçe kaydı için orijinal dosya adını belirle
                        original_filename = None
                        if 'uploaded_doc' in locals() and uploaded_doc:
                            original_filename = uploaded_doc.name
                        elif 'uploaded_img' in locals() and uploaded_img:
                            original_filename = uploaded_img.name

                        save_audit_history(
                            original_filename=original_filename,
                            risk_level=risk_level,
                            compliance_status=is_compliant,
                            confidence_score=confidence,
                            json_report=report_data,
                            pdf_path=redacted_pdf_path or pdf_path
                        )
                        
                except Exception as e:
                    st.error(f"Uygulama Hatası: {str(e)}")
    else:
        st.info("Sonuçları görmek için sol taraftan veri girin ve işlemi başlatın.")
