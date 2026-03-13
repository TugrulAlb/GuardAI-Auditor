import json
import re
import os
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import logging

from pydantic import BaseModel, Field

# LangChain yığınları
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
from datetime import datetime, timezone

# PDF raporlaması için ReportLab - Paragraph & SimpleDocTemplate ile UTF-8 desteği
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
from reportlab.lib import colors

# SQLAlchemy imports
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv()

# ==========================================
# 1. State Tanımları (LangGraph State Machine)
# ==========================================
class AuditState(TypedDict):
    original_text: str          # Yüklenen orijinal doküman
    scrubbed_text: str          # Maskelenmiş metin
    audit_report: Dict[str, Any]# LLM'in ürettiği analiz raporu (JSON)
    confidence_score: float     # LLM'in analizdeki güven skoru
    iteration: int              # Yeniden deneme iterasyonu (Loop önlemi)
    error_message: str          # Hata mesajı varsa
    source_docs: List[Dict[str, Any]]  # Chroma'dan gelen kaynak belge bilgileri
    redacted_pdf_path: Optional[str]  # Redakte edilmiş (siyah bantlı) PDF yolu

# Structured JSON çıkışını garantilemek için Pydantic
class RiskReport(BaseModel):
    is_compliant: bool = Field(description="KVKK'ya veya politikalara uyumlu mu?")
    violations: List[str] = Field(description="Bulunan ihlallerin listesi.")
    risk_level: str = Field(description="Risk seviyesi: Düşük, Orta, Yüksek, Kritik.")
    confidence_score: float = Field(description="Analiz güven skoru (0.0 - 1.0 arası).")

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///audit_history.db', echo=False)
SessionLocal = sessionmaker(bind=engine)

class AuditHistory(Base):
    __tablename__ = 'audit_history'
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.utcnow)
    original_filename = Column(String, nullable=True)
    risk_level = Column(String, nullable=False)
    compliance_status = Column(Boolean, nullable=False)
    json_report = Column(Text, nullable=False)
    pdf_path = Column(String, nullable=True)
    confidence_score = Column(String, nullable=False)

# create tables / migration helper
def ensure_db_schema():
    # if table exists but missing confidence_score column, drop & recreate
    inspector = None
    try:
        from sqlalchemy import inspect
        inspector = inspect(engine)
    except ImportError:
        pass
    if inspector and inspector.has_table('audit_history'):
        cols = [c['name'] for c in inspector.get_columns('audit_history')]
        if 'confidence_score' not in cols:
            # recreate
            engine.dispose()
            if os.path.exists('audit_history.db'):
                os.remove('audit_history.db')
            Base.metadata.create_all(bind=engine)
        else:
            # nothing to do
            pass
    else:
        Base.metadata.create_all(bind=engine)

ensure_db_schema()

# Utility functions

import fitz  # PyMuPDF (PDF redaction)


def redact_pdf(input_pdf_path: str, output_pdf_path: str, patterns: List[str]) -> str:
    """Redact sensitive data in a PDF by drawing black rectangles over matches.

    Patterns should be regular expressions used to locate sensitive text.
    """
    doc = fitz.open(input_pdf_path)
    for page in doc:
        for pattern in patterns:
            try:
                matches = page.search_for(pattern, regex=True)
            except Exception:
                # fallback: non-regex search
                matches = page.search_for(pattern)
            for rect in matches:
                # Add a redaction annotation filled with black
                page.add_redact_annot(rect, fill=(0, 0, 0))
        # Apply all redactions on this page (safely callable even if none)
        try:
            page.apply_redactions()
        except Exception:
            pass
    doc.save(output_pdf_path, garbage=4, deflate=True)
    doc.close()
    return output_pdf_path


def generate_pdf_report(state: AuditState, filename: str) -> str:
    """Creates a PDF report from the audit state and returns the file path."""
    
    # Adım 1: Font Kaydı - Mac üzerindeki Arial.ttf dosyasını kaydet
    font_paths = [
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf"
    ]
    
    base_font = 'Helvetica'  # Fallback
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont('Arial-Turkish', font_path))
                base_font = 'Arial-Turkish'
                logger.info(f"Font kaydı başarılı: {font_path}")
                break
            except Exception as e:
                logger.warning(f"Font kaydı hatası ({font_path}): {e}")
    
    # Adım 2: Stil Ataması - getSampleStyleSheet ile stilleri al ve fontName'ı set et
    styles = getSampleStyleSheet()
    
    # Temel Normal stil'in fontName'ini set et (parent olarak kullanılacak)
    styles['Normal'].fontName = base_font
    
    # Özel stiller oluştur (tümü Arial-Turkish veya fallback font'u kullanacak)
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Normal'],
        fontName=base_font,
        fontSize=16,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=12,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Normal'],
        fontName=base_font,
        fontSize=12,
        textColor=colors.HexColor('#333333'),
        spaceAfter=8,
        spaceBefore=6
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName=base_font,
        fontSize=10,
        leading=14
    )
    
    # Adım 3: PDF Belgesi oluştur (SimpleDocTemplate kullanarak)
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    # İçerik listesi
    content = []
    
    # Title - Adım 3a: Paragraph objesi içinde başlık
    title_text = "KURUMSAL GÜVENLİK DENETİM RAPORU"
    content.append(Paragraph(title_text, title_style))
    content.append(Spacer(1, 0.3*inch))
    
    # Tarih ve Güven Skoru
    report = state.get("audit_report", {})
    score = report.get("confidence_score", 0.0)
    timestamp = datetime.now(timezone.utc).isoformat()
    
    meta_text = f"<b>Tarih:</b> {timestamp}<br/><b>Güven Skoru:</b> %{int(score*100)}"
    content.append(Paragraph(meta_text, body_style))
    content.append(Spacer(1, 0.2*inch))
    
    # Risk Seviyesi - Adım 3b: Paragraph objesi içinde
    risk_level = report.get('risk_level', 'Bilinmiyor')
    risk_text = f"<b>Risk Seviyesi:</b> {risk_level}"
    content.append(Paragraph(risk_text, heading_style))
    content.append(Spacer(1, 0.15*inch))

    # Kaynak Dokümanlar - Referans
    source_docs = report.get("source_documents", [])
    if source_docs:
        content.append(Paragraph("Kaynak Dokümanlar:", heading_style))
        for src in source_docs:
            src_name = src.get("source") or "Bilinmeyen kaynak"
            page = src.get("page")
            if page:
                content.append(Paragraph(f"• {src_name} (Sayfa {page})", body_style))
            else:
                content.append(Paragraph(f"• {src_name}", body_style))
        content.append(Spacer(1, 0.2*inch))

    # Redakte Edilmiş PDF (Siyah Bantlı)
    redacted_pdf_path = state.get("redacted_pdf_path")
    if redacted_pdf_path:
        content.append(Paragraph("Redakte Edilmiş PDF (siyah bantlı):", heading_style))
        content.append(Paragraph(str(redacted_pdf_path), body_style))
        content.append(Spacer(1, 0.2*inch))

    # İhlaller Bölümü - Adım 3c: Paragraph objeleri ile ihlal listesi
    violations = report.get("violations", [])
    if violations:
        content.append(Paragraph("İhlaller:", heading_style))
        for vio in violations:
            # UTF-8 desteği ile Paragraph
            violation_text = f"• {vio}"
            content.append(Paragraph(violation_text, body_style))
        content.append(Spacer(1, 0.2*inch))
    
    # Maskelenmiş Metin Bölümü - Adım 3d: Paragraph objesi içinde
    content.append(Paragraph("Maskelenmiş Metin:", heading_style))
    scrubbed_text = state.get("scrubbed_text", "")
    
    # Metni satır satır Paragraph objelerine dönüştür
    for line in scrubbed_text.split("\n"):
        if line.strip():  # Boş satırları atlama
            # UTF-8 karakterleri düzgün göster
            content.append(Paragraph(line.replace("<", "&lt;").replace(">", "&gt;"), body_style))
    
    # PDF'i oluştur
    try:
        doc.build(content)
        logger.info(f"PDF raporu başarıyla oluşturuldu: {filename}")
    except Exception as e:
        logger.error(f"PDF oluşturma hatası: {e}")
        raise
    
    return filename


def save_audit_history(original_filename: str, risk_level: str, compliance_status: bool, confidence_score: float, json_report: Dict[str, Any], pdf_path: str=None) -> AuditHistory:
    """Persist audit results to SQLite and return the record."""
    session = SessionLocal()
    record = AuditHistory(
        original_filename=original_filename,
        risk_level=risk_level,
        compliance_status=compliance_status,
        confidence_score=str(confidence_score),
        json_report=json.dumps(json_report, ensure_ascii=False),
        pdf_path=pdf_path
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    session.close()
    return record


def get_history() -> list:
    ensure_db_schema()
    session = SessionLocal()
    records = session.query(AuditHistory).order_by(AuditHistory.date.desc()).all()
    session.close()
    return records


# LangGraph yığınları
from langgraph.graph import StateGraph, END

# Ortam değişkenleri ve log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Örnek OPENAI API Key (Gerçek projede .env dosyasından alınmalıdır)
# os.environ["OPENAI_API_KEY"] = "sk-..."



# ==========================================
# 2. Vector DB (ChromaDB) Kurulumu & Veri Yükleme
# ==========================================
# Gerçek projede HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") veya OpenAI kullanılabilir.
def setup_chroma():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
    )
    
    db_directory = "./chroma_db"
    
    vectorstore = Chroma(
        collection_name="kvkk_policies", 
        embedding_function=embeddings,
        persist_directory=db_directory
    )
    
    # Sadece veritabanı daha önce oluşturulmamışsa veya boşsa yükle
    if not os.path.exists(db_directory) or not os.listdir(db_directory):
        pdf_files = glob.glob("*.pdf")
        all_docs = []
        
        for pdf_file in pdf_files:
            logger.info(f"PDF yükleniyor: {pdf_file}")
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            all_docs.extend(docs)
            
        if all_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)
            logger.info(f"Toplam {len(splits)} doküman parçası ChromaDB'ye ekleniyor...")
            vectorstore.add_documents(splits)
            logger.info("ChromaDB başarıyla oluşturuldu ve veriler eklendi.")
        else:
            logger.warning("Klasörde hiç PDF dosyası bulunamadı.")
            
    return vectorstore

# ==========================================
# 3. LangGraph Düğümleri (Nodes)
# ==========================================

def scrubbing_node(state: AuditState) -> AuditState:
    """
    Local Guardrails (Regex Scrubber): 
    Doküman LLM'e gitmeden önce yerelde (local) çalışır ve maskeleme (redaction) yapar.
    """
    logger.info("--- NODE: ScrubbingNode Çalışıyor ---")
    text = state.get("original_text", "")

    # Varsayılan değerler
    state.setdefault("source_docs", [])
    state.setdefault("redacted_pdf_path", None)
    
    # Regex Kuralları
        # 1. TC Kimlik No (11 hane)
    text = re.sub(r'\b[1-9]{1}[0-9]{10}\b', 'XXXXX', text)
    
    # 2. IBAN (TR ile başlayan 24 hane, aralık/boşluklu veya bitişik)
    text = re.sub(r'TR\d{2}(\s?\d{4}){5}\s?\d{2}', 'XXXXX', text)
    
    # 3. Telefon Numaraları (05XX veya +905XX formatı)
    text = re.sub(r'(?:\+90|0)?5\d{2}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}', 'XXXXX', text)
    text = re.sub(r'\b05\d{9}\b', 'XXXXX', text)
    
    state["scrubbed_text"] = text
    return state

def audit_node(state: AuditState) -> AuditState:
    """
    AuditNode:
    Maskelenmiş metni alır, ChromaDB'den ilgili kuralları çeker ve analiz eder.
    """
    logger.info(f"--- NODE: AuditNode Çalışıyor (Iteration: {state.get('iteration', 0)}) ---")
    
    text_to_audit = state.get("scrubbed_text", "")
    iteration = state.get("iteration", 0)
    
    # 1. RAG ile Kuralları Çek
    source_docs: List[Dict[str, Any]] = []
    try:
        vectorstore = setup_chroma()
        retrieved_docs = vectorstore.similarity_search(text_to_audit, k=2)
        policy_context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Kaynak dokümanları (metadata) sakla
        for doc in retrieved_docs:
            metadata = getattr(doc, "metadata", {}) or {}
            source_docs.append({
                "source": metadata.get("source") or metadata.get("sourcefile") or metadata.get("file_name") or "",
                "page": metadata.get("page") or metadata.get("page_number") or metadata.get("page_num") or None,
            })
    except Exception as e:
        logger.warning(f"ChromaDB başlatılamadı veya boş, örnek politikalar kullanılmadan devam ediliyor... ({e})")
        policy_context = "Genel KVKK veri koruma ilkeleri zorunludur."

    # 2. LLM Modelini Çağır (Deteministic, JSON Forced)
    # temperature=0.0 ile halüsinasyon(hallucination) önlenir.
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_LLM_DEPLOYMENT_NAME", "gpt-4o"), 
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    # Daha fazla iterasyonda Prompt detaylandırılır
    prompt_suffix = ""
    if iteration > 0:
        prompt_suffix = "LÜTFEN ÖNCEKİ ANALİZ EKSİKTİ VEYA GÜVENSİZDİ. Kuralları daha derin analiz edip açıkla."
    
    prompt = f"""
Aşağıdaki doküman üzerinde bir "AI Compliance & Security Auditor" gibi davranarak KVKK ve Kurumsal Güvenlik politikaları denetimi yap ve çıktıyı HER ZAMAN JSON olarak döndür.
Çıktı JSON Şeması:
{{
  "is_compliant": bool,
  "violations": ["ihlal detay 1", "ihlal detay 2"],
  "risk_level": "Düşük/Orta/Yüksek/Kritik",
  "confidence_score": float (0-1)
}}

{prompt_suffix}

KURUMSAL BELGELER (RAG Kapsamı):
{policy_context}

DENETLENECEK DOKÜMAN:
{text_to_audit}
"""
    
    try:
        response = llm.invoke(prompt)
        report_data = json.loads(response.content)

        # Kaynak dokümanları rapora ekle
        report_data["source_documents"] = source_docs

        # Violations listesi varsa, her birine kaynak bilgisi ekle
        violations = report_data.get("violations", []) or []
        violations_with_ref = []
        for idx, vio in enumerate(violations):
            ref = None
            if source_docs:
                src = source_docs[idx % len(source_docs)]
                src_str = src.get("source", "")
                page = src.get("page")
                if page is not None and page != "":
                    ref = f"{src_str} (sayfa {page})"
                elif src_str:
                    ref = f"{src_str}"
            violations_with_ref.append({
                "text": vio,
                "reference": ref
            })
        report_data["violations_with_reference"] = violations_with_ref

        # State'i güncelle
        state["audit_report"] = report_data
        state["confidence_score"] = float(report_data.get("confidence_score", 0.0))
        state["iteration"] = iteration + 1
        state["source_docs"] = source_docs
    except Exception as e:
        logger.error(f"Audit Node'da bir hata oluştu: {str(e)}")
        state["error_message"] = str(e)
        state["confidence_score"] = 0.0
    finally:
        # Kaynak doküman bilgilerini state'e ekle (boş kalsa da)
        state["source_docs"] = source_docs

    return state

# ==========================================
# 4. Yönlendirme ve Döngü Mantığı
# ==========================================

def should_verify(state: AuditState) -> str:
    """
    VerificationNode Mantığı (Conditional Edge):
    confidence_score düşükse ve maksimum denemeye (örn 3) ulaşılmadıysa Audit'e geri döner (Cycle).
    """
    confidence = state.get("confidence_score", 0.0)
    iteration = state.get("iteration", 0)
    
    logger.info(f"--- EDGE: Sonuç Kontrolü (Confidence: {confidence}, Iteration: {iteration}) ---")
    
    if confidence < 0.8 and iteration < 3:
        logger.info("--- EDGE: Güven skoru düşük, re-evaluate için AuditNode'a dönülüyor... ---")
        return "audit_node"
    
    logger.info("--- EDGE: Denetim tamamlandı -> END ---")
    return "end"

# ==========================================
# 5. LangGraph'ı İnşa Etme (Compile)
# ==========================================

workflow = StateGraph(AuditState)

# Düğümleri (Nodes) ekle
workflow.add_node("scrubbing_node", scrubbing_node)
workflow.add_node("audit_node", audit_node)

# Kenarları (Edges) belirle
workflow.set_entry_point("scrubbing_node")
workflow.add_edge("scrubbing_node", "audit_node")

# Conditional Edge ile Verify (Döngü)
workflow.add_conditional_edges(
    "audit_node",
    should_verify,
    {
        "audit_node": "audit_node",
        "end": END
    }
)

# Sistemi derle
app = workflow.compile()

# ==========================================
# TEST EDİLMESİ
# ==========================================
if __name__ == "__main__":
    import sys
    
    sample_text = (
        "Müşteri Ahmet Yılmaz'ın sipariş teyidi alındı. İletişim numarası: 05551234567. "
        "Fatura adresi için TC 11223344556 numarasına kayıtlı. "
        "Hesabından ücret tahsilatı için IBAN TR120006200000012345678912 kullanıldı. "
        "Ayrıca, verileri açık rızası alınmadan işlenmeye devam ediyor."
    )
    
    initial_state = AuditState(
        original_text=sample_text,
        scrubbed_text="",
        audit_report={},
        confidence_score=0.0,
        iteration=0,
        error_message=""
    )
    
    logger.info("LangGraph Agent çalıştırılıyor...")
    result = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print("ORİJİNAL DOKÜMAN:\n", result.get("original_text"))
    print("\n" + "="*50)
    print("MASKELEME SONRASI DOKÜMAN (Scrubber:\n", result.get("scrubbed_text"))
    print("\n" + "="*50)
    print("DENETİM RAPORU (JSON):\n", json.dumps(result.get("audit_report"), indent=2, ensure_ascii=False))
    print("="*50 + "\n")
