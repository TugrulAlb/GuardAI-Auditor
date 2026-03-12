import json
import re
import os
from typing import TypedDict, Annotated, List, Dict, Any
import logging

from pydantic import BaseModel, Field

# LangChain yığınları
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
from datetime import datetime

# PDF raporlaması için ReportLab
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

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

# create tables
Base.metadata.create_all(bind=engine)

# Utility functions

def generate_pdf_report(state: AuditState, filename: str) -> str:
    """Creates a PDF report from the audit state and returns the file path."""
    # register a standard UTF-8 supporting font
    try:
        pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
        font_name = 'DejaVu'
    except Exception:
        font_name = 'Helvetica'

    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    c.setFont(font_name, 16)
    c.drawCentredString(width/2, height-50, "KURUMSAL GÜVENLİK DENETİM RAPORU")
    c.setFont(font_name, 10)
    c.drawString(50, height-80, f"Tarih: {datetime.utcnow().isoformat()}")
    report = state.get("audit_report", {})
    score = report.get("confidence_score", 0.0)
    c.drawString(50, height-95, f"Güven Skoru: %{int(score*100)}")

    # risk level
    c.setFont(font_name, 12)
    c.drawString(50, height-120, f"Risk Seviyesi: {report.get('risk_level', '')}")

    # violations list
    c.setFont(font_name, 10)
    y = height-140
    c.drawString(50, y, "İhlaller:")
    for vio in report.get("violations", []):
        y -= 15
        c.drawString(60, y, f"- {vio}")
        if y < 50:
            c.showPage()
            y = height-50
            c.setFont(font_name, 10)
    # scrubbed text
    y -= 30
    c.drawString(50, y, "Maskelenmiş Metin:")
    y -= 15
    text = state.get("scrubbed_text", "")
    for line in text.split("\n"):
        if y < 50:
            c.showPage()
            y = height-50
            c.setFont(font_name, 10)
        c.drawString(60, y, line)
        y -= 12
    c.save()
    return filename


def save_audit_history(original_filename: str, risk_level: str, compliance_status: bool, json_report: Dict[str, Any], pdf_path: str=None) -> AuditHistory:
    """Persist audit results to SQLite and return the record."""
    session = SessionLocal()
    record = AuditHistory(
        original_filename=original_filename,
        risk_level=risk_level,
        compliance_status=compliance_status,
        json_report=json.dumps(json_report, ensure_ascii=False),
        pdf_path=pdf_path
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    session.close()
    return record


def get_history() -> list:
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
    
    # Regex Kuralları
    # 1. TC Kimlik No (11 hane)
    text = re.sub(r'\b[1-9]{1}[0-9]{10}\b', 'XXXXX', text)
    
    # 2. IBAN (TR ile başlayan 24 hane/karakter boşluksuz veya boşluklu)
    text = re.sub(r'\bTR\d{2}\s?(?:\d{4}\s?){4}\d{2}\b', 'XXXXX', text)
    text = re.sub(r'\bTR\d{24}\b', 'XXXXX', text)
    
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
    try:
        vectorstore = setup_chroma()
        retrieved_docs = vectorstore.similarity_search(text_to_audit, k=2)
        policy_context = "\\n".join([doc.page_content for doc in retrieved_docs])
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
        state["audit_report"] = report_data
        state["confidence_score"] = float(report_data.get("confidence_score", 0.0))
        state["iteration"] = iteration + 1
    except Exception as e:
        logger.error(f"Audit Node'da bir hata oluştu: {str(e)}")
        state["error_message"] = str(e)
        state["confidence_score"] = 0.0
        
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
