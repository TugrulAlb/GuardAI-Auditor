import json
import re
import os
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import logging

from dotenv import load_dotenv
load_dotenv()

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

# Web arama için
from langchain_community.tools import DuckDuckGoSearchRun

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
    redacted_file_path: Optional[str]  # Redakte edilmiş dosya yolu (PDF veya resim)
    web_search_results: List[Dict[str, Any]]  # Web arama sonuçları

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
from PIL import Image, ImageDraw  # For image redaction


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


def redact_image(input_image_path: str, output_image_path: str, text: str, patterns: List[str], ocr_data: List[Dict[str, Any]]) -> str:
    """Redact sensitive data in an image by drawing black rectangles over OCR-detected text using bounding boxes."""
    try:
        # Open image
        img = Image.open(input_image_path)
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size

        # Convert OCR data to text for pattern matching
        ocr_text = "".join([item['char'] for item in ocr_data])

        # Find sensitive data positions using regex on OCR text
        redaction_rects = []

        for pattern in patterns:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                matches = list(regex.finditer(ocr_text))

                for match in matches:
                    # Find corresponding bounding boxes for this match
                    start_idx = match.start()
                    end_idx = match.end()

                    # Collect bounding boxes for characters in this match
                    match_boxes = []
                    current_idx = 0

                    for item in ocr_data:
                        char_len = len(item['char'])
                        if current_idx >= start_idx and current_idx < end_idx:
                            match_boxes.append(item['bbox'])
                        current_idx += char_len
                        if current_idx >= end_idx:
                            break

                    if match_boxes:
                        # Create a bounding rectangle that covers all character boxes
                        x1 = min(box[0] for box in match_boxes)
                        y1 = min(box[1] for box in match_boxes)
                        x2 = max(box[2] for box in match_boxes)
                        y2 = max(box[3] for box in match_boxes)

                        # Convert to PIL coordinates (Tesseract uses bottom-left origin, PIL uses top-left)
                        pil_y1 = img_height - y2
                        pil_y2 = img_height - y1

                        redaction_rects.append((x1, pil_y1, x2, pil_y2))

            except re.error:
                # Skip invalid regex patterns
                continue

        # Apply redactions
        for rect in redaction_rects:
            draw.rectangle(rect, fill=(0, 0, 0))

        # Save the redacted image
        img.save(output_image_path)
        return output_image_path
    except Exception as e:
        logger.error(f"Image redaction failed: {e}")
        # Return original if redaction fails
        return input_image_path


def perform_web_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Perform web search for additional context when local RAG is insufficient."""
    try:
        search_tool = DuckDuckGoSearchRun()
        results = search_tool.run(query)

        # Parse results (DuckDuckGo returns string, we need to structure it)
        search_results = []
        if results:
            # Split by common separators and create structured results
            lines = results.split('\n')
            current_result = {}
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                if line.startswith('Title:'):
                    if current_result and current_result.get('title'):  # Only add if has title
                        search_results.append(current_result)
                    current_result = {'title': line.replace('Title:', '').strip(), 'link': '', 'snippet': ''}
                elif line.startswith('Link:'):
                    current_result['link'] = line.replace('Link:', '').strip()
                elif line.startswith('Snippet:') or line.startswith('Description:'):
                    current_result['snippet'] = line.replace('Snippet:', '').replace('Description:', '').strip()

            if current_result and current_result.get('title'):  # Add last result
                search_results.append(current_result)

            # Filter out incomplete results
            search_results = [r for r in search_results if r.get('title') and r.get('snippet')]

        return search_results[:max_results]
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []


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

    # Web Arama Sonuçları
    web_search_results = report.get("web_search_results", [])
    if web_search_results:
        content.append(Paragraph("Web Arama Sonuçları (Güncel Mevzuat):", heading_style))
        for result in web_search_results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            if title:
                content.append(Paragraph(f"• {title}", body_style))
                if snippet:
                    content.append(Paragraph(f"  {snippet[:200]}...", body_style))
        content.append(Spacer(1, 0.2*inch))

    # Redakte Edilmiş Dosya
    redacted_file_path = state.get("redacted_file_path")
    if redacted_file_path:
        content.append(Paragraph("Redakte Edilmiş Dosya (siyah bantlı):", heading_style))
        content.append(Paragraph(str(redacted_file_path), body_style))
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
    state.setdefault("redacted_file_path", None)
    state.setdefault("web_search_results", [])

    # Kapsamlı maskeleme (Scrubber) - Hassas verileri gizle
    # TCKN (11 haneli)
    text = re.sub(r'\b\d{11}\b', 'XXXXXXXXXXX', text)
    # IBAN
    text = re.sub(r'TR\d{2}(?:\s?\d{4}){5}\s?\d{2}', 'TRXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', text)
    # Telefon
    text = re.sub(r'(?:\+?90|0)?\s*[1-9]\d{2}\s*\d{3}\s*\d{2}\s*\d{2}', '0XXXXXXXXXX', text)
    # E-posta
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'XXXX@XXXX.XXX', text)
    # Adres bilgileri (basit pattern)
    text = re.sub(r'\b\d{5}\s+[A-ZÇĞİÖŞÜ\s]+/[A-ZÇĞİÖŞÜ\s]+\b', 'XXXXX XXXXXXXX/XXXXXXXX', text)

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

        # Web arama: Eğer rapor belirsiz kurallar içeriyorsa veya güncel mevzuat gerekiyorsa
        web_search_results = []
        if report_data.get("confidence_score", 0.0) < 0.8 or "güncel" in text_to_audit.lower() or "2026" in text_to_audit.lower() or "mevzuat" in text_to_audit.lower():
            logger.info("--- Web arama yapılıyor: Güncel mevzuat bilgileri aranıyor ---")
            search_queries = []

            # Metindeki potansiyel arama terimleri - daha kapsamlı
            text_lower = text_to_audit.lower()
            if "ceza" in text_lower or "cezalandırma" in text_lower:
                search_queries.append("2026 KVKK cezaları güncel miktarları")
                search_queries.append("KVKK veri ihlali cezaları 2026")
            if "ihlal" in text_lower or "ihlaller" in text_lower:
                search_queries.append("KVKK veri ihlali cezaları ve yaptırımlar")
                search_queries.append("2026 KVKK ihlal türleri")
            if "güncel" in text_lower or "mevzuat" in text_lower:
                search_queries.append("2026 KVKK mevzuatı güncel değişiklikler")
                search_queries.append("KVKK 6698 sayılı kanun güncel hali")
            if "veri" in text_lower and "işleme" in text_lower:
                search_queries.append("KVKK veri işleme şartları 2026")
            if "açık rıza" in text_lower or "rıza" in text_lower:
                search_queries.append("KVKK açık rıza şartları 2026")
            if "veri güvenliği" in text_lower or "güvenlik" in text_lower:
                search_queries.append("KVKK veri güvenliği tedbirleri")

            # Varsayılan arama - daha spesifik
            if not search_queries:
                search_queries.append("2026 KVKK mevzuatı güncel değişiklikler ve cezalar")

            for query in search_queries[:2]:  # Maksimum 2 arama
                results = perform_web_search(query, max_results=2)
                web_search_results.extend(results)

        # Web arama sonuçlarını rapora ekle
        if web_search_results:
            report_data["web_search_results"] = web_search_results
            # Web sonuçlarını policy_context'e ekle
            web_context = "\n\nWEB ARAMA SONUÇLARI (Güncel Mevzuat):\n"
            for result in web_search_results:
                web_context += f"- {result.get('title', '')}: {result.get('snippet', '')}\n"
            policy_context += web_context

            # Güven skoru web arama ile arttıysa güncelle
            if report_data.get("confidence_score", 0.0) < 0.9:
                report_data["confidence_score"] = min(0.9, report_data.get("confidence_score", 0.0) + 0.1)

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
            elif web_search_results:
                # Web arama sonucu varsa referans olarak kullan
                web_ref = web_search_results[idx % len(web_search_results)]
                ref = f"Web: {web_ref.get('title', 'Güncel Mevzuat')}"
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
        state["web_search_results"] = web_search_results
    except Exception as e:
        logger.error(f"Audit Node'da bir hata oluştu: {str(e)}")
        state["error_message"] = str(e)
        state["confidence_score"] = 0.0
    finally:
        # Kaynak doküman bilgilerini state'e ekle (boş kalsa da)
        state["source_docs"] = source_docs
        state["web_search_results"] = web_search_results

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
