import os
import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
import pdfplumber
import camelot
import google.generativeai as genai
from datetime import datetime, timedelta
import hashlib
import json
from typing import Optional, List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ CONFIG ------------------
NOTICES_DIR = "notices"
CACHE_DIR = "cache"
CACHE_EXPIRY_HOURS = 24
MAX_NOTICES = 10
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# Initialize directories
os.makedirs(NOTICES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure Gemini
API_KEY = "AIzaSyBgQNzh5JanRXNrtSNgxQDNYh8bjyJr1sI"  # must be set before running
if not API_KEY:
    st.error("⚠️ Please set GOOGLE_API_KEY environment variable")
    st.stop()

genai.configure(api_key=API_KEY)

# ------------------ UTILITY FUNCTIONS ------------------
def get_cache_path(key: str) -> str:
    """Generate cache file path for a given key."""
    hashed = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed}.json")

def load_from_cache(key: str, expiry_hours: int = CACHE_EXPIRY_HOURS) -> Optional[any]:
    """Load data from cache if not expired."""
    cache_path = get_cache_path(key)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time < timedelta(hours=expiry_hours):
                return cache_data['data']
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
    return None

def save_to_cache(key: str, data: any):
    """Save data to cache with timestamp."""
    cache_path = get_cache_path(key)
    try:
        with open(cache_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'data': data
            }, f)
    except Exception as e:
        logger.warning(f"Cache write error: {e}")

# ------------------ NOTICE FETCHING ------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_notice_links() -> List[Dict[str, str]]:
    """Fetch notice links with improved error handling and metadata."""
    cache_key = "notice_links"
    cached = load_from_cache(cache_key)
    if cached:
        return cached
    
    try:
        response = requests.get(
            "https://www.imsnsit.org/imsnsit/notifications.php",
            timeout=TIMEOUT_SECONDS,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        notices = []
        for i, a in enumerate(soup.find_all("a", href=True), 1):
            href = a["href"]
            if "docs.google.com" in href or "drive.google.com" in href:
                # Extract notice title from link text or use default
                title = a.get_text(strip=True) or f"Notice {len(notices) + 1}"
                
                # Try to extract date if present in the text
                date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
                date_match = re.search(date_pattern, title)
                date_str = date_match.group() if date_match else "Date not available"
                
                notices.append({
                    'url': href,
                    'title': title[:100],  # Limit title length
                    'date': date_str,
                    'index': len(notices) + 1
                })
                
                if len(notices) >= MAX_NOTICES:
                    break
        
        save_to_cache(cache_key, notices)
        return notices
    
    except requests.RequestException as e:
        st.error(f"Failed to fetch notices: {e}")
        return []

# ------------------ PDF PROCESSING ------------------
def extract_file_id(url: str) -> Optional[str]:
    """Extract Google Drive file ID from URL."""
    patterns = [
        r"/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"/file/d/([a-zA-Z0-9_-]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def download_google_doc_as_pdf(file_id: str, output_path: str) -> bool:
    """Download Google Doc as PDF with retry logic."""
    urls = [
        f"https://docs.google.com/document/d/{file_id}/export?format=pdf",
        f"https://drive.google.com/uc?export=download&id={file_id}"
    ]
    
    for attempt in range(MAX_RETRIES):
        for url in urls:
            try:
                resp = requests.get(url, timeout=TIMEOUT_SECONDS, stream=True)
                content_type = resp.headers.get("Content-Type", "")
                
                if resp.status_code == 200 and ("pdf" in content_type.lower() or resp.content[:4] == b'%PDF'):
                    with open(output_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Verify PDF is valid
                    if os.path.getsize(output_path) > 1000:  # At least 1KB
                        return True
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                continue
    
    return False

@st.cache_data
def extract_text_and_tables(pdf_path: str) -> Dict[str, any]:
    """Extract text and tables from PDF with enhanced error handling."""
    result = {
        'text': '',
        'tables': [],
        'metadata': {},
        'errors': []
    }
    
    # Extract text
    try:
        with pdfplumber.open(pdf_path) as pdf:
            result['metadata'] = {
                'pages': len(pdf.pages),
                'file_size_kb': os.path.getsize(pdf_path) / 1024
            }
            
            text_parts = []
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i} ---\n{page_text}")
            
            result['text'] = "\n\n".join(text_parts)
    except Exception as e:
        result['errors'].append(f"Text extraction error: {e}")
        logger.error(f"PDFPlumber error: {e}")
    
    # Extract tables
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice', suppress_stdout=True)
        if not tables:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream', suppress_stdout=True)
        
        for i, table in enumerate(tables, 1):
            df = table.df
            # Clean up empty cells
            df = df.replace('', None).dropna(how='all').dropna(axis=1, how='all')
            if not df.empty:
                result['tables'].append({
                    'index': i,
                    'data': df.to_dict('records'),
                    'csv': df.to_csv(index=False)
                })
    except Exception as e:
        result['errors'].append(f"Table extraction error: {e}")
        logger.warning(f"Camelot error: {e}")
    
    return result

def format_extracted_content(extraction_result: Dict) -> str:
    """Format extracted content for display and AI processing."""
    parts = []
    
    # Add text content
    if extraction_result['text']:
        parts.append("📄 TEXT CONTENT:\n" + extraction_result['text'])
    
    # Add tables
    if extraction_result['tables']:
        parts.append("\n📊 TABLES FOUND:")
        for table in extraction_result['tables']:
            parts.append(f"\nTable {table['index']}:\n{table['csv']}")
    
    # Add errors if any
    if extraction_result['errors']:
        parts.append("\n⚠️ EXTRACTION ISSUES:")
        for error in extraction_result['errors']:
            parts.append(f"- {error}")
    
    return "\n\n".join(parts)

def create_summary(text: str, max_chars: int = 500) -> str:
    """Create an intelligent summary of the text."""
    if not text:
        return "No content available"
    
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Try to find key sentences (those with important keywords)
    important_keywords = ['notice', 'announcement', 'date', 'deadline', 'important', 
                          'required', 'must', 'should', 'exam', 'fee', 'admission']
    
    sentences = text.split('.')
    important_sentences = []
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in important_keywords):
            important_sentences.append(sentence.strip())
    
    if important_sentences:
        summary = '. '.join(important_sentences[:3]) + '.'
        if len(summary) > max_chars:
            summary = summary[:max_chars] + '...'
        return summary
    
    # Fallback to simple truncation
    return text[:max_chars] + ('...' if len(text) > max_chars else '')

# ------------------ CHAT MANAGEMENT ------------------
class ChatManager:
    """Manage chat sessions for different notices."""
    
    @staticmethod
    def get_session_key(notice_index: int) -> str:
        return f"chat_notice_{notice_index}"
    
    @staticmethod
    def initialize_chat(notice_index: int, content: str):
        """Initialize or retrieve chat for a specific notice."""
        session_key = ChatManager.get_session_key(notice_index)
        
        if session_key not in st.session_state:
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                # Create a concise system prompt
                system_prompt = f"""You are a helpful assistant analyzing NSIT Notice #{notice_index}.
                
Here's the notice content:
{content[:3000]}  # Limit initial context to avoid token issues

Guidelines:
- Answer questions clearly and concisely
- Reference specific parts of the notice when relevant
- If information isn't in the notice, say so
- Be helpful with dates, deadlines, and requirements"""
                
                chat = model.start_chat(history=[
                    {"role": "user", "parts": system_prompt},
                    {"role": "model", "parts": "I understand. I'll help you with questions about this NSIT notice. What would you like to know?"}
                ])
                
                st.session_state[session_key] = {
                    'chat': chat,
                    'history': [],
                    'notice_index': notice_index
                }
            except Exception as e:
                st.error(f"Failed to initialize chat: {e}")
                return None
        
        return st.session_state[session_key]

# ------------------ STREAMLIT UI ------------------
st.set_page_config(
    page_title="NSIT Notices Chat",
    page_icon="📢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    .notice-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #1f77b4;
    }
    .notice-date {
        color: #666;
        font-size: 0.9em;
    }
    .summary-box {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.title("📢 NSIT Notices Chat with Gemini AI")
st.markdown("*Select a notice to view and ask questions about it*")

# Fetch notices
with st.spinner("Loading notices..."):
    notices = fetch_notice_links()

if not notices:
    st.error("No notices available. Please try again later.")
    st.stop()

# Create notice selection
col1, col2 = st.columns([3, 1])
with col1:
    # Format notice options for display
    notice_options = [f"{n['index']}. {n['title'][:50]}... ({n['date']})" 
                     for n in notices]
    selected_option = st.selectbox(
        "📋 Select a Notice:",
        notice_options,
        help="Choose a notice to view and chat about"
    )

selected_index = notice_options.index(selected_option)
selected_notice = notices[selected_index]

with col2:
    st.markdown(f"### Notice #{selected_notice['index']}")
    if st.button("🔄 Refresh Notices"):
        st.cache_data.clear()
        st.rerun()

# Process selected notice
file_id = extract_file_id(selected_notice['url'])
if not file_id:
    st.error("Could not parse the notice URL. Please try another notice.")
    st.stop()

pdf_path = os.path.join(NOTICES_DIR, f"notice_{selected_notice['index']}_{file_id[:8]}.pdf")

# Download PDF if needed
if not os.path.exists(pdf_path):
    with st.spinner(f"Downloading Notice #{selected_notice['index']}..."):
        if not download_google_doc_as_pdf(file_id, pdf_path):
            st.error("Failed to download the notice. It might be restricted or unavailable.")
            st.stop()

# Extract content
with st.spinner("Processing notice content..."):
    extraction_result = extract_text_and_tables(pdf_path)
    full_content = format_extracted_content(extraction_result)
    summary = create_summary(extraction_result['text'])

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header(f"📄 Notice #{selected_notice['index']}")
    
    # Metadata
    if extraction_result['metadata']:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pages", extraction_result['metadata'].get('pages', 'N/A'))
        with col2:
            size_kb = extraction_result['metadata'].get('file_size_kb', 0)
            st.metric("Size", f"{size_kb:.1f} KB")
    
    # Summary
    st.subheader("📝 Summary")
    st.info(summary)
    
    # Tables info
    if extraction_result['tables']:
        st.subheader(f"📊 Tables Found: {len(extraction_result['tables'])}")
        for table in extraction_result['tables']:
            with st.expander(f"Table {table['index']}"):
                st.text(table['csv'][:500])
    
    # Full content
    with st.expander("📖 View Full Notice Text"):
        st.text_area("", extraction_result['text'], height=400, key="full_text_sidebar")
    
    # Download options
    st.subheader("💾 Download Options")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📄 TXT",
            data=full_content,
            file_name=f"notice_{selected_notice['index']}.txt",
            mime="text/plain"
        )
    with col2:
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📑 PDF",
                data=f.read(),
                file_name=f"notice_{selected_notice['index']}.pdf",
                mime="application/pdf"
            )

# ------------------ CHAT INTERFACE ------------------
st.divider()
st.subheader("💬 Chat about this Notice")

# Initialize chat for selected notice
chat_session = ChatManager.initialize_chat(selected_notice['index'], full_content)

if chat_session:
    # Display chat history
    for msg in chat_session['history']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about this notice..."):
        # Add user message
        chat_session['history'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_session['chat'].send_message(prompt)
                    reply = response.text
                    st.markdown(reply)
                    chat_session['history'].append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Error generating response: {e}")
            logger.error(f"Gemini API error: {e}")

    # Quick action buttons
    st.markdown("### 🎯 Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📅 What are the dates?"):
            prompt = "What are the important dates mentioned in this notice?"
            chat_session['history'].append({"role": "user", "content": prompt})
            response = chat_session['chat'].send_message(prompt)
            st.info(response.text)
    
    with col2:
        if st.button("✅ Requirements?"):
            prompt = "What are the requirements or eligibility criteria mentioned?"
            chat_session['history'].append({"role": "user", "content": prompt})
            response = chat_session['chat'].send_message(prompt)
            st.info(response.text)
    
    with col3:
        if st.button("⏰ Deadlines?"):
            prompt = "Are there any deadlines mentioned in this notice?"
            chat_session['history'].append({"role": "user", "content": prompt})
            response = chat_session['chat'].send_message(prompt)
            st.info(response.text)
else:
    st.error("Failed to initialize chat session. Please check your API key.")

# Footer
st.divider()
st.caption("Built with Streamlit, Google Gemini AI, and ❤️ for NSIT")