import hashlib
import io
import os
import re
import time
from html import escape
from pathlib import Path

# Paddle settings
  
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"   
os.environ["FLAGS_use_mkldnn"] = "0"                           
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import joblib
import numpy as np
import pandas as pd
import pymupdf                      
import streamlit as st
from PIL import Image, ImageOps

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
MODEL_DIR = Path(__file__).parent     
MAX_PAGES = 3                          
MAX_SIDE = 1600                        
MIN_CHARS = 30                        
CPU_THREADS = 4                      
KEEP_LAST = 5                          

CATEGORY_STYLE = {
    "invoice": {"icon": "🧾", "color": "#6366f1", "tagline": "A request for payment sent by a seller"},
    "receipt": {"icon": "🛒", "color": "#10b981", "tagline": "Proof of a payment that was already made"},
    "bill":    {"icon": "💡", "color": "#f59e0b", "tagline": "A charge for a service or for usage"},
}

st.set_page_config(page_title="Doc_classifier...", page_icon="🗂️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="css"] {font-family: 'Poppins', sans-serif;}

.hero {background: linear-gradient(120deg, #312e81 0%, #6d28d9 55%, #db2777 100%);
       padding: 2rem 2.2rem; border-radius: 20px; color: #fff; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2.2rem; color: #fff;}
.hero p {margin: 0.4rem 0 0 0; opacity: 0.92;}
.steps {display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 1rem;}
.step {background: rgba(255,255,255,0.18); padding: 0.35rem 0.9rem;
       border-radius: 999px; font-size: 0.85rem;}

.card {border-radius: 18px; padding: 1.3rem 1.5rem; color: #fff; margin-bottom: 1rem;}
.card .big {font-size: 2.1rem; font-weight: 700; margin: 0;}
.card .sub {opacity: 0.92; margin: 0.2rem 0 0 0;}

.bar-row {margin: 0.7rem 0;}
.bar-label {display: flex; justify-content: space-between; font-size: 0.95rem; margin-bottom: 0.25rem;}
.bar-track {background: rgba(128,128,128,0.22); border-radius: 999px; height: 12px; overflow: hidden;}
.bar-fill {height: 12px; border-radius: 999px;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Loading the classifier and the OCR engine
# ---------------------------------------------------------------
@st.cache_resource
def load_classifier():
    model = joblib.load(MODEL_DIR / "model.joblib")
    if hasattr(model, "steps"):                    # saved as a pipeline
        return model.steps[0][1], model.steps[-1][1]
    return joblib.load(MODEL_DIR / "tfidf.joblib"), model


@st.cache_resource
def load_ocr():
    """Returns (engine name, engine, short note). Cleared after every document."""
    try:
        from paddleocr import PaddleOCR
        base = dict(lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False)
        # try the best settings first; drop options this Paddle version does not know
        options = [
            dict(enable_mkldnn=False, cpu_threads=CPU_THREADS, use_textline_orientation=False),
            dict(enable_mkldnn=False, cpu_threads=CPU_THREADS),
            dict(enable_mkldnn=False),
        ]
        for extra in options:
            try:
                engine = PaddleOCR(**base, **extra)
                threads = extra.get("cpu_threads", "default")
                return "PaddleOCR", engine, f"{threads} CPU threads"
            except (TypeError, ValueError):
                continue
    except Exception:
        pass

    # fallback: Tesseract
    try:
        import pytesseract
        tessdata = MODEL_DIR / "tessdata"
        if tessdata.exists():
            os.environ["TESSDATA_PREFIX"] = str(tessdata)
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        return "Tesseract (fallback)", pytesseract, "fallback engine"
    except Exception:
        return None, None, ""


def warm_up():
    # build a fresh OCR engine now, so the next upload does not wait for it
    with st.spinner("Preparing the OCR engine for the next document..."):
        load_ocr()


# ---------------------------------------------------------------
# Reading the document
# ---------------------------------------------------------------
def read_image(engine_name, engine, img):
    img = ImageOps.exif_transpose(img).convert("RGB")

    longest = max(img.size)                         
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        img = img.resize((int(img.width * scale), int(img.height * scale)))

    if engine_name == "PaddleOCR":
        pixels = np.array(img)[:, :, ::-1]          
        result = engine.predict(pixels)
        return "\n".join(result[0]["rec_texts"]) if result else ""
    return engine.image_to_string(ImageOps.grayscale(img))


def read_pages(pdf, pages_to_read, bar):
    engine_name, engine, note = load_ocr()
    if engine is None:
        raise RuntimeError("No OCR engine found. Install PaddleOCR or Tesseract.")

    texts = []
    for number in range(pages_to_read):
        st.write(f"Reading page {number + 1} of {pages_to_read}...")
        picture = pdf[number].get_pixmap(dpi=200)    
        img = Image.open(io.BytesIO(picture.tobytes("png")))
        texts.append(read_image(engine_name, engine, img))
        bar.progress((number + 1) / pages_to_read)
    return "\n".join(texts), engine_name, note


def read_document(pdf_bytes, bar):
    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        total_pages = pdf.page_count
        if total_pages == 0:
            raise ValueError("The PDF has no pages.")
        pages_to_read = min(total_pages, MAX_PAGES)

        small = pdf[0].get_pixmap(dpi=70)           
        preview = Image.open(io.BytesIO(small.tobytes("png")))

        started = time.time()
        try:
            text, engine_name, note = read_pages(pdf, pages_to_read, bar)
        except Exception:
            st.write("Something went wrong, retrying with a fresh OCR engine...")
            load_ocr.clear()
            text, engine_name, note = read_pages(pdf, pages_to_read, bar)

        return {"text": text, "pages": total_pages, "engine": engine_name,
                "note": note, "seconds": round(time.time() - started),
                "preview": preview}
    finally:
        pdf.close()


# ---------------------------------------------------------------
# Cleaning and predicting (same steps as in training)
# ---------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "0", text)              
    text = re.sub(r"[^a-z0 ]+", " ", text)        
    return re.sub(r"\s+", " ", text).strip()


def get_probabilities(classifier, features):
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(features)[0]
    scores = classifier.decision_function(features)[0]   
    exp = np.exp(scores - scores.max())
    return exp / exp.sum()


def top_words(vectorizer, classifier, features, class_index, n=8):
    if not hasattr(classifier, "coef_"):
        return None                               
    row = features.toarray()[0]
    present = row.nonzero()[0]
    contribution = row[present] * classifier.coef_[class_index][present]
    order = contribution.argsort()[::-1][:n]
    names = vectorizer.get_feature_names_out()[present]
    return pd.DataFrame({"word or phrase": names[order],
                         "push toward this category": contribution[order].round(3)})


def confidence_bars(classes, probs):
    html = ""
    for name, p in sorted(zip(classes, probs), key=lambda pair: -pair[1]):
        color = CATEGORY_STYLE.get(name, {"color": "#6b7280"})["color"]
        html += (f"<div class='bar-row'><div class='bar-label'>"
                 f"<span>{escape(str(name).title())}</span><span>{p:.0%}</span></div>"
                 f"<div class='bar-track'><div class='bar-fill' "
                 f"style='width:{p * 100:.1f}%; background:{color}'></div></div></div>")
    return html


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Upload a PDF\n"
        "2. Every page is turned into an image and read by OCR\n"
        "3. The text is cleaned and turned into numbers (TF-IDF)\n"
        "4. A trained model picks the category"
    )
    threshold = st.slider(
        "Minimum confidence", 0.30, 0.95, 0.60, 0.05,
        help="Below this, the app says 'inconclusive' instead of guessing.")
    st.caption(f"Reads up to {MAX_PAGES} pages per PDF.")


# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🗂️ Doc_classifier...</h1>
  <p>Drop in a PDF and find out whether it is an invoice, a receipt or a bill.</p>
  <div class="steps">
    <span class="step">1 · Upload</span>
    <span class="step">2 · Read the text</span>
    <span class="step">3 · Get the category</span>
  </div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Drop a PDF here", type=["pdf"])
if uploaded is None:
    st.caption("Nothing uploaded yet.")
    warm_up()
    st.stop()


# ---------------------------------------------------------------
# Read the document 
# ---------------------------------------------------------------
pdf_bytes = uploaded.getvalue()
doc_key = hashlib.md5(pdf_bytes).hexdigest()
store = st.session_state.setdefault("results", {})

if doc_key not in store:
    failure = None
    with st.status("Reading your document...", expanded=True) as status:
        st.write("Large or detailed scans can take 1 to 2 minutes. Please keep this tab open.")
        bar = st.progress(0.0)
        try:
            store[doc_key] = read_document(pdf_bytes, bar)
            status.update(label=f"Document read in {store[doc_key]['seconds']} s",
                          state="complete", expanded=False)
            while len(store) > KEEP_LAST:              
                store.pop(next(iter(store)))
        except Exception as error:
            failure = error
            status.update(label="Could not read this document", state="error")
        finally:
            load_ocr.clear()        

    if failure is not None:
        st.error(f"This file could not be processed. ({failure})")
        warm_up()
        st.stop()


# ---------------------------------------------------------------
# Show the result
# ---------------------------------------------------------------
doc = store[doc_key]
text = doc["text"]
total_pages = doc["pages"]
cleaned = clean_text(text)
vectorizer, classifier = load_classifier()

left, right = st.columns([1, 2], gap="large")
with left:
    st.image(doc["preview"], caption=f"{uploaded.name} · page 1 of {total_pages}")

probs = None
with right:
    if len(cleaned) < MIN_CHARS:
        st.warning("Not enough readable text was found in this document, so it cannot "
                   "be classified. Try a clearer or higher-quality PDF.")
    else:
        features = vectorizer.transform([cleaned])
        probs = get_probabilities(classifier, features)
        best = int(probs.argmax())
        label = classifier.classes_[best]
        confidence = float(probs[best])
        style = CATEGORY_STYLE.get(label, {"icon": "📄", "color": "#6b7280", "tagline": ""})

        if confidence < threshold:
            st.markdown(
                f"<div class='card' style='background:linear-gradient(135deg,#d97706,#b45309)'>"
                f"<p class='big'>🤔 Not sure</p>"
                f"<p class='sub'>Best guess: {escape(str(label))} ({confidence:.0%}), "
                f"which is below your {threshold:.0%} limit.</p></div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='card' style='background:linear-gradient(135deg,{style['color']},{style['color']}bb)'>"
                f"<p class='big'>{style['icon']} {escape(str(label)).upper()}</p>"
                f"<p class='sub'>{style['tagline']}</p></div>",
                unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{confidence:.0%}")
        m2.metric("Pages read", f"{min(total_pages, MAX_PAGES)} of {total_pages}")
        m3.metric("Characters", len(text.strip()))
        st.caption(f"Read with {doc['engine']} ({doc['note']}) in {doc['seconds']} s")

st.divider()

if probs is None:
    st.subheader("Extracted text")
    st.text_area("Text read from the document", text, height=250)
else:
    tab_text, tab_conf, tab_why = st.tabs(["📝 Text", "📊 Scores", "🔍 Why"])

    with tab_text:
        st.text_area("Text read from the document", text, height=300)

    with tab_conf:
        st.markdown(confidence_bars(classifier.classes_, probs), unsafe_allow_html=True)

    with tab_why:
        table = top_words(vectorizer, classifier, features, best)
        if table is None:
            st.write("This model type does not provide word-level explanations.")
        else:
            st.write(f"Words in this document that pushed the model toward **{label}**:")
            st.dataframe(table, hide_index=True)

warm_up()      