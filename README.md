# OCR Document Classifier: Invoice, Bill or Receipt

> **Mini project built during my internship at SURE ProEd** (a unit of SURE Trust, Skill Upgradation for Rural-youth Empowerment).
> Mentor: `Mr. Vamsi Krishna` | Author: `Suraj Tiwari`

Upload a PDF or an image of a document. The app reads the text with **PaddleOCR**, then a **TF-IDF + Logistic Regression** model decides whether the document is an **invoice**, a **bill** or a **receipt**, and shows how confident it is.


## What it does

- Accepts a PDF (scanned or digital).
- Extracts the text with PaddleOCR and shows it in the interface.
- Predicts the document category (Invoice, Bill or Receipt) with a confidence score.
- Shows a message when the document cannot be processed or the prediction is inconclusive.

## How it works

```
       PDF 
        |
        v
 PDF pages rendered to images (PyMuPDF, 200 DPI, first 3 pages)
        |
        v
 PaddleOCR (English)  ->  raw text
        |
        v
 Text cleaning: lowercase, numbers -> 0, remove symbols
        |
        v
 TF-IDF (1-2 word n-grams)  ->  Logistic Regression
        |
        v
 Invoice / Bill / Receipt + confidence
```

Every file goes through OCR the same way, including digital PDFs. This keeps "the PDF has embedded text" from leaking the label to the model.

## Project structure

```
ocr-doc-classifier/
├── app.py                     # Streamlit app
├── model.joblib               # trained Logistic Regression
├── tfidf.joblib               # fitted TF-IDF vectorizer
├── requirements.txt
├── README.md
├── .gitignore
└── notebooks/
    └── ocr_doc_classifier.ipynb   # data prep, OCR, training and evaluation (Google Colab)
```

## Run it locally

Set up for VS Code with a virtual environment. Python 3.10 to 3.13.

```bash
# 1. clone the repo
git clone https://github.com/Suraj03-07/ocr-doc-classifier.git
cd <repo-name>

# 2. create and activate a virtual environment
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 3. install the requirements
pip install -r requirements.txt

# 4. start the app
streamlit run app.py
```

Notes:
- The first upload is slow because PaddleOCR downloads its model files the first time it runs.
- The `.joblib` files were saved with a specific scikit-learn version. If loading fails with a version warning, install the same version that trained the model (see `requirements.txt`).
- OCR on a CPU takes a few seconds per page. The model was trained on a Colab GPU, but the app works on CPU.

## Dataset

The training notebook builds a small, labelled dataset from public sources plus generated bills. Final size after removing duplicates: **441 documents** (360 train, 81 test).

| Class | Train | Test |
|---|---|---|
| Invoice | 130 | 30 |
| Receipt | 139 | 30 |
| Bill | 91 | 21 |

| Source | Used for | Licence (as shown on the source page) |
|---|---|---|
| FATURA (Hugging Face copy `mathieu1256/FATURA2-invoices`) | Invoices, split by template | CC BY 4.0 |
| RVL-CDIP invoices (`chainyo/rvl-cdip-invoice`) | Real, noisy scanned invoices | "other" (research use) |
| katanaml invoices (`katanaml-org/invoices-donut-data-v1`) | Invoices | MIT |
| CORD v2 (`naver-clova-ix/cord-v2`) | Receipts | CC BY 4.0 |
| SROIE (ICDAR 2019) | Receipts | check the source page before reuse |
| Public sample utility bills | Real bills | various, see the source pages |
| Synthetic bills (generated with ReportLab) | Bills: electricity, water, gas, internet, mobile | generated for this project |

The dataset itself is **not stored in this repository**. Please check each source's licence before reusing the data.

### How the split avoids easy wins

- FATURA test invoices come from templates that never appear in training (0 of 10 test templates seen in training).
- Synthetic test bills use a page layout that never appears in training.
- Exact duplicate texts were removed.

## Results

Evaluated on the 81 held-out test documents:

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression (used in the app) | 1.00 | 1.00 |
| Random Forest | 0.99 | 0.99 |

**Please read these numbers with care.** The test set is small, and a stricter "leave one source out" check (train on all sources except one, test on the one left out) averaged only about **0.48** accuracy for the same baseline model (0.58 with character n-grams). The model learns the style of the sources it was trained on, so it can be wrong on documents that look very different from the training data. The confidence score and the "inconclusive" message exist for this reason.

## Limitations and future work

- Small dataset, and the bill class relies partly on synthetic documents.
- English text only.
- Weak on document styles that are far from the training sources.
- Ideas: add more real bills and receipts from more regions, try character n-gram features, compare a layout-aware model (for example LayoutLM), and add a feedback button to collect corrected labels.

## Acknowledgements

- **SURE ProEd** (a unit of SURE Trust) and my mentor `Mr Vamsi Krishna` for guidance during the internship.
- Dataset authors: FATURA, CORD (Naver Clova), SROIE (ICDAR 2019), RVL-CDIP, and katanaml.
- Libraries: PaddleOCR, PyMuPDF, scikit-learn, Streamlit.

## Author

`Suraj Tiwari` | [GitHub](https://github.com/Suraj03-07) | [LinkedIn](https://www.linkedin.com/in/suraj-tiwari-1491a72a0)

## Licence

To be confirmed with the mentor. Dataset licences are listed above and apply to the data, not to this code.
