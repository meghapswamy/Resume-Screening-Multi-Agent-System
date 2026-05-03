import fitz  # PyMuPDF

def read_pdf(file_path: str) -> str:
    """
    Extracts clean text from a PDF resume.
    Returns the full text as a single string.
    """
    doc = fitz.open(file_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    doc.close()
    return full_text.strip()