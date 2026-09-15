from pathlib import Path
from pypdf import PdfReader

PDF_PATH = Path("data/research_report_rag.pdf")

print("=" * 70)
print("PDF INSPECTION")
print("=" * 70)

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH.resolve()}\n"
        "Check that research_paper.pdf is inside Day-3/data/."
    )

reader = PdfReader(str(PDF_PATH))

print(f"\nPDF file: {PDF_PATH.name}")
print(f"Total pages loaded: {len(reader.pages)}")

for page_number, page in enumerate(reader.pages[:3], start=1):
    extracted_text = page.extract_text() or ""

    print("\n" + "-" * 70)
    print(f"PAGE {page_number}")
    print("-" * 70)
    print(f"Characters extracted: {len(extracted_text)}")
    print("\nExtracted text preview:")
    print(extracted_text[:700])