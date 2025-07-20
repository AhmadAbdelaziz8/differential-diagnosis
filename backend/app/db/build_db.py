import pymupdf
import os

def extract_text_from_pdf(pdf_path):
    print('starting to extract text from pdf')
    all_text = []

    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text() 
            if text: 
                all_text.append({
                    "page_content": text,
                    "metadata": {"source": "Oxford Handbook", "page": page.number + 1}
                })

    print(f'finished extracting text from pdf, the total number of pages is {len(all_text)}')
    return all_text


extract_text_from_pdf('assets/oxford.pdf')
