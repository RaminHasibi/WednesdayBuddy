import io

import pypdf


def extract_page_pdf(source_path, page_num):
    with open(source_path, "rb") as f:
        pdf_reader = pypdf.PdfReader(f)
        source_page = pdf_reader.pages[page_num]
        writer = pypdf.PdfWriter()
        writer.add_page(source_page)
        stream = io.BytesIO(b"")
        _, stream = writer.write(stream)
        stream.seek(0)
        return stream.read()
