def make_pdf(pdf_path, meta, model_output, guidelines, artifacts):
    # Dummy PDF generator
    with open(pdf_path, 'wb') as f:
        f.write(b'%PDF-1.4\n%Dummy SurgicalAI PDF\n')
