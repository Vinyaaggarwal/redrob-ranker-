import zipfile
import xml.etree.ElementTree as ET
import sys
import os

def get_docx_text(path):
    WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    TEXT = WORD_NAMESPACE + 't'
    PARA = WORD_NAMESPACE + 'p'
    
    if not os.path.exists(path):
        return f"Error: File {path} does not exist."
        
    try:
        with zipfile.ZipFile(path) as docx:
            tree = ET.parse(docx.open('word/document.xml'))
            root = tree.getroot()
            paragraphs = []
            for paragraph in root.iter(PARA):
                texts = [node.text for node in paragraph.iter(TEXT) if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            return "\n".join(paragraphs)
    except Exception as e:
        return f"Error reading {path}: {str(e)}"

def main():
    src_dir = r"C:\Users\vinya\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"
    dest_dir = r"c:\Users\vinya\OneDrive\Documents\redrob\redrob-ranker-"
    
    files = {
        "README.docx": "readme.txt",
        "job_description.docx": "job_description.txt",
        "redrob_signals_doc.docx": "redrob_signals_doc.txt",
        "submission_spec.docx": "submission_spec.txt"
    }
    
    for docx_name, txt_name in files.items():
        docx_path = os.path.join(src_dir, docx_name)
        txt_path = os.path.join(dest_dir, txt_name)
        print(f"Converting {docx_name} to {txt_name}...")
        text = get_docx_text(docx_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Done. Wrote {os.path.getsize(txt_path)} bytes.")

if __name__ == '__main__':
    main()
