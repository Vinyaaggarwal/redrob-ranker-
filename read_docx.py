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

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python read_docx.py <path_to_docx_file>")
        sys.exit(1)
    
    # Reconfigure stdout for Windows console UTF-8 support
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
        
    print(get_docx_text(sys.argv[1]))
