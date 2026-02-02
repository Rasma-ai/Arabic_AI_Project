import re

def clean_text(text):
    """
    Clean Arabic text for sentiment analysis.
    """
    
    text = str(text)

    text = re.sub(r'[A-Za-z0-9]', '', text)

    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)

    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

