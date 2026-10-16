import re
import unicodedata
 
HTML_RE = re.compile(r"<[^>]+>")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}|\{[A-Za-z_]+\}|\[[A-Za-z _]+\]")
SPACES_RE = re.compile(r"[ \t]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")
 
 
def _to_str(value) -> str:
    """None / NaN (bos deyer) -> bos setir, qalani str."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value)
 
 
def normalize(text, stats: dict | None = None) -> str:
    text = _to_str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HTML_RE.sub(" ", text)
 
    if stats is not None:
        stats["emails_masked"] = stats.get("emails_masked", 0) + len(EMAIL_RE.findall(text))
    text = EMAIL_RE.sub("<EMAIL>", text)
 
    if stats is not None:
        stats["phones_masked"] = stats.get("phones_masked", 0) + len(PHONE_RE.findall(text))
    text = PHONE_RE.sub("<PHONE>", text)
 
    if stats is not None:
        stats["placeholder_like_found"] = (
            stats.get("placeholder_like_found", 0) + len(PLACEHOLDER_RE.findall(text))
        )
    text = PLACEHOLDER_RE.sub(" ", text)
 
    text = SPACES_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()
 
 
def build_text(subject, body) -> str:
    """clean.py ile eyni: normalize(subject) + 2 setir + normalize(body)."""
    return normalize(subject) + "\n\n" + normalize(body)