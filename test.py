from pathlib import Path

law_path = Path(r"G:\graduate_1\Code\Camel\data\meta\laws.txt")
text = law_path.read_text(encoding="utf-8", errors="ignore")
print(text)
