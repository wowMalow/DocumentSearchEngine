import re

from typing import List


class Cleaner:
    def clean_text(self, text: str) -> str:
        cleaned = re.sub(r"<(.|\n)+?>", '', text.lower())
        cleaned = re.sub(r"[\W]", " ", cleaned)
        cleaned = re.sub(r"&\w+?;", "", cleaned)
        return re.sub(r"\s+", " ", cleaned)

class Tokenizer(Cleaner):
    def tokenize(self, text: str) -> List[str]:
        cleaned = self.clean_text(text)
        return re.findall(r"\w+", cleaned)
    
    
        