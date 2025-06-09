import spacy
from spacy.matcher import PhraseMatcher
from typing import Dict, List


class DiamondEntityExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="TEXT")
        self.entity_definitions = {
            "Clarity": ['VVS2', 'SI2', 'I1', 'I3', 'SI1', 'VS1', 'VVS1', 'I2', 'VS2', 'IF', 'SI3', 'FL'],
            "Color": ['M', 'i', 'w-x', 'M+', 'x', 'l', 'N', 'L+', 'f', 'J', 'y', 'G', 'O', 'Y-Z', 'u-v', 'z',
                      'E+', 'p', 'j', 'U-V', 'F+', 'm', 'I', 'k', 'J+', 'g', 'H+', 'I+', 'q', 'e', 'o-p', 'N+',
                      'K+', 't', 'd', 'W-X', 'Q-R', 'v', 'O-P', 'D', 's-t', 'q-r', 'r', 'u', 'w', 's', 'y-z',
                      'n', 'L', 'D+', 'H', 'F', 'K', 'E', 'G+', 'S-T'],
            "Cut": ['VG', 'EX', 'P', 'G', 'F'],
            "Polish": ['P', 'G', 'F', 'VG', 'EX'],
            "Lab": ['IOD', 'HRD', 'GIA']
        }
        self._register_patterns()

    def _register_patterns(self):
        for label, terms in self.entity_definitions.items():
            patterns = [self.nlp.make_doc(term) for term in terms]
            self.matcher.add(label, patterns)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        matches = self.matcher(doc)
        results = {}

        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            value = doc[start:end].text.upper()
            results.setdefault(label, set()).add(value)

        # Convert sets to sorted lists
        return {key: sorted(list(val)) for key, val in results.items()}
