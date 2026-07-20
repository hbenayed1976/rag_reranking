# -*- coding: utf-8 -*-
import re

LETTERS = "أبجد"
LETTER_TO_LATIN = {"أ": "A", "ب": "B", "ج": "C", "د": "D"}
_LATIN_TO_LETTER = {v: k for k, v in LETTER_TO_LATIN.items()}

# Lookahead de fin de lettre : ponctuation habituelle, tiret (ASCII '-' et
# tiret demi-cadratin '–'), N'IMPORTE QUEL espace (donc \n, \t, ' '...), ou
# fin de chaîne. Non consommant (?=...) : ne mange plus le caractère, donc
# n'empêche plus un match suivant si besoin.
_END = r'(?=[:\.\)\,،\-–]|\s|$)'

_ISOLATED_LETTER = re.compile(
    r'(?:(?<=^)|(?<=[\s\(]))([أبجد])' + _END,
    re.MULTILINE,
)

_KEYWORD_LETTER = re.compile(
    r'(?:الإجابة(?:\s+الصحيحة)?|الجواب(?:\s+الصحيح)?|الخيار(?:\s+الصحيح)?)'
    r'\s*(?:هي|هو)?\s*:?\s*'
    r'\(?([أبجد])\)?' + _END,
)

# "The correct answer is: B" / "...is: ب (...)" / "correct choice is A"
_ENGLISH_KEYWORD_LETTER = re.compile(
    r'(?:correct\s+)?(?:answer|choice|option)\s+is\s*:?\s*'
    r'\(?([ABCDأبجد])\)?' + _END,
    re.IGNORECASE,
)


def _to_latin(letter: str) -> str:
    return LETTER_TO_LATIN.get(letter, letter.upper())


def extract_answer_letter(generated_text: str):
    """Renvoie la lettre latine (A/B/C/D) correspondant à la réponse finale
    du modèle, ou None si aucun motif valide n'est trouvé.

    Priorité : DERNIÈRE déclaration mot-clé arabe ("الإجابة/الجواب/الخيار
    [الصحيح] هو/هي: X"), puis DERNIÈRE déclaration mot-clé anglaise
    ("(correct) answer/choice is: X"), puis PREMIÈRE lettre isolée valide
    dans le texte."""
    if not generated_text:
        return None

    kw_matches = list(_KEYWORD_LETTER.finditer(generated_text))
    if kw_matches:
        return _to_latin(kw_matches[-1].group(1))

    en_matches = list(_ENGLISH_KEYWORD_LETTER.finditer(generated_text))
    if en_matches:
        return _to_latin(en_matches[-1].group(1))

    match = _ISOLATED_LETTER.search(generated_text)
    if not match:
        return None
    return _to_latin(match.group(1))


def validate_answer(generated_text: str, correct_letter) -> bool:
    """correct_letter : lettre arabe (أ/ب/ج/د) OU latine (A/B/C/D)."""
    pred_latin = extract_answer_letter(generated_text)
    if pred_latin is None:
        return False
    correct = correct_letter.strip()
    correct_latin = _to_latin(correct) if correct in LETTER_TO_LATIN else correct.upper()
    return pred_latin == correct_latin
