"""
BPE Tokenizer starter
"""

# pylint:disable=too-many-locals, unused-variable
from main import collect_frequencies, get_vocabulary, train

def main() -> None:
    """
    Launches an implementation
    """
    with open("lab_2_tokenize_by_bpe/assets/text.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/en_raw.txt", "r", encoding="utf-8") as text_file:
        text_reference_translation = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/en_encoded.txt", "r", encoding="utf-8") as text_file:
        translation_encoded_raw = text_file.read()
    result = None
    result = collect_frequencies(text, None, "</s>")
    print(result)
    freqs = train(result, 100)
    if freqs is None:
        return
    new_vocab = get_vocabulary(freqs, "<unk>")
    print(new_vocab)
    assert result, "Translation not working"


if __name__ == "__main__":
    main()
