"""
Generation by NGrams starter
"""

# pylint:disable=too-many-locals, unused-import, unused-variable

from main import GreedyTextGenerator, NGramLanguageModel, TextProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open(
        "./lab_3_generate_by_ngrams/assets/Harry_Potter.txt", "r", encoding="utf-8"
    ) as text_file:
        text = text_file.read()

    text_processor = TextProcessor("_")

    tokens = text_processor._tokenize(text)
    encoded_text = text_processor.encode(text)
    decoded_text = text_processor.decode(encoded_text)

    print("First 50 tokens:", tokens[:50])
    print("Encoded text example:", encoded_text[:50])
    print("Decoded text example:", decoded_text[:200])

    language_model = NGramLanguageModel(encoded_text, 7)

    mid_result = language_model.build()
    assert mid_result

    greedy_generator = GreedyTextGenerator(language_model, text_processor)

    result = greedy_generator.run("Vernon", 51)
    assert result

    print("Generated text:", result)


if __name__ == "__main__":
    main()