"""
Language detection starter.
"""

# pylint: disable=unused-variable, duplicate-code

def main() -> None:
    """
    Launches an implementation.
    """
    from main import (
        calculate_frequencies,
        calculate_mse,
        check_profile,
        compare_profiles_by_mse,
        compare_profiles_by_top_n,
        create_language_profile,
        detect_language_by_top_n,
        detect_language_by_mse,
        get_top_n_words,
        remove_stop_words,
        tokenize,
    )

    with open("lab_1_classify_profile/assets/texts/de.txt", "r", encoding="utf-8") as file:
        de_text = file.read()
    with open("lab_1_classify_profile/assets/texts/unknown.txt", "r", encoding="utf-8") as file:
        unknown_text = file.read()
    with open("lab_1_classify_profile/assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    with open("lab_1_classify_profile/assets/texts/en.txt", "r", encoding="utf-8") as file:
        en_text = file.read()
    tokens = tokenize(de_text)
    clean_text = remove_stop_words(tokens, stopwords)
    freq_dict = calculate_frequencies(clean_text)
    result = get_top_n_words(freq_dict, 7)
    assert result, "Detection result is None"

if __name__ == "__main__":
    main()


