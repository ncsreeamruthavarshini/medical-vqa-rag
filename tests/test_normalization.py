from src.answer_normalization import answers_match, is_yes_no, normalize_answer


def test_whitespace_and_case():
    assert normalize_answer("  Yes  ") == "yes"
    assert answers_match("YES", "yes")


def test_trailing_punctuation():
    assert normalize_answer("no.") == "no"
    assert normalize_answer("axial!") == "axial"


def test_yes_is_never_no():
    assert normalize_answer("yes") != normalize_answer("no")
    assert not answers_match("yes", "no")


def test_not_seen_here_is_not_no():
    assert normalize_answer("not seen here") == "not seen here"
    assert not is_yes_no("not seen here")
    assert not answers_match("not seen here", "no")


def test_yes_phrase_is_not_automatically_yes():
    assert normalize_answer("yes there is an aneurysm") == "yes there is an aneurysm"
    assert not is_yes_no("yes there is an aneurysm")
    assert not answers_match("yes there is an aneurysm", "yes")


def test_tabs_and_internal_space():
    assert normalize_answer("skull\tcartilage and medulla") == "skull cartilage and medulla"


def test_empty_and_none():
    assert normalize_answer("") == ""
    assert normalize_answer(None) == ""
