"""Citations are renumbered 1, 2, 3... in order of first appearance; passages from the same page share a number."""
from citations import renumber


def source(n):
    return {"n": n, "arxiv_id": f"id{n}", "page": n, "title": f"Paper {n}"}


def test_renumbers_in_order_of_first_citation():
    answer, sources = renumber("A [2][4]. B [2][3].", [source(2), source(3), source(4)])
    assert answer == "A [1][2]. B [1][3]."
    assert [(s["n"], s["arxiv_id"]) for s in sources] == [(1, "id2"), (2, "id4"), (3, "id3")]


def test_already_ordered_answer_is_unchanged():
    answer, sources = renumber("A [1]. B [2].", [source(1), source(2)])
    assert answer == "A [1]. B [2]."
    assert [s["n"] for s in sources] == [1, 2]


def test_unknown_citation_is_left_as_is():
    answer, sources = renumber("A [3]. B [7].", [source(3)])
    assert answer == "A [1]. B [7]."
    assert [s["arxiv_id"] for s in sources] == ["id3"]


def test_passages_from_the_same_page_share_one_number():
    sources = [
        {"n": 1, "arxiv_id": "idA", "page": 2, "title": "A"},
        {"n": 2, "arxiv_id": "idB", "page": 1, "title": "B"},
        {"n": 3, "arxiv_id": "idA", "page": 2, "title": "A"},
        {"n": 4, "arxiv_id": "idC", "page": 1, "title": "C"},
    ]
    answer, sources = renumber("X [1][2][3]. Y [1][4]. Z [4][3].", sources)
    assert answer == "X [1][2]. Y [1][3]. Z [3][1]."
    assert [(s["n"], s["arxiv_id"], s["page"]) for s in sources] == [(1, "idA", 2), (2, "idB", 1), (3, "idC", 1)]