"""Renumber the cited sources 1, 2, 3... in the order they are first cited, one number per paper page."""
import re

CITATION = re.compile(r"\[(\d+)\]")
CITATION_RUN = re.compile(r"(?:\[\d+\])+")


def renumber(answer, sources):
    """Return the answer and its sources numbered from 1 without gaps; passages from the same page share a number."""
    page_of = {source["n"]: (source["arxiv_id"], source["page"]) for source in sources}
    number = {}
    for match in CITATION.finditer(answer):
        page = page_of.get(int(match.group(1)))
        if page and page not in number:
            number[page] = len(number) + 1
    for page in page_of.values():  # a listed page never cited in the text goes last
        number.setdefault(page, len(number) + 1)

    def relabel(match):
        page = page_of.get(int(match.group(1)))
        return f"[{number[page]}]" if page else match.group(0)

    def drop_repeats(run):
        # "[1][2][1]" -> "[1][2]": a page cited twice in the same group is shown once
        return "".join(f"[{n}]" for n in dict.fromkeys(CITATION.findall(run.group(0))))

    answer = CITATION_RUN.sub(drop_repeats, CITATION.sub(relabel, answer))

    first_source = {}
    for source in sources:
        first_source.setdefault(page_of[source["n"]], source)
    sources = [{**first_source[page], "n": n} for page, n in sorted(number.items(), key=lambda item: item[1])]
    return answer, sources