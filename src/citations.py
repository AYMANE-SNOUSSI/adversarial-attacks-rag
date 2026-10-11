"""Renumber the cited sources 1, 2, 3... in the order they are first cited."""
import re

CITATION = re.compile(r"\[(\d+)\]")


def renumber(answer, sources):
    """Return the answer and its sources with citations renumbered from 1, without gaps."""
    cited = {source["n"] for source in sources}
    order = {}
    for match in CITATION.finditer(answer):
        n = int(match.group(1))
        if n in cited and n not in order:
            order[n] = len(order) + 1
    for source in sources:  # a listed source never cited in the text goes last
        order.setdefault(source["n"], len(order) + 1)

    answer = CITATION.sub(lambda match: f"[{order.get(int(match.group(1)), match.group(1))}]", answer)
    sources = sorted(({**source, "n": order[source["n"]]} for source in sources), key=lambda source: source["n"])
    return answer, sources