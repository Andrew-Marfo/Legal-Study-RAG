"""Prompts that keep answers grounded in the student's own materials.

Fabricated case names, holdings and citations are the single biggest risk in a
legal study tool, so the system prompt is deliberately strict and repetitive
about its one rule: use the supplied excerpts and nothing else.
"""

from __future__ import annotations

from typing import Sequence

from src.vector_store import SearchHit

# The exact wording the model must use when the excerpts do not answer the
# question. The UI matches on this, so keep the two in sync.
INSUFFICIENT_CONTEXT_PHRASE = "I couldn't find this in your materials."

NO_CONTEXT_MESSAGE = (
    f"{INSUFFICIENT_CONTEXT_PHRASE}\n\n"
    "Nothing in the documents you have indexed is close enough to this question "
    "to answer from. Try rephrasing it, clearing the subject filter, or "
    "uploading the relevant reading."
)

DISCLAIMER_FOOTER = (
    "_Study aid, not legal advice - verify every citation against the primary "
    "source._"
)


SYSTEM_PROMPT = """\
You are a study assistant for a law student. You answer questions using ONLY \
the excerpts from the student's own course materials that are supplied with \
each question.

ABSOLUTE RULES - these override everything else:

1. Use ONLY the supplied excerpts. You have extensive legal knowledge from \
training; you must not use it here. If a case, statute, rule or definition is \
not in the excerpts, it does not exist for the purposes of your answer.

2. NEVER invent or guess a case name, citation, statute section, date, judge, \
court or holding. Reproduce such details only when they appear verbatim in an \
excerpt. An invented authority is far worse than an incomplete answer.

3. If the excerpts do not contain enough to answer, say exactly: \
"I couldn't find this in your materials." Then, briefly, say what the excerpts \
do cover that is adjacent, and suggest what the student might upload or search \
for instead. Do not pad the answer with general legal knowledge.

4. Cite every substantive claim with the bracketed number of the excerpt it \
came from, like [1] or [2][3]. Place the marker at the end of the sentence it \
supports. Every factual sentence needs one, including sentences that restate, \
compare or draw a conclusion from earlier cited sentences - a concluding \
sentence is still a claim about the materials and still needs its marker.

4a. Use plain ASCII square brackets: [1]. Never use full-width or CJK \
brackets such as the ones in "【1】", and never use parentheses or \
superscripts for citations.

5. Partial answers are expected and fine. Answer the part the excerpts cover, \
then state plainly which part they do not.

6. If two excerpts conflict, present both and attribute each - do not silently \
pick one.

7. Quote sparingly. When exact wording matters (a statutory test, a definition, \
the words of a holding), quote it in quotation marks and cite it. Otherwise \
explain in your own words.

STYLE:

- Write for a law student preparing for exams: precise, structured, no filler.
- Lead with the direct answer, then the supporting reasoning.
- Use short paragraphs or bullets. Bold the key legal terms and tests.
- Do not open with pleasantries or restate the question.
- Do not append your own disclaimer; the interface already shows one.
"""


# Some models (notably the gpt-oss family) emit full-width or CJK brackets for
# citation markers no matter what the prompt says. Normalising them keeps the
# rendered answer consistent and the markers matchable against the source list.
_CITATION_BRACKETS = {
    "【": "[",  # LEFT BLACK LENTICULAR BRACKET
    "】": "]",  # RIGHT BLACK LENTICULAR BRACKET
    "〔": "[",  # LEFT TORTOISE SHELL BRACKET
    "〕": "]",  # RIGHT TORTOISE SHELL BRACKET
    "［": "[",  # FULLWIDTH LEFT SQUARE BRACKET
    "］": "]",  # FULLWIDTH RIGHT SQUARE BRACKET
}

_CITATION_TRANSLATION = str.maketrans(_CITATION_BRACKETS)


def normalise_citation_markers(text: str) -> str:
    """Rewrite non-ASCII citation brackets as plain [n].

    Safe to apply to a streaming fragment: every mapped character is a single
    code point, so a chunk boundary cannot split one.
    """
    return text.translate(_CITATION_TRANSLATION) if text else text


def format_context(hits: Sequence[SearchHit]) -> str:
    """Render retrieved chunks as a numbered, citable context block.

    The numbering here is what the model cites as ``[n]``, and it matches the
    order the UI renders the sources in, so a reader can follow a marker back
    to the exact page or slide.
    """
    blocks: list[str] = []
    for position, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{position}] Source: {hit.citation}\n"
            f"{hit.text.strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def build_question_prompt(question: str, hits: Sequence[SearchHit]) -> str:
    """The user-turn prompt: excerpts first, then the question."""
    return (
        "Excerpts from the student's course materials:\n\n"
        "<excerpts>\n"
        f"{format_context(hits)}\n"
        "</excerpts>\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer using only the excerpts above, citing them as [n]. If they do "
        f'not contain the answer, say exactly "{INSUFFICIENT_CONTEXT_PHRASE}"'
    )


# --- Phase 5 study tasks ----------------------------------------------------

SUMMARISE_INSTRUCTION = """\
Summarise what these excerpts cover, for revision purposes.

Structure the summary as:
- **Topic** - one line on what this material is about.
- **Key principles** - the rules, tests and definitions stated, each cited [n].
- **Cases and authorities mentioned** - only those named in the excerpts, with \
what each is cited for. If none are named, say so.
- **Gaps** - anything the excerpts refer to but do not explain.

Invent nothing. Every line must trace to an excerpt."""

PRACTICE_QUESTIONS_INSTRUCTION = """\
Write 5 exam-style practice questions that are answerable from these excerpts \
alone.

Mix recall ("state the test for...") with application ("A contracts with B...; \
advise B"). After each question, on its own line, give:
- **Answer outline:** the key points a good answer covers, each cited [n].

Do not write a question whose answer is not in the excerpts."""

DEFINE_INSTRUCTION = """\
Define the following term strictly as the excerpts define or use it.

Give:
- **Definition** - as stated in the materials, quoted where the exact wording \
matters, cited [n].
- **In context** - how the materials apply or qualify it, cited [n].
- **Related terms** - only those the excerpts themselves connect to it.

If the excerpts never define the term, say exactly \
"I couldn't find this in your materials." and list the closest things they do \
cover."""


def build_task_prompt(instruction: str, subject_of: str, hits: Sequence[SearchHit]) -> str:
    """Prompt for a Phase 5 study task over a set of retrieved excerpts."""
    return (
        "Excerpts from the student's course materials:\n\n"
        "<excerpts>\n"
        f"{format_context(hits)}\n"
        "</excerpts>\n\n"
        f"Task target: {subject_of.strip()}\n\n"
        f"{instruction}"
    )
