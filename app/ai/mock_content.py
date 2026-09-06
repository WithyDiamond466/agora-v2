"""Shared deterministic feedback used by local mock grading."""

MOCK_CRITERION_COMMENTS = {
    "thesis": {
        "strong": "The thesis is stated once, sharply, and every section is visibly working for it.",
        "solid": "A defensible thesis, though it goes quiet in the middle and has to be reconstructed.",
        "developing": "There is a position here, but it arrives as a summary rather than a claim to defend.",
        "weak": "No thesis I can argue with — the section restates the prompt instead of answering it.",
    },
    "evidence": {
        "strong": "The readings are doing real work: you quote the passage that actually carries the point.",
        "solid": "The sources are relevant, but two of them are cited rather than used.",
        "developing": "The idea is there but under-argued; show the reader the reasoning steps.",
        "weak": "Almost nothing from the course materials reaches the argument.",
    },
    "counterargument": {
        "strong": "You take on the strongest version of the opposing view and answer it on its own terms.",
        "solid": "The objection is named and met, though the version you answer is the easier one.",
        "developing": "An objection appears late and is dismissed rather than engaged.",
        "weak": "The opposing view never appears, so the conclusion is never tested.",
    },
    "mechanics": {
        "strong": "Clean, readable prose; the structure alone tells me where the argument is going.",
        "solid": "Readable throughout, with a few paragraphs that carry more than one idea each.",
        "developing": "The sentences are doing too much at once and the paragraph breaks fall in odd places.",
        "weak": "Structure and citation are inconsistent enough to obscure the argument underneath.",
    },
}
