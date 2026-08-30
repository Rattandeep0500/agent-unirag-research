import json
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = Path(
    r".\cyber_data\raw\enterprise-attack-19.1.json"
)

OUTPUT = Path(
    r".\cyber_data\processed\cyber_benchmark.json"
)

NUM_QUESTIONS = 10


# ============================================================
# LOAD STIX DATA
# ============================================================

def load_bundle():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"MITRE STIX file not found: {INPUT}"
        )

    with INPUT.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# OBJECT MAPS
# ============================================================

def build_object_maps(objects):

    objects_by_id = {}
    names_by_id = {}

    for obj in objects:

        object_id = obj.get("id")

        if not object_id:
            continue

        objects_by_id[object_id] = obj

        name = obj.get("name")

        if name:
            names_by_id[object_id] = name

    return objects_by_id, names_by_id


# ============================================================
# ATT&CK RELATIONSHIPS
# ============================================================

def build_relationships(objects):

    source_to_techniques = {}
    technique_to_mitigations = {}

    for obj in objects:

        if obj.get("type") != "relationship":
            continue

        relationship_type = obj.get(
            "relationship_type"
        )

        source = obj.get(
            "source_ref"
        )

        target = obj.get(
            "target_ref"
        )

        if not source or not target:
            continue

        # -----------------------------------------------------
        # Threat group / malware / tool -> technique
        # -----------------------------------------------------

        if relationship_type == "uses":

            valid_source = (
                source.startswith("intrusion-set--")
                or source.startswith("malware--")
                or source.startswith("tool--")
            )

            if (
                valid_source
                and target.startswith("attack-pattern--")
            ):

                source_to_techniques.setdefault(
                    source,
                    set(),
                ).add(target)

        # -----------------------------------------------------
        # Mitigation -> technique
        #
        # ATT&CK stores:
        #
        # course-of-action --mitigates-->
        # attack-pattern
        #
        # Convert to:
        #
        # attack-pattern -> course-of-action
        # -----------------------------------------------------

        elif relationship_type == "mitigates":

            if (
                source.startswith("course-of-action--")
                and target.startswith("attack-pattern--")
            ):

                technique_to_mitigations.setdefault(
                    target,
                    set(),
                ).add(source)

    return (
        source_to_techniques,
        technique_to_mitigations,
    )


# ============================================================
# CLEAN TECHNIQUE DESCRIPTION
# ============================================================

def clean_description(
    description,
    technique_name,
):
    """
    Turn the ATT&CK technique description into a retrieval clue
    while removing the exact technique name so the benchmark
    does not directly reveal the intermediate answer.
    """

    if not description:
        return ""

    text = str(description)

    # Remove markdown links while preserving visible text.
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Remove the exact technique name.
    if technique_name:
        text = re.sub(
            re.escape(technique_name),
            "",
            text,
            flags=re.IGNORECASE,
        )

    # Normalize whitespace.
    text = " ".join(
        text.split()
    ).strip()

    # Remove leading punctuation left by the name removal.
    text = text.strip(
        " :-–—,.;"
    )

    # Avoid very long questions.
    if len(text) > 500:
        text = text[:500].rsplit(
            " ",
            1,
        )[0]

    return text


# ============================================================
# BUILD CANDIDATES
# ============================================================

def build_candidates(
    objects,
    objects_by_id,
    names_by_id,
):

    (
        source_to_techniques,
        technique_to_mitigations,
    ) = build_relationships(
        objects
    )

    candidates = []

    for source_id in sorted(
        source_to_techniques
    ):

        source_obj = objects_by_id.get(
            source_id
        )

        if not source_obj:
            continue

        source_name = names_by_id.get(
            source_id
        )

        if not source_name:
            continue

        if source_id.startswith(
            "intrusion-set--"
        ):
            source_type = "threat group"

        elif source_id.startswith(
            "malware--"
        ):
            source_type = "malware"

        elif source_id.startswith(
            "tool--"
        ):
            source_type = "tool"

        else:
            continue

        for technique_id in sorted(
            source_to_techniques[source_id]
        ):

            technique_obj = objects_by_id.get(
                technique_id
            )

            if not technique_obj:
                continue

            if technique_obj.get("type") != (
                "attack-pattern"
            ):
                continue

            technique_name = names_by_id.get(
                technique_id
            )

            if not technique_name:
                continue

            description = clean_description(
                technique_obj.get(
                    "description",
                    "",
                ),
                technique_name,
            )

            if len(description.split()) < 8:
                continue

            mitigation_ids = (
                technique_to_mitigations.get(
                    technique_id,
                    set(),
                )
            )

            if not mitigation_ids:
                continue

            for mitigation_id in sorted(
                mitigation_ids
            ):

                mitigation_name = names_by_id.get(
                    mitigation_id
                )

                if not mitigation_name:
                    continue

                candidates.append(
                    {
                        "source_type": source_type,
                        "source_id": source_id,
                        "source_name": source_name,
                        "technique_id": technique_id,
                        "technique_name": technique_name,
                        "technique_description": description,
                        "mitigation_id": mitigation_id,
                        "mitigation_name": mitigation_name,
                    }
                )

    return candidates


# ============================================================
# SELECT UNIQUE AND DIVERSE CHAINS
# ============================================================

def select_candidates(
    candidates,
    count,
):

    # --------------------------------------------------------
    # First reduce to unique source -> technique pairs.
    #
    # This prevents the previous problem where one pair
    # generated multiple nearly identical questions.
    # --------------------------------------------------------

    unique_pairs = {}

    for candidate in candidates:

        key = (
            candidate["source_id"],
            candidate["technique_id"],
        )

        if key not in unique_pairs:
            unique_pairs[key] = candidate

    candidates = list(
        unique_pairs.values()
    )

    # --------------------------------------------------------
    # Deterministic ordering.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: (
            item["source_type"],
            item["source_name"].lower(),
            item["technique_name"].lower(),
        )
    )

    selected = []

    used_sources = set()
    used_techniques = set()
    used_pairs = set()
    used_questions = set()

    # --------------------------------------------------------
    # Pass 1:
    # Prefer different source entities and techniques.
    # --------------------------------------------------------

    for candidate in candidates:

        pair = (
            candidate["source_id"],
            candidate["technique_id"],
        )

        if pair in used_pairs:
            continue

        if candidate["source_id"] in used_sources:
            continue

        if candidate["technique_id"] in used_techniques:
            continue

        question = make_question(
            candidate
        )

        if question in used_questions:
            continue

        selected.append(
            candidate
        )

        used_sources.add(
            candidate["source_id"]
        )

        used_techniques.add(
            candidate["technique_id"]
        )

        used_pairs.add(
            pair
        )

        used_questions.add(
            question
        )

        if len(selected) >= count:
            return selected

    # --------------------------------------------------------
    # Pass 2:
    # Allow repeated source entities but keep unique
    # source -> technique pairs.
    # --------------------------------------------------------

    for candidate in candidates:

        pair = (
            candidate["source_id"],
            candidate["technique_id"],
        )

        if pair in used_pairs:
            continue

        question = make_question(
            candidate
        )

        if question in used_questions:
            continue

        selected.append(
            candidate
        )

        used_pairs.add(
            pair
        )

        used_questions.add(
            question
        )

        if len(selected) >= count:
            return selected

    return selected


# ============================================================
# QUESTION CONSTRUCTION
# ============================================================

def make_question(candidate):

    source_type = candidate[
        "source_type"
    ]

    source_name = candidate[
        "source_name"
    ]

    activity = candidate[
        "technique_description"
    ]

    question = (
        f"Which mitigation applies to the "
        f"attack technique used by "
        f"{source_type} {source_name} "
        f"when {activity}?"
    )

    return question


# ============================================================
# BUILD FINAL BENCHMARK RECORDS
# ============================================================

def build_questions(
    selected,
):

    questions = []

    for index, candidate in enumerate(
        selected,
        start=1,
    ):

        questions.append(
            {
                "question_id": (
                    f"cyber_{index:03d}"
                ),

                "question": make_question(
                    candidate
                ),

                "hop_count": 3,

                "chain": [
                    {
                        "type": candidate[
                            "source_type"
                        ],
                        "id": candidate[
                            "source_id"
                        ],
                        "name": candidate[
                            "source_name"
                        ],
                    },

                    {
                        "type": "attack-pattern",
                        "id": candidate[
                            "technique_id"
                        ],
                        "name": candidate[
                            "technique_name"
                        ],
                    },

                    {
                        "type": "course-of-action",
                        "id": candidate[
                            "mitigation_id"
                        ],
                        "name": candidate[
                            "mitigation_name"
                        ],
                    },
                ],
            }
        )

    return questions


# ============================================================
# VALIDATION
# ============================================================

def validate_questions(
    questions,
):

    if len(questions) != NUM_QUESTIONS:

        raise RuntimeError(
            f"Expected {NUM_QUESTIONS} questions, "
            f"got {len(questions)}."
        )

    question_ids = set()
    question_texts = set()
    source_technique_pairs = set()
    technique_names = set()

    for question in questions:

        question_id = question[
            "question_id"
        ]

        question_text = question[
            "question"
        ]

        chain = question[
            "chain"
        ]

        # ----------------------------------------------------
        # Unique IDs
        # ----------------------------------------------------

        if question_id in question_ids:

            raise RuntimeError(
                f"Duplicate question ID: "
                f"{question_id}"
            )

        # ----------------------------------------------------
        # Unique question text
        # ----------------------------------------------------

        if question_text in question_texts:

            raise RuntimeError(
                f"Duplicate question text: "
                f"{question_text}"
            )

        # ----------------------------------------------------
        # Exactly 3 hops
        # ----------------------------------------------------

        if len(chain) != 3:

            raise RuntimeError(
                f"Invalid chain length for "
                f"{question_id}"
            )

        # ----------------------------------------------------
        # Valid object types
        # ----------------------------------------------------

        if chain[0]["type"] not in {
            "threat group",
            "malware",
            "tool",
        }:

            raise RuntimeError(
                f"Invalid source type for "
                f"{question_id}"
            )

        if chain[1]["type"] != (
            "attack-pattern"
        ):

            raise RuntimeError(
                f"Invalid technique type for "
                f"{question_id}"
            )

        if chain[2]["type"] != (
            "course-of-action"
        ):

            raise RuntimeError(
                f"Invalid mitigation type for "
                f"{question_id}"
            )

        # ----------------------------------------------------
        # Unique source -> technique pair
        # ----------------------------------------------------

        pair = (
            chain[0]["id"],
            chain[1]["id"],
        )

        if pair in source_technique_pairs:

            raise RuntimeError(
                "Duplicate source -> technique "
                f"pair: {pair}"
            )

        # ----------------------------------------------------
        # Make sure the technique name is NOT leaked
        # into the question text.
        # ----------------------------------------------------

        technique_name = chain[1][
            "name"
        ]

        if technique_name.lower() in (
            question_text.lower()
        ):

            raise RuntimeError(
                f"Technique name leaked into "
                f"question {question_id}: "
                f"{technique_name}"
            )

        question_ids.add(
            question_id
        )

        question_texts.add(
            question_text
        )

        source_technique_pairs.add(
            pair
        )

        technique_names.add(
            technique_name
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "========== BUILDING CYBER BENCHMARK =========="
    )

    bundle = load_bundle()

    objects = bundle.get(
        "objects",
        [],
    )

    print(
        "STIX objects:",
        len(objects),
    )

    (
        objects_by_id,
        names_by_id,
    ) = build_object_maps(
        objects
    )

    candidates = build_candidates(
        objects,
        objects_by_id,
        names_by_id,
    )

    print(
        "Candidate chains:",
        len(candidates),
    )

    selected = select_candidates(
        candidates,
        NUM_QUESTIONS,
    )

    print(
        "Selected chains:",
        len(selected),
    )

    questions = build_questions(
        selected
    )

    validate_questions(
        questions
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            questions,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "========== CYBER BENCHMARK BUILT =========="
    )

    print(
        "Questions:",
        len(questions),
    )

    print(
        "Validation: PASS"
    )

    print()
    print(
        "Questions:"
    )

    for question in questions:

        print()
        print(
            question["question_id"]
        )

        print(
            question["question"]
        )

        print(
            "Gold chain:"
        )

        for hop in question["chain"]:

            print(
                f"  {hop['type']}: "
                f"{hop['name']}"
            )

    print()
    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()