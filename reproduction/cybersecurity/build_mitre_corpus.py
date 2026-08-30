import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = Path(
    r".\cyber_data\raw\enterprise-attack-19.1.json"
)

OUTPUT = Path(
    r".\cyber_data\processed\mitre_attack.jsonl"
)


# ATT&CK objects we want as entity documents.
ENTITY_TYPES = {
    "attack-pattern",
    "course-of-action",
    "intrusion-set",
    "malware",
    "tool",
    "campaign",
    "x-mitre-data-source",
    "x-mitre-data-component",
}


# Relationship types needed for our multi-hop benchmark.
RELATIONSHIP_TYPES = {
    "uses",
    "mitigates",
}


# ============================================================
# HELPERS
# ============================================================

def clean(value):
    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(
            str(item)
            for item in value
            if item is not None
        )

    return str(value)


def external_ids(obj):
    ids = []

    for ref in obj.get(
        "external_references",
        [],
    ):

        external_id = ref.get(
            "external_id"
        )

        if external_id:
            ids.append(
                external_id
            )

    return ids


def object_name(
    obj,
):
    if not obj:
        return ""

    return clean(
        obj.get("name")
    )


# ============================================================
# BUILD ENTITY DOCUMENT
# ============================================================

def build_entity_document(obj):

    object_id = clean(
        obj.get("id")
    )

    object_type = clean(
        obj.get("type")
    )

    name = clean(
        obj.get("name")
    )

    description = clean(
        obj.get("description")
    )

    aliases = obj.get(
        "aliases",
        [],
    )

    platforms = obj.get(
        "x_mitre_platforms",
        [],
    )

    permissions = obj.get(
        "x_mitre_permissions_required",
        [],
    )

    detection = clean(
        obj.get(
            "x_mitre_detection"
        )
    )

    ids = external_ids(
        obj
    )

    text_parts = [
        name,
        description,
        "Object type: " + object_type,
        "Aliases: " + clean(aliases),
        "External IDs: " + clean(ids),
        "Platforms: " + clean(platforms),
        "Permissions required: " + clean(permissions),
        "Detection: " + detection,
    ]

    paragraph_text = "\n".join(
        part
        for part in text_parts
        if part.strip()
    )

    return {
        "id": object_id,

        # UniRAG-compatible fields
        "title": name,
        "paragraph_text": paragraph_text,
        "url": "",
        "is_abstract": True,
        "paragraph_index": 0,
        "paragraph_type": "entity",

        # ATT&CK metadata
        "object_type": object_type,
        "name": name,
        "description": description,
        "aliases": aliases,
        "external_ids": ids,
        "platforms": platforms,
        "permissions_required": permissions,
        "detection": detection,
        "created": clean(
            obj.get("created")
        ),
        "modified": clean(
            obj.get("modified")
        ),
    }


# ============================================================
# BUILD RELATIONSHIP DOCUMENT
# ============================================================

def build_relationship_document(
    relationship,
    objects_by_id,
):

    relationship_id = clean(
        relationship.get("id")
    )

    relationship_type = clean(
        relationship.get(
            "relationship_type"
        )
    )

    source_id = clean(
        relationship.get(
            "source_ref"
        )
    )

    target_id = clean(
        relationship.get(
            "target_ref"
        )
    )

    source_obj = objects_by_id.get(
        source_id
    )

    target_obj = objects_by_id.get(
        target_id
    )

    source_name = object_name(
        source_obj
    )

    target_name = object_name(
        target_obj
    )

    if not source_name:
        source_name = source_id

    if not target_name:
        target_name = target_id

    # --------------------------------------------------------
    # Human-readable relationship statements.
    #
    # Include both directions in the text because retrieval
    # may query either entity or the relationship concept.
    # --------------------------------------------------------

    if relationship_type == "uses":

        title = (
            f"{source_name} uses "
            f"{target_name}"
        )

        paragraph_text = (
            f"{source_name} uses "
            f"{target_name}.\n"
            f"Source: {source_name}\n"
            f"Relationship: uses\n"
            f"Target: {target_name}\n"
            f"{target_name} is a technique or "
            f"capability used by {source_name}."
        )

    elif relationship_type == "mitigates":

        title = (
            f"{source_name} mitigates "
            f"{target_name}"
        )

        paragraph_text = (
            f"{source_name} mitigates "
            f"{target_name}.\n"
            f"Source: {source_name}\n"
            f"Relationship: mitigates\n"
            f"Target: {target_name}\n"
            f"{target_name} is mitigated by "
            f"{source_name}."
        )

    else:

        title = (
            f"{source_name} "
            f"{relationship_type} "
            f"{target_name}"
        )

        paragraph_text = (
            f"{source_name} "
            f"{relationship_type} "
            f"{target_name}.\n"
            f"Source: {source_name}\n"
            f"Relationship: {relationship_type}\n"
            f"Target: {target_name}"
        )

    return {
        "id": relationship_id,

        # UniRAG-compatible fields
        "title": title,
        "paragraph_text": paragraph_text,
        "url": "",
        "is_abstract": False,
        "paragraph_index": 0,
        "paragraph_type": "relationship",

        # Relationship metadata
        "object_type": "relationship",
        "relationship_type": relationship_type,
        "source_id": source_id,
        "source_name": source_name,
        "target_id": target_id,
        "target_name": target_name,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"MITRE STIX file not found: {INPUT}"
        )

    with INPUT.open(
        "r",
        encoding="utf-8",
    ) as file:

        bundle = json.load(
            file
        )

    objects = bundle.get(
        "objects",
        [],
    )

    print(
        "========== BUILDING MITRE CORPUS =========="
    )

    print(
        "Source STIX objects:",
        len(objects),
    )

    # --------------------------------------------------------
    # Build lookup table first.
    # --------------------------------------------------------

    objects_by_id = {}

    for obj in objects:

        object_id = obj.get(
            "id"
        )

        if object_id:
            objects_by_id[
                object_id
            ] = obj

    documents = []

    entity_counts = {}
    relationship_counts = {}

    # --------------------------------------------------------
    # ENTITY DOCUMENTS
    # --------------------------------------------------------

    for obj in objects:

        object_type = obj.get(
            "type"
        )

        if object_type not in ENTITY_TYPES:
            continue

        if obj.get(
            "revoked"
        ) is True:
            continue

        if obj.get(
            "x_mitre_deprecated"
        ) is True:
            continue

        document = build_entity_document(
            obj
        )

        documents.append(
            document
        )

        entity_counts[
            object_type
        ] = (
            entity_counts.get(
                object_type,
                0,
            )
            + 1
        )

    # --------------------------------------------------------
    # RELATIONSHIP DOCUMENTS
    # --------------------------------------------------------

    for obj in objects:

        if obj.get(
            "type"
        ) != "relationship":
            continue

        relationship_type = obj.get(
            "relationship_type"
        )

        if relationship_type not in (
            RELATIONSHIP_TYPES
        ):
            continue

        if obj.get(
            "revoked"
        ) is True:
            continue

        source_id = obj.get(
            "source_ref"
        )

        target_id = obj.get(
            "target_ref"
        )

        if not source_id or not target_id:
            continue

        # Only keep relationships where both endpoints
        # are present in the corpus.
        if (
            source_id not in objects_by_id
            or target_id not in objects_by_id
        ):
            continue

        document = build_relationship_document(
            obj,
            objects_by_id,
        )

        documents.append(
            document
        )

        relationship_counts[
            relationship_type
        ] = (
            relationship_counts.get(
                relationship_type,
                0,
            )
            + 1
        )

    # --------------------------------------------------------
    # WRITE JSONL
    # --------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        for document in documents:

            file.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    entity_total = sum(
        entity_counts.values()
    )

    relationship_total = sum(
        relationship_counts.values()
    )

    print()
    print(
        "========== MITRE CORPUS BUILT =========="
    )

    print(
        "Entity documents:",
        entity_total,
    )

    print(
        "Relationship documents:",
        relationship_total,
    )

    print(
        "Total documents:",
        len(documents),
    )

    print()
    print(
        "Entity counts:"
    )

    for key in sorted(
        entity_counts
    ):

        print(
            f"  {key}: "
            f"{entity_counts[key]}"
        )

    print()
    print(
        "Relationship counts:"
    )

    for key in sorted(
        relationship_counts
    ):

        print(
            f"  {key}: "
            f"{relationship_counts[key]}"
        )

    print()
    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()