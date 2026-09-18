import json
from functools import lru_cache


def split_mids(value):
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache(maxsize=8)
def load_ancestor_map(ontology_path):
    with open(ontology_path, encoding="utf-8") as f:
        nodes = json.load(f)

    parents = {}
    for node in nodes:
        parent_id = node["id"]
        for child_id in node.get("child_ids", []):
            parents.setdefault(child_id, set()).add(parent_id)

    @lru_cache(maxsize=None)
    def ancestors(mid):
        result = set()
        for parent in parents.get(mid, set()):
            result.add(parent)
            result.update(ancestors(parent))
        return frozenset(result)

    return {node["id"]: ancestors(node["id"]) for node in nodes}


def has_label_conflict(mids_a, mids_b, ancestor_map):
    set_a = set(mids_a)
    set_b = set(mids_b)
    if not set_a or not set_b:
        return False
    if set_a & set_b:
        return True
    for mid_a in set_a:
        ancestors_a = ancestor_map.get(mid_a, frozenset())
        for mid_b in set_b:
            ancestors_b = ancestor_map.get(mid_b, frozenset())
            if mid_a in ancestors_b or mid_b in ancestors_a:
                return True
    return False
