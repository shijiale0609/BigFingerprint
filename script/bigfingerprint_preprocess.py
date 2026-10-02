import itertools
import re
from dataclasses import dataclass

from .rdkit_repeat_orientation import (
    DESCRIPTOR_RE,
    orient_bigsmiles_repeat_unit_fragments,
)


@dataclass(frozen=True)
class HEndGroupSite:
    side: str
    object_text: str
    h_span: tuple
    candidates: tuple


def preprocess_bigsmiles(bigsmiles):
    """Expand missing or explicit ``[H]`` polymer end groups.

    Parameters
    ----------
    bigsmiles : str
        BigSMILES representation to preprocess.

    Returns
    -------
    list[str]
        Expanded BigSMILES candidates. Independent end-group sites produce a
        Cartesian product; an input requiring no expansion returns as a
        one-element list.
    """
    with_h = add_missing_h_endgroups(bigsmiles)
    return expand_h_endgroups(with_h)


def needs_h_endgroup_completion(bigsmiles):
    """Check whether a stochastic object needs an explicit ``[H]`` end group.

    Parameters
    ----------
    bigsmiles : str
        BigSMILES representation to inspect.

    Returns
    -------
    bool
        ``True`` when at least one open terminal should receive ``[H]``.
    """
    return add_missing_h_endgroups(bigsmiles) != bigsmiles


def add_missing_h_endgroups(bigsmiles):
    """Add ``[H]`` to stochastic objects at open string or branch ends.

    Parameters
    ----------
    bigsmiles : str
        BigSMILES representation to complete.

    Returns
    -------
    str
        BigSMILES with missing terminal ``[H]`` markers inserted.

    Examples:
        "{[>][<]CCO[>][<]}CC" -> "[H]{[>][<]CCO[>][<]}CC"
        "CC{[>][<]CCO[>][<]}" -> "CC{[>][<]CCO[>][<]}[H]"
        "CC({[>][<]CCO[>][<]})CC" -> "CC({[>][<]CCO[>][<]}[H])CC"
    """
    objects = _find_objects(bigsmiles)
    insertions = []

    for start, end, object_text in objects:
        left_terminal, right_terminal = _object_terminals(object_text)

        if left_terminal != "[]" and _needs_left_h(bigsmiles, start):
            insertions.append((start, "[H]"))

        if right_terminal != "[]" and _needs_right_h(bigsmiles, end):
            insertions.append((end, "[H]"))

    if not insertions:
        return bigsmiles

    pieces = []
    cursor = 0
    for position, text in sorted(insertions, key=lambda item: item[0]):
        pieces.append(bigsmiles[cursor:position])
        pieces.append(text)
        cursor = position
    pieces.append(bigsmiles[cursor:])
    return "".join(pieces)


def expand_h_endgroups(bigsmiles):
    """Replace explicit ``[H]`` end groups with compatible repeat fragments.

    Parameters
    ----------
    bigsmiles : str
        BigSMILES containing zero or more explicit ``[H]`` end groups.

    Returns
    -------
    list[str]
        All Cartesian-product candidates across independent end-group sites.
    """
    sites = _find_h_endgroup_sites(bigsmiles)
    if not sites:
        return [bigsmiles]

    candidate_lists = [site.candidates for site in sites]
    expanded = []
    for replacement_combo in itertools.product(*candidate_lists):
        expanded.append(_replace_spans(bigsmiles, sites, replacement_combo))

    return _unique_preserve_order(expanded)


def _find_h_endgroup_sites(bigsmiles):
    sites = []
    for start, end, object_text in _find_objects(bigsmiles):
        left_terminal, right_terminal = _object_terminals(object_text)
        repeats = _repeat_units(object_text)

        if start >= 3 and bigsmiles[start - 3 : start] == "[H]":
            candidates = _candidate_fragments(
                repeats=repeats,
                terminal_descriptor=left_terminal,
                h_side="left",
            )
            if candidates:
                sites.append(
                    HEndGroupSite(
                        side="left",
                        object_text=object_text,
                        h_span=(start - 3, start),
                        candidates=tuple(candidates),
                    )
                )

        if bigsmiles[end : end + 3] == "[H]":
            candidates = _candidate_fragments(
                repeats=repeats,
                terminal_descriptor=right_terminal,
                h_side="right",
            )
            if candidates:
                sites.append(
                    HEndGroupSite(
                        side="right",
                        object_text=object_text,
                        h_span=(end, end + 3),
                        candidates=tuple(candidates),
                    )
                )

    return sorted(sites, key=lambda site: site.h_span[0])


def _candidate_fragments(repeats, terminal_descriptor, h_side):
    candidates = []

    for repeat in repeats:
        left_desc, right_desc = _repeat_terminals(repeat)

        if "$" in terminal_descriptor:
            fragments = orient_bigsmiles_repeat_unit_fragments(
                repeat,
                insert_side=h_side,
                descriptor=terminal_descriptor,
            )
            candidates.extend(fragment.smiles for fragment in fragments)
            continue

        if h_side == "left" and right_desc == terminal_descriptor:
            candidates.append(_strip_outer_repeat_descriptors(repeat))

        if h_side == "right" and left_desc == terminal_descriptor:
            candidates.append(_strip_outer_repeat_descriptors(repeat))

    return _unique_preserve_order(candidates)


def _find_objects(bigsmiles):
    objects = []
    stack = []

    for index, char in enumerate(bigsmiles):
        if char == "{":
            stack.append(index)
        elif char == "}":
            if not stack:
                raise ValueError(
                    f"Unmatched closing brace at index {index}: {bigsmiles}"
                )
            start = stack.pop()
            if not stack:
                objects.append((start, index + 1, bigsmiles[start : index + 1]))

    if stack:
        raise ValueError(f"Unmatched opening brace at index {stack[-1]}: {bigsmiles}")

    return objects


def _object_terminals(object_text):
    # Only the tokens immediately inside the outer braces are object terminals.
    # A closed [] terminal is intentionally not part of DESCRIPTOR_RE, so a
    # whole-object search would mistake a repeat descriptor for an open end.
    inner = object_text[1:-1].strip()
    terminal_pattern = rf"(?:\[\]|{DESCRIPTOR_RE.pattern})"
    left_match = re.match(terminal_pattern, inner)
    right_match = re.search(rf"({terminal_pattern})\s*$", inner)

    left_terminal = left_match.group(0) if left_match else "[]"
    right_terminal = right_match.group(1) if right_match else "[]"
    return left_terminal, right_terminal


def _repeat_units(object_text):
    inner = object_text[object_text.find("]") + 1 :]
    inner = inner[: inner.rfind("[")]
    repeat_side, _implicit_side = _split_implicit_ends(inner)
    return [
        repeat.strip()
        for repeat in _split_top_level(repeat_side, ",")
        if repeat.strip()
    ]


def _repeat_terminals(repeat_unit):
    descriptors = list(DESCRIPTOR_RE.finditer(repeat_unit))
    if len(descriptors) < 2:
        raise ValueError(
            f"Repeat unit must contain at least two descriptors: {repeat_unit}"
        )
    return descriptors[0].group(0), descriptors[-1].group(0)


def _strip_outer_repeat_descriptors(repeat_unit):
    """Remove terminal descriptors without rewriting nested BigSMILES text."""
    descriptors = list(DESCRIPTOR_RE.finditer(repeat_unit))
    if len(descriptors) < 2:
        raise ValueError(
            f"Repeat unit must contain at least two descriptors: {repeat_unit}"
        )
    return repeat_unit[descriptors[0].end() : descriptors[-1].start()]


def _split_implicit_ends(text):
    parts = _split_top_level(text, ";")
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], ";".join(parts[1:])


def _split_top_level(text, delimiter):
    parts = []
    token = []
    brace_depth = 0
    bracket_depth = 0

    for char in text:
        if char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth > 0:
            bracket_depth -= 1
        elif char == "{" and bracket_depth == 0:
            brace_depth += 1
        elif char == "}" and bracket_depth == 0:
            brace_depth -= 1

        if char == delimiter and brace_depth == 0 and bracket_depth == 0:
            parts.append("".join(token))
            token = []
        else:
            token.append(char)

    parts.append("".join(token))
    return parts


def _needs_left_h(bigsmiles, object_start):
    if object_start >= 3 and bigsmiles[object_start - 3 : object_start] == "[H]":
        return False
    return object_start == 0 or bigsmiles[object_start - 1] == "("


def _needs_right_h(bigsmiles, object_end):
    if bigsmiles[object_end : object_end + 3] == "[H]":
        return False
    return object_end == len(bigsmiles) or bigsmiles[object_end : object_end + 1] == ")"


def _replace_spans(bigsmiles, sites, replacements):
    pieces = []
    cursor = 0

    for site, replacement in zip(sites, replacements):
        start, end = site.h_span
        pieces.append(bigsmiles[cursor:start])
        pieces.append(replacement)
        cursor = end

    pieces.append(bigsmiles[cursor:])
    return "".join(pieces)


def _unique_preserve_order(values):
    unique = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


if __name__ == "__main__":
    examples = [
        "OCCO{[>][<]CCO[>], [<]CC(C)O[>][<]}[H]",
        "[H]{[>][<]CCO[>], [<]CC(C)O[>][<]}CC",
        "{[>][<]CCO[>], [<]CC(C)O[>][<]}CC",
        "[H]{[$][$]CC(C)[$][$]}[H]",
        "[H]{[$][$]CC(C)[$],[$]CC(CC)[$][$]}[H]",
    ]
    for example in examples:
        print("INPUT:", example)
        for candidate in preprocess_bigsmiles(example):
            print("  ", candidate)
        print()
