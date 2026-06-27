"""Context assembly for domain-level signaling-question workers."""

from __future__ import annotations

import re
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from arbiter.config import EnvSettings
from arbiter.ingestion.paper import SECTION_KEYWORDS, TOP_LEVEL_SECTION_LABELS, normalize_heading
from arbiter.models import (
    DocumentSection,
    DomainContext,
    OutcomeComparison,
    SectionMap,
    SupplementSegment,
    TrialMetadata,
)
from arbiter.retrieval.supplement_index import SupplementIndex, TOKEN_PATTERN

DOMAIN_SECTIONS: dict[str, tuple[str, ...]] = {
    "D1": (
        "random",
        "allocation",
        "conceal",
        "baseline",
        "sequence",
        "methods",
        "statistical analysis",
    ),
    "D2": (
        "blinding",
        "masking",
        "open-label",
        "deviation",
        "adherence",
        "compliance",
        "intervention",
    ),
    "D3": (
        "missing",
        "withdraw",
        "lost to follow-up",
        "dropout",
        "censor",
        "analysed",
        "analyzed",
        "participant flow",
    ),
    "D4": (
        "outcome",
        "endpoint",
        "assessment",
        "assessor",
        "adjudication",
        "measurement",
    ),
    "D5": (
        "pre-specified",
        "prespecified",
        "protocol",
        "statistical analysis plan",
        "registry",
        "clinicaltrials.gov",
    ),
}

PREFIX_SECTION_PRIORITY = ("METHODS", "RESULTS")
MIN_PREFIX_SECTION_CHARS = 500
FLOW_TERMS = (
    "randomised",
    "randomized",
    "assessed for eligibility",
    "discontinued",
    "lost to follow-up",
    "analysed",
    "analyzed",
    "withdrew",
    "withdrawn",
    "withdrawal",
)
FLOW_PATTERN = re.compile(
    r"\b(randomi[sz]ed|assessed for eligibility|discontinued|lost to follow-up|analy[sz]ed|withdrew|withdrawn|withdrawal|N\s*=\s*\d+)\b",
    re.IGNORECASE,
)


def build_shared_prefix(
    *,
    trial_metadata: TrialMetadata | Mapping[str, Any] | None,
    section_map: SectionMap,
    ctgov_record: Mapping[str, Any] | None = None,
    settings: EnvSettings | None = None,
) -> tuple[str, str]:
    """Build the trial-static cacheable prefix once after ingestion."""

    active_settings = settings or EnvSettings()
    ct_gov_block = render_ct_gov_block(ctgov_record)
    parts = [
        _trial_metadata_block(trial_metadata),
        ct_gov_block,
        *_prefix_sections(section_map),
    ]
    prefix = "\n\n".join(part for part in parts if part.strip())
    return _cap_tokens(prefix, active_settings.prefix_token_budget), ct_gov_block


def context_assembly_node_factory(domain: str) -> Callable[[Mapping[str, Any]], dict[str, DomainContext]]:
    """Build a LangGraph-compatible node for one RoB 2 domain."""

    if domain not in DOMAIN_SECTIONS:
        raise ValueError(f"Unknown domain: {domain}")

    def context_assembly_node(state: Mapping[str, Any]) -> dict[str, DomainContext]:
        settings = _settings_from_state(state)
        section_map = _require_section_map(state)
        supplement_index = _supplement_index_from_state(state)
        ctgov_record = _ctgov_record_from_state(state)
        domain_text = build_domain_specific_text(domain, section_map, settings=settings)
        extra_blocks: list[str] = []

        if domain == "D3":
            flow_block = build_participant_flow_block(section_map, ctgov_record)
            if flow_block:
                extra_blocks.append(flow_block)

        if domain == "D5":
            comparison = _outcome_comparison_from_state(state)
            comparison_block = render_outcome_comparison_block(comparison)
            if comparison_block:
                extra_blocks.append(comparison_block)

        if extra_blocks:
            domain_text = "\n\n".join([domain_text, *extra_blocks]).strip()
            domain_text = _cap_tokens(domain_text, settings.domain_text_token_budget)

        query_terms = _domain_key_terms(domain)
        retrieval = supplement_index.retrieve_with_metadata(query_terms, domain, top_k=settings.retrieval_top_k)
        segments = cast(list[SupplementSegment], retrieval["segments"])
        top_score = cast(float | None, retrieval["top_score"])
        supplement_block = build_supplement_block(
            segments,
            query_terms=query_terms,
            settings=settings,
        )

        context = DomainContext(
            domain=domain,
            domain_specific_text=domain_text,
            supplement_block=supplement_block,
            retrieval_top_score=top_score,
            segments_retrieved=len(segments),
            segments_available=len(supplement_index.segments),
        )
        _record_context_trace(
            state=state,
            domain=domain,
            query_terms=query_terms,
            supplement_index=supplement_index,
            retrieval=retrieval,
            context=context,
        )
        return {"domain_context": context}

    return context_assembly_node


def build_domain_specific_text(
    domain: str,
    section_map: SectionMap,
    *,
    settings: EnvSettings | None = None,
) -> str:
    """Collect and bound the domain's dynamic main-paper suffix."""

    active_settings = settings or EnvSettings()
    sections = [
        section
        for section in section_map.sections
        if _matches_domain_section(section, domain) and not _is_shared_prefix_section(section)
    ]
    text = "\n\n".join(_format_section(section) for section in sections).strip()
    if len(text) < active_settings.domain_text_min_chars:
        abstract = _abstract_text(section_map)
        if abstract:
            text = "\n\n".join([_format_section(abstract), text]).strip()
    if not text and section_map.sections:
        text = _format_section(section_map.sections[0])
    return _cap_tokens(text, active_settings.domain_text_token_budget)


def build_supplement_block(
    segments: Sequence[SupplementSegment],
    *,
    query_terms: Sequence[str],
    settings: EnvSettings | None = None,
) -> str:
    active_settings = settings or EnvSettings()
    blocks: list[str] = []
    for segment in segments:
        if segment.char_count >= active_settings.large_segment_char_threshold:
            body = _subrank_sentences(segment.annotated_text, query_terms, active_settings.supplement_token_budget)
        else:
            body = segment.annotated_text
        blocks.append(
            "\n".join(
                [
                    f"[Supplement: {segment.source_file}; heading: {segment.heading}; pages: {', '.join(map(str, segment.pages))}]",
                    body,
                ]
            ).strip()
        )
    return _cap_tokens("\n\n".join(blocks), active_settings.supplement_token_budget)


def build_participant_flow_block(section_map: SectionMap, ctgov_record: Mapping[str, Any] | None = None) -> str:
    sentences = _participant_flow_sentences(section_map)
    enrollment_count = _ctgov_enrollment_count(ctgov_record)
    lines: list[str] = []
    if sentences:
        lines.append("[Participant flow text]")
        lines.extend(sentences)
    if enrollment_count is not None:
        lines.append(f"[ClinicalTrials.gov enrolment count hint] enrollmentInfo.count = {enrollment_count}")
    return "\n".join(lines)


def render_ct_gov_block(ctgov_record: Mapping[str, Any] | None) -> str:
    if ctgov_record is None:
        return ""

    protocol = _mapping(ctgov_record.get("protocolSection"))
    design = _mapping(_mapping(protocol.get("designModule")).get("designInfo"))
    outcomes = _mapping(protocol.get("outcomesModule"))
    arms = _mapping(protocol.get("armsInterventionsModule"))
    enrollment_count = _ctgov_enrollment_count(ctgov_record)

    lines = ["[ClinicalTrials.gov]"]
    allocation = design.get("allocation")
    masking = _mapping(design.get("maskingInfo")).get("masking")
    if allocation:
        lines.append(f"Allocation: {allocation}")
    if masking:
        lines.append(f"Masking: {masking}")
    if enrollment_count is not None:
        lines.append(f"Enrollment count: {enrollment_count}")
    outcome_lines = _render_registered_outcomes(outcomes, "primaryOutcomes", "Primary outcomes")
    outcome_lines.extend(_render_registered_outcomes(outcomes, "secondaryOutcomes", "Secondary outcomes"))
    lines.extend(outcome_lines)
    arm_groups = arms.get("armGroups")
    if isinstance(arm_groups, list) and arm_groups:
        names = [str(arm.get("label")) for arm in arm_groups if isinstance(arm, Mapping) and arm.get("label")]
        if names:
            lines.append(f"Arms: {'; '.join(names)}")
    return "\n".join(lines) if len(lines) > 1 else ""


def render_outcome_comparison_block(comparison: OutcomeComparison | None) -> str:
    if comparison is None or comparison.outcome_change_detected is None:
        return ""
    return "\n".join(
        [
            "[Registered outcome comparison]",
            f"Published outcome: {comparison.published_outcome or ''}",
            f"Best registered outcome: {comparison.registered_outcome or ''}",
            f"Similarity score: {comparison.outcome_similarity_score}",
            f"Outcome change detected: {comparison.outcome_change_detected}",
            f"Registered as primary: {comparison.registered_as_primary}",
        ]
    )


def _settings_from_state(state: Mapping[str, Any]) -> EnvSettings:
    settings = state.get("settings")
    if isinstance(settings, EnvSettings):
        return settings
    config = state.get("config")
    env = getattr(config, "env", None)
    if isinstance(env, EnvSettings):
        return env
    return EnvSettings()


def _require_section_map(state: Mapping[str, Any]) -> SectionMap:
    section_map = state.get("section_map")
    if not isinstance(section_map, SectionMap):
        raise TypeError("context assembly requires state['section_map'] as a SectionMap")
    return section_map


def _supplement_index_from_state(state: Mapping[str, Any]) -> SupplementIndex:
    supplement_index = state.get("supplement_index")
    if isinstance(supplement_index, SupplementIndex):
        return supplement_index
    runtime = state.get("runtime")
    runtime_index = getattr(runtime, "supplement_index", None)
    if isinstance(runtime_index, SupplementIndex):
        return runtime_index
    return SupplementIndex.empty()


def _ctgov_record_from_state(state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    record = state.get("ctgov_record", state.get("ct_gov_data"))
    return record if isinstance(record, Mapping) else None


def _outcome_comparison_from_state(state: Mapping[str, Any]) -> OutcomeComparison | None:
    comparison = state.get("outcome_comparison")
    if isinstance(comparison, OutcomeComparison):
        return comparison
    if isinstance(comparison, Mapping):
        return OutcomeComparison.model_validate(comparison)
    fields = {
        "registered_outcome": state.get("registered_outcome"),
        "published_outcome": state.get("published_outcome"),
        "outcome_similarity_score": state.get("outcome_similarity_score"),
        "outcome_change_detected": state.get("outcome_change_detected"),
        "registered_as_primary": state.get("registered_as_primary"),
    }
    if all(value is None for value in fields.values()):
        return None
    return OutcomeComparison.model_validate(fields)


def _trial_metadata_block(trial_metadata: TrialMetadata | Mapping[str, Any] | None) -> str:
    if trial_metadata is None:
        return ""
    data = trial_metadata.model_dump() if isinstance(trial_metadata, TrialMetadata) else dict(trial_metadata)
    lines = ["[Trial metadata]"]
    for key in ("trial_id", "title", "intervention", "comparator", "primary_outcome", "effect_of_interest", "blinding", "nct_number"):
        value = data.get(key)
        if value:
            lines.append(f"{key}: {value}")
    all_outcomes = data.get("all_outcomes")
    if isinstance(all_outcomes, list) and all_outcomes:
        lines.append(f"all_outcomes: {'; '.join(str(outcome) for outcome in all_outcomes)}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _prefix_sections(section_map: SectionMap) -> list[str]:
    sections: list[str] = []
    for label in PREFIX_SECTION_PRIORITY:
        matched = [section for section in section_map.sections if _label_matches(section.label, (label,))]
        text_length = sum(len(section.text.strip()) for section in matched)
        if matched and text_length >= MIN_PREFIX_SECTION_CHARS:
            sections.extend(_format_section(section) for section in matched)
            continue
        fallback = _slice_full_text_section(section_map, (label,))
        if fallback:
            display_label = matched[0].label if matched else label
            sections.append(f"[{display_label}]\n{fallback}".strip())
        else:
            sections.extend(_format_section(section) for section in matched)
    return sections


def _slice_full_text_section(section_map: SectionMap, labels: Sequence[str]) -> str:
    starts = [
        section
        for section in section_map.sections
        if _label_matches(section.label, labels) and 0 <= section.char_start < len(section_map.full_text)
    ]
    if not starts:
        return ""
    start_section = min(starts, key=lambda section: section.char_start)
    later_top_level = [
        section.char_start
        for section in section_map.sections
        if section.char_start > start_section.char_start
        and normalize_heading(section.label) in TOP_LEVEL_SECTION_LABELS
    ]
    end = min(later_top_level, default=len(section_map.full_text))
    return section_map.full_text[start_section.char_start : end].strip()


def _matches_domain_section(section: DocumentSection, domain: str) -> bool:
    if domain in section.domain_tags:
        return True
    return _label_matches(section.label, DOMAIN_SECTIONS[domain])


def _is_shared_prefix_section(section: DocumentSection) -> bool:
    return _label_matches(section.label, PREFIX_SECTION_PRIORITY)


def _label_matches(label: str, needles: Sequence[str]) -> bool:
    normalized = normalize_heading(label).lower()
    return any(needle.lower() in normalized for needle in needles)


def _abstract_text(section_map: SectionMap) -> DocumentSection | None:
    return next((section for section in section_map.sections if _label_matches(section.label, ("ABSTRACT",))), None)


def _format_section(section: DocumentSection) -> str:
    return f"[{section.label}]\n{section.text.strip()}".strip()


def _domain_key_terms(domain: str) -> list[str]:
    return sorted({*DOMAIN_SECTIONS[domain], *SECTION_KEYWORDS.get(domain, ())})


def _participant_flow_sentences(section_map: SectionMap) -> list[str]:
    sources: list[str] = []
    for section in section_map.sections:
        if _label_matches(section.label, ("RESULT", "RESULTS")):
            sources.append(section.text)
    sources.extend(box.text for box in section_map.page_boxes if FLOW_PATTERN.search(box.text))
    sentences: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for sentence in _sentences(source):
            if FLOW_PATTERN.search(sentence):
                normalized = " ".join(sentence.split())
                if normalized not in seen:
                    sentences.append(normalized)
                    seen.add(normalized)
    return sentences


def _sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence.strip()]


def _subrank_sentences(text: str, query_terms: Sequence[str], token_budget: int) -> str:
    sentences = _sentences(text)
    if not sentences:
        return ""
    query_tokens = _tokens(" ".join(query_terms))
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (-_sentence_score(item[1], query_tokens), item[0]),
    )
    selected: list[tuple[int, str]] = []
    current_tokens = 0
    for original_index, sentence in ranked:
        sentence_tokens = len(_tokens(sentence))
        if current_tokens and current_tokens + sentence_tokens > token_budget:
            continue
        selected.append((original_index, sentence))
        current_tokens += sentence_tokens
        if current_tokens >= token_budget:
            break
    return " ".join(sentence for _, sentence in sorted(selected))


def _sentence_score(sentence: str, query_tokens: list[str]) -> float:
    counts = Counter(_tokens(sentence))
    return float(sum(counts[token] for token in query_tokens))


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _cap_tokens(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    matches = list(TOKEN_PATTERN.finditer(text))
    if len(matches) <= token_budget:
        return text.strip()
    return text[: matches[token_budget - 1].end()].rstrip()


def _ctgov_enrollment_count(ctgov_record: Mapping[str, Any] | None) -> int | None:
    if ctgov_record is None:
        return None
    protocol = _mapping(ctgov_record.get("protocolSection"))
    design = _mapping(protocol.get("designModule"))
    enrollment = _mapping(design.get("enrollmentInfo"))
    count = enrollment.get("count")
    return count if isinstance(count, int) else None


def _render_registered_outcomes(outcomes: Mapping[str, Any], key: str, label: str) -> list[str]:
    values = outcomes.get(key)
    if not isinstance(values, list):
        return []
    rendered: list[str] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        measure = value.get("measure")
        time_frame = value.get("timeFrame")
        if measure and time_frame:
            rendered.append(f"{label}: {measure} ({time_frame})")
        elif measure:
            rendered.append(f"{label}: {measure}")
    return rendered


def _mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _record_context_trace(
    *,
    state: Mapping[str, Any],
    domain: str,
    query_terms: Sequence[str],
    supplement_index: SupplementIndex,
    retrieval: Mapping[str, Any],
    context: DomainContext,
) -> None:
    qa_trace = _qa_trace_from_state(state)
    if qa_trace is None:
        return
    scope = _trace_scope(state, domain)
    retrieval_ref = f"retrieval/{_trace_artifact_name(scope, domain)}.json"
    context_ref = f"context/{_trace_artifact_name(scope, domain)}.json"
    status = _supplement_status(len(supplement_index.segments), len(context.supplement_block.strip()))
    retrieval_payload = {
        "scope": scope,
        "request": {
            "domain": domain,
            "query_terms": list(query_terms),
            "query": " ".join(query_terms),
            "top_k": _settings_from_state(state).retrieval_top_k,
        },
        "supplement_status": status,
        "segments_available": len(supplement_index.segments),
        "segments_selected": context.segments_retrieved,
        "candidates": _segment_records(
            supplement_index,
            cast(list[int], retrieval.get("candidate_indices", [])),
            retrieval,
        ),
        "selected": _segment_records(
            supplement_index,
            cast(list[int], retrieval.get("selected_indices", [])),
            retrieval,
        ),
        "top_score": context.retrieval_top_score,
        "source_artifact_refs": _source_artifact_refs(state, supplement_index),
    }
    context_payload = {
        "scope": scope,
        "retrieval_artifact_ref": retrieval_ref,
        "supplement_status": status,
        "source_artifact_refs": _source_artifact_refs(state, supplement_index),
        "domain_context": context,
        "assembled_context": _assembled_context(state, context),
    }
    qa_trace.write_json_artifact(retrieval_ref, retrieval_payload)
    qa_trace.record_event(
        event_type="retrieval.completed",
        status="completed",
        trial_id=scope["trial_id"],
        outcome=scope["outcome"],
        domain=scope["domain"],
        sq_id=scope["sq_id"],
        artifact_refs=[retrieval_ref],
        payload={"supplement_status": status, "segments_selected": context.segments_retrieved},
    )
    qa_trace.write_json_artifact(context_ref, context_payload)
    qa_trace.record_event(
        event_type="context_assembly.completed",
        status="completed",
        trial_id=scope["trial_id"],
        outcome=scope["outcome"],
        domain=scope["domain"],
        sq_id=scope["sq_id"],
        artifact_refs=[context_ref],
        payload={"retrieval_artifact_ref": retrieval_ref, "supplement_status": status},
    )


def _qa_trace_from_state(state: Mapping[str, Any]) -> Any | None:
    trace = state.get("trace")
    if trace is not None and getattr(trace, "trace_level", None) == "full":
        return getattr(trace, "qa_trace", None)
    config = state.get("config")
    if config is not None and getattr(config, "trace_level", None) == "full":
        return getattr(config, "qa_trace", None)
    return None


def _trace_scope(state: Mapping[str, Any], domain: str) -> dict[str, str | None]:
    metadata = state.get("trial_metadata")
    trial_id = getattr(metadata, "trial_id", None)
    if trial_id is None and isinstance(metadata, Mapping):
        trial_id = metadata.get("trial_id")
    return {
        "trial_id": str(trial_id) if trial_id is not None else None,
        "outcome": str(state.get("outcome")) if state.get("outcome") is not None else None,
        "domain": domain,
        "sq_id": None,
    }


def _trace_artifact_name(scope: Mapping[str, str | None], domain: str) -> str:
    parts = [scope.get("trial_id") or "trial", scope.get("outcome") or "trial", domain, uuid.uuid4().hex[:8]]
    return "-".join(_safe_name(part) for part in parts)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "item"


def _supplement_status(available: int, selected_text_len: int) -> str:
    if available == 0:
        return "none_available"
    if selected_text_len == 0:
        return "none_selected"
    return "selected"


def _segment_records(
    supplement_index: SupplementIndex,
    indices: Sequence[int],
    retrieval: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bm25_scores = cast(Mapping[int, float], retrieval.get("bm25_scores", {}))
    dense_scores = cast(Mapping[int, float], retrieval.get("dense_scores", {}))
    rrf_scores = cast(Mapping[int, float], retrieval.get("rrf_scores", {}))
    records: list[dict[str, Any]] = []
    for rank, idx in enumerate(indices, start=1):
        segment = supplement_index.segments[idx]
        records.append(
            {
                "rank": rank,
                "segment_id": segment.segment_id,
                "text": segment.annotated_text,
                "source_ref": {
                    "source_file": segment.source_file,
                    "doc_type": segment.doc_type,
                    "heading": segment.heading,
                    "pages": segment.pages,
                },
                "scores": {
                    "bm25": bm25_scores.get(idx),
                    "dense": dense_scores.get(idx),
                    "rrf": rrf_scores.get(idx),
                },
                "fusion": {
                    "method": "reciprocal_rank_fusion",
                    "rank": rank,
                },
            }
        )
    return records


def _source_artifact_refs(state: Mapping[str, Any], supplement_index: SupplementIndex) -> list[str]:
    refs = state.get("source_artifact_refs")
    if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
        return [str(ref) for ref in refs]
    index_refs = getattr(supplement_index, "source_artifact_refs", None)
    if isinstance(index_refs, Sequence) and not isinstance(index_refs, (str, bytes)):
        return [str(ref) for ref in index_refs]
    return []


def _assembled_context(state: Mapping[str, Any], context: DomainContext) -> str:
    parts = [
        str(state.get("shared_prefix_text") or "").strip(),
        context.domain_specific_text.strip(),
        context.supplement_block.strip(),
    ]
    return "\n\n".join(part for part in parts if part)
