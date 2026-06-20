from __future__ import annotations

from arbiter.config import EnvSettings
from arbiter.graph.nodes.context_assembly import (
    build_shared_prefix,
    context_assembly_node_factory,
)
from arbiter.models import (
    BlindingStatus,
    DocumentSection,
    DomainContext,
    DocType,
    EffectOfInterest,
    PageBox,
    SectionMap,
    SupplementSegment,
    TrialMetadata,
)
from arbiter.retrieval.supplement_index import SupplementIndex


def _section(label: str, text: str, tags: list[str] | None = None, start: int = 0) -> DocumentSection:
    return DocumentSection(
        label=label,
        pages=[0],
        char_start=start,
        char_end=start + len(text),
        text=text,
        domain_tags=tags or [],
    )


def _section_map() -> SectionMap:
    return SectionMap(
        source_path="paper.pdf",
        full_text="",
        sections=[
            _section("ABSTRACT", "Abstract says the trial was randomised and assessed survival."),
            _section("METHODS", "Central randomisation and allocation concealment were used.", ["D1"]),
            _section("RESULTS", "The long narrative omits the flow sentence from the prefix tail.", ["D3"]),
            _section(
                "Participant Flow",
                "Overall, 120 patients were randomised. Six patients withdrew and 114 were analysed.",
                ["D3"],
            ),
            _section("Outcome Assessment", "An independent committee assessed the endpoint.", ["D4"]),
        ],
        page_boxes=[
            PageBox(
                boxclass="text",
                text="CONSORT caption: 180 assessed for eligibility; 120 randomised.",
                bbox=(0, 0, 10, 10),
                page=2,
            )
        ],
    )


def _ctgov_record() -> dict:
    return {
        "protocolSection": {
            "designModule": {
                "designInfo": {
                    "allocation": "Randomized",
                    "maskingInfo": {"masking": "Double"},
                },
                "enrollmentInfo": {"count": 123},
            },
            "outcomesModule": {
                "primaryOutcomes": [{"measure": "Overall survival", "timeFrame": "36 months"}],
                "secondaryOutcomes": [{"measure": "Progression-free survival"}],
            },
            "armsInterventionsModule": {
                "armGroups": [{"label": "Intervention"}, {"label": "Control"}],
            },
        }
    }


def test_build_shared_prefix_includes_metadata_ctgov_methods_and_results_with_cap() -> None:
    settings = EnvSettings()
    settings.prefix_token_budget = 40
    metadata = TrialMetadata(
        trial_id="T1",
        title="Trial title",
        intervention="Drug A",
        comparator="Placebo",
        primary_outcome="Overall survival",
        all_outcomes=["Overall survival"],
        effect_of_interest=EffectOfInterest.ASSIGNMENT,
        blinding=BlindingStatus.DOUBLE_BLIND,
        nct_number="NCT12345678",
    )

    prefix, ctgov_block = build_shared_prefix(
        trial_metadata=metadata,
        section_map=_section_map(),
        ctgov_record=_ctgov_record(),
        settings=settings,
    )

    assert prefix
    assert "Trial title" in prefix
    assert "Enrollment count: 123" in prefix
    assert "Primary outcomes: Overall survival" in prefix
    assert "METHODS" not in prefix
    assert ctgov_block.startswith("[ClinicalTrials.gov]")
    assert len(prefix.split()) <= 45


def test_context_assembly_returns_domain_context_and_reads_existing_prefix() -> None:
    state = {
        "shared_prefix_text": "already-built-prefix",
        "section_map": _section_map(),
        "supplement_index": SupplementIndex.empty(),
    }

    result = context_assembly_node_factory("D4")(state)

    assert isinstance(result["domain_context"], DomainContext)
    assert result["domain_context"].domain == "D4"
    assert "Outcome Assessment" in result["domain_context"].domain_specific_text
    assert "already-built-prefix" not in result["domain_context"].domain_specific_text


def test_d3_context_injects_flow_sentences_and_ctgov_enrollment_hint() -> None:
    settings = EnvSettings()
    settings.prefix_token_budget = 5
    prefix, _ = build_shared_prefix(
        trial_metadata=None,
        section_map=_section_map(),
        ctgov_record=_ctgov_record(),
        settings=settings,
    )
    assert "120 randomised" not in prefix

    result = context_assembly_node_factory("D3")(
        {
            "section_map": _section_map(),
            "ctgov_record": _ctgov_record(),
            "supplement_index": SupplementIndex.empty(),
        }
    )

    text = result["domain_context"].domain_specific_text
    assert "120 patients were randomised" in text
    assert "114 were analysed" in text
    assert "enrollmentInfo.count = 123" in text


def test_d5_context_includes_outcome_comparison_block_when_present() -> None:
    result = context_assembly_node_factory("D5")(
        {
            "section_map": _section_map(),
            "supplement_index": SupplementIndex.empty(),
            "registered_outcome": "Overall survival",
            "published_outcome": "Time to treatment failure",
            "outcome_similarity_score": 0.42,
            "outcome_change_detected": True,
            "registered_as_primary": True,
        }
    )

    text = result["domain_context"].domain_specific_text
    assert "[Registered outcome comparison]" in text
    assert "Outcome change detected: True" in text


def test_supplement_block_reranks_large_segments_and_respects_budget() -> None:
    settings = EnvSettings()
    settings.supplement_token_budget = 20
    settings.large_segment_char_threshold = 10
    settings.retrieval_top_k = 1
    segment = SupplementSegment(
        segment_id="s1",
        source_file="sap.pdf",
        doc_type=DocType.SAP,
        heading="Missing data",
        pages=[1],
        raw_text=(
            "This irrelevant sentence is very long and should not dominate. "
            "Missing data were handled with multiple imputation. "
            "Another unrelated sentence follows."
        ),
        annotation="Supplement describes missing outcome data handling.",
        domain_tags=["D3"],
        char_count=200,
    )
    index = SupplementIndex([segment], settings=settings)

    result = context_assembly_node_factory("D3")(
        {
            "settings": settings,
            "section_map": _section_map(),
            "supplement_index": index,
        }
    )

    context = result["domain_context"]
    assert context.segments_available == 1
    assert context.segments_retrieved == 1
    assert context.retrieval_top_score is not None
    assert "Missing data were handled" in context.supplement_block
    assert len(context.supplement_block.split()) <= 30


def test_domain_specific_text_respects_independent_token_budget() -> None:
    settings = EnvSettings()
    settings.domain_text_min_chars = 0
    settings.domain_text_token_budget = 8
    section_map = SectionMap(
        source_path="paper.pdf",
        full_text="",
        sections=[
            _section(
                "Missing Data",
                "Missing data " + " ".join(f"token{i}" for i in range(100)),
                ["D3"],
            )
        ],
        page_boxes=[],
    )

    result = context_assembly_node_factory("D3")(
        {
            "settings": settings,
            "section_map": section_map,
            "supplement_index": SupplementIndex.empty(),
        }
    )

    assert len(result["domain_context"].domain_specific_text.split()) <= 10
