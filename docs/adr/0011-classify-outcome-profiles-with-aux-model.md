# Classify outcome profiles with the aux model

ARBITER classifies an outcome measurement profile once per assessed outcome with the auxiliary LLM, rather than using deterministic keyword rules, because the profile is defined by measurement characteristics and registry context rather than outcome names alone. The resulting structured profile is non-citable advisory orientation for Domain 4 and Domain 5 prompts; if classification fails or the available definition is insufficient, ARBITER injects an explicit `unclear` profile so downstream prompts see that the classification was attempted without fabricating certainty.

This trades the repeatability of keyword matching for better handling of composites, scales, thresholds, and trial-specific outcome definitions. The deterministic graph still owns when the profile is computed and where it is injected; the LLM owns only the taxonomy assignment and short basis.
