# Context-Enrichment Method Comparison — Summary

## 1. Executive summary

We tested whether different context-enrichment methods improve RAG retrieval and answer quality across structured scientific text, unstructured biomedical text, tables, charts, and formulas. For each document type we compared a **baseline** (raw content) against three targeted enrichment methods and a **combined_best** condition, measuring retrieval quality, answer quality, latency, token usage, and cost.

## 2. Dataset overview

| Dataset | Data type | Baseline representation | Enrichment methods tested | Docs (chunks) | Queries |
|---|---|---|---|---|---|
| scifact | Structured scientific text | Raw abstract sentence | title_abstract_context, neighboring_context, llm_generated_chunk_context, combined_best | 654 | 30 |
| nfcorpus | Unstructured biomedical text | Raw biomedical passage chunk | generated_questions, keywords_entities, plain_summary, combined_best | 1324 | 30 |
| wikitablequestions | Tables | Linearized table row | column_headers_per_row, table_page_title, natural_language_row_summary, combined_best | 938 | 30 |
| chartqa | Charts / graphs | Chart OCR/caption text | chart_to_table_data, axis_legend_title_metadata, chart_summary, combined_best | 40 | 30 |
| formulareasoning | Mathematical formulas | Raw formula text | surrounding_text, variable_definitions, latex_structure, combined_best | 272 | 30 |

## 3. Results by dataset

### scifact — Structured scientific text

- **Best retrieval method:** `title_abstract_context` (nDCG@10 0.906, baseline 0.881)
- **Worst retrieval method:** `neighboring_context` (nDCG@10 0.892)
- **Best answer-quality method:** `llm_generated_chunk_context` (faithfulness 0.460, baseline 0.567)
- Baseline → retrieval nDCG@10 0.881, MRR 0.871; answer faithfulness 0.567, relevance 0.500.

### nfcorpus — Unstructured biomedical text

- **Best retrieval method:** `combined_best` (nDCG@10 0.547, baseline 0.523)
- **Worst retrieval method:** `keywords_entities` (nDCG@10 0.533)
- **Best answer-quality method:** `plain_summary` (faithfulness 0.527, baseline 0.433)
- Baseline → retrieval nDCG@10 0.523, MRR 0.831; answer faithfulness 0.433, relevance 0.400.

### wikitablequestions — Tables

- **Best retrieval method:** `table_page_title` (nDCG@10 0.823, baseline 0.809)
- **Worst retrieval method:** `natural_language_row_summary` (nDCG@10 0.792)
- **Best answer-quality method:** `column_headers_per_row` (faithfulness 0.533, baseline 0.533)
- Baseline → retrieval nDCG@10 0.809, MRR 0.800; answer faithfulness 0.533, relevance 0.733.

### chartqa — Charts / graphs

- **Best retrieval method:** `axis_legend_title_metadata` (nDCG@10 0.248, baseline 0.166)
- **Worst retrieval method:** `chart_to_table_data` (nDCG@10 0.178)
- **Best answer-quality method:** `chart_summary` (faithfulness 0.067, baseline 0.100)
- Baseline → retrieval nDCG@10 0.166, MRR 0.098; answer faithfulness 0.100, relevance 0.167.

### formulareasoning — Mathematical formulas

- **Best retrieval method:** `combined_best` (nDCG@10 0.079, baseline 0.019)
- **Worst retrieval method:** `latex_structure` (nDCG@10 0.016)
- **Best answer-quality method:** `variable_definitions` (faithfulness 0.100, baseline 0.033)
- Baseline → retrieval nDCG@10 0.019, MRR 0.012; answer faithfulness 0.033, relevance 0.233.

## 4. Method comparison table

| Dataset | Method | Recall@5 | MRR | nDCG@10 | Hit@5 | Faithfulness | Answer rel. | Retr. latency (ms) | p95 (ms) | Token cost ($) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scifact | baseline | 0.893 | 0.871 | 0.881 | 0.900 | 0.567 | 0.500 | 918.634 | 2726.099 | 0.001319 | — |
| scifact | title_abstract_context | 0.913 | 0.917 | 0.906 | 0.933 | 0.417 | 0.433 | 695.254 | 1930.351 | 0.004782 | retrieval ↑ / answer ↓ |
| scifact | neighboring_context | 0.920 | 0.894 | 0.892 | 0.933 | 0.383 | 0.400 | 559.173 | 1623.650 | 0.002868 | retrieval ↑ / answer ↓ |
| scifact | llm_generated_chunk_context | 0.913 | 0.917 | 0.906 | 0.933 | 0.460 | 0.467 | 700.727 | 2392.639 | 0.002491 | retrieval ↑ / answer ↓ |
| scifact | combined_best | 0.913 | 0.917 | 0.906 | 0.933 | 0.417 | 0.433 | 662.507 | 1992.931 | 0.006463 | retrieval ↑ / answer ↓ |
| nfcorpus | baseline | 0.124 | 0.831 | 0.523 | 0.900 | 0.433 | 0.400 | 647.087 | 1960.900 | 0.00452 | — |
| nfcorpus | generated_questions | 0.125 | 0.908 | 0.535 | 0.967 | 0.233 | 0.233 | 717.799 | 2317.294 | 0.00573 | retrieval ↑ / answer ↓ |
| nfcorpus | keywords_entities | 0.124 | 0.883 | 0.533 | 0.967 | 0.433 | 0.400 | 702.104 | 2211.047 | 0.005246 | retrieval ↑ / answer ↓ |
| nfcorpus | plain_summary | 0.121 | 0.875 | 0.534 | 0.933 | 0.527 | 0.533 | 804.736 | 2547.438 | 0.005749 | retrieval+answer ↑ |
| nfcorpus | combined_best | 0.122 | 0.887 | 0.547 | 0.933 | 0.267 | 0.250 | 879.756 | 2562.990 | 0.00742 | retrieval ↑ / answer ↓ |
| wikitablequestions | baseline | 0.833 | 0.800 | 0.809 | 0.833 | 0.533 | 0.733 | 515.175 | 1417.916 | 0.001538 | — |
| wikitablequestions | column_headers_per_row | 0.833 | 0.790 | 0.801 | 0.833 | 0.533 | 0.700 | 634.472 | 1997.948 | 0.001929 | both ↓ |
| wikitablequestions | table_page_title | 0.867 | 0.808 | 0.823 | 0.867 | 0.367 | 0.600 | 587.842 | 1851.373 | 0.00179 | retrieval ↑ / answer ↓ |
| wikitablequestions | natural_language_row_summary | 0.833 | 0.778 | 0.792 | 0.833 | 0.467 | 0.533 | 750.076 | 2170.918 | 0.002805 | both ↓ |
| wikitablequestions | combined_best | 0.833 | 0.800 | 0.809 | 0.833 | 0.400 | 0.467 | 671.934 | 2015.637 | 0.003615 | both ↓ |
| chartqa | baseline | 0.133 | 0.098 | 0.166 | 0.133 | 0.100 | 0.167 | 229.579 | 412.990 | 0.000462 | — |
| chartqa | chart_to_table_data | 0.200 | 0.130 | 0.178 | 0.200 | 0.033 | 0.367 | 294.088 | 429.759 | 0.001102 | retrieval ↑ / answer ↓ |
| chartqa | axis_legend_title_metadata | 0.200 | 0.184 | 0.248 | 0.200 | 0.033 | 0.433 | 262.419 | 464.347 | 0.000901 | retrieval ↑ / answer ↓ |
| chartqa | chart_summary | 0.200 | 0.119 | 0.184 | 0.200 | 0.067 | 0.100 | 272.828 | 489.324 | 0.001008 | retrieval ↑ / answer ↓ |
| chartqa | combined_best | 0.233 | 0.148 | 0.206 | 0.233 | 0.000 | 0.133 | 260.260 | 522.330 | 0.001957 | retrieval ↑ / answer ↓ |
| formulareasoning | baseline | 0.000 | 0.012 | 0.019 | 0.000 | 0.033 | 0.233 | 726.696 | 1719.456 | 0.00156 | — |
| formulareasoning | surrounding_text | 0.044 | 0.033 | 0.037 | 0.100 | 0.033 | 0.100 | 939.155 | 1663.259 | 0.001921 | retrieval ↑ / answer ↓ |
| formulareasoning | variable_definitions | 0.128 | 0.060 | 0.078 | 0.167 | 0.100 | 0.200 | 601.644 | 1561.999 | 0.002838 | retrieval+answer ↑ |
| formulareasoning | latex_structure | 0.000 | 0.011 | 0.016 | 0.000 | 0.033 | 0.033 | 782.844 | 1621.100 | 0.002705 | both ↓ |
| formulareasoning | combined_best | 0.094 | 0.057 | 0.079 | 0.133 | 0.067 | 0.167 | 670.905 | 1640.364 | 0.003864 | retrieval+answer ↑ |

## 5. Overall ranking (best enrichment method per dataset)

| Dataset | Best retrieval Δ | Best answer Δ | Lowest added latency | Best overall trade-off |
|---|---|---|---|---|
| scifact | title_abstract_context (+0.025) | llm_generated_chunk_context (-0.107) | neighboring_context (-359.461) | llm_generated_chunk_context |
| nfcorpus | combined_best (+0.024) | plain_summary (+0.093) | keywords_entities (+55.018) | plain_summary |
| wikitablequestions | table_page_title (+0.014) | column_headers_per_row (+0.000) | table_page_title (+72.667) | column_headers_per_row |
| chartqa | axis_legend_title_metadata (+0.082) | chart_summary (-0.033) | combined_best (+30.681) | axis_legend_title_metadata |
| formulareasoning | combined_best (+0.060) | variable_definitions (+0.067) | variable_definitions (-125.052) | variable_definitions |

## 6. Main findings

- **Best method for Structured scientific text (SciFact):** `title_abstract_context`
- **Best method for Unstructured biomedical text (NFCorpus):** `combined_best`
- **Best method for Tables (WikiTableQuestions):** `table_page_title`
- **Best method for Charts (ChartQA):** `axis_legend_title_metadata`
- **Best method for Formulas (FormulaReasoning):** `combined_best`
- **Methods that hurt retrieval (nDCG@10 below baseline):** wikitablequestions/column_headers_per_row, wikitablequestions/natural_language_row_summary, formulareasoning/latex_structure
- **Did better retrieval always mean better answers?** No. Cases where retrieval improved but answer quality dropped: scifact/title_abstract_context, scifact/neighboring_context, scifact/llm_generated_chunk_context, scifact/combined_best, nfcorpus/generated_questions, nfcorpus/combined_best, wikitablequestions/table_page_title, chartqa/chart_to_table_data, chartqa/axis_legend_title_metadata, chartqa/chart_summary, chartqa/combined_best.
- **Highest indexing-latency condition:** nfcorpus/combined_best (185.6382s).

## 7. Final recommendation table

| Data type | Recommended method | Why | Trade-off | Use or avoid? |
|---|---|---|---|---|
| Structured scientific text | llm_generated_chunk_context | no net gain over baseline in this run | adds encoding latency + token cost | Avoid (enrichment did not help here) |
| Unstructured biomedical text | plain_summary | improves retrieval and/or answer grounding | adds encoding latency + token cost | Use |
| Tables | column_headers_per_row | no net gain over baseline in this run | adds encoding latency + token cost | Avoid (enrichment did not help here) |
| Charts / graphs | axis_legend_title_metadata | improves retrieval and/or answer grounding | adds encoding latency + token cost | Use |
| Mathematical formulas | variable_definitions | improves retrieval and/or answer grounding | adds encoding latency + token cost | Use |

## Notes & caveats

- Small debug sample; treat magnitudes as indicative, not final.
- ChartQA chart-to-table and axis metadata are documented placeholders (require OCR/vision). FormulaReasoning 'surrounding text' is a placeholder (formula DB stores standalone formulas).
- Answer grades come from an LLM judge (gpt-4o-mini) and are noisy.
