# Context-Enrichment Method Comparison — Summary

## 1. Executive summary

We tested whether different context-enrichment methods improve RAG retrieval and answer quality across structured scientific text, unstructured biomedical text, tables, charts, and formulas. For each document type we compared a **baseline** (raw content) against three targeted enrichment methods and a **combined_best** condition, measuring retrieval quality, answer quality, latency, token usage, and cost.

## 2. Dataset overview

| Dataset | Data type | Baseline representation | Enrichment methods tested | Docs (chunks) | Queries |
|---|---|---|---|---|---|
| formulareasoning | Mathematical formulas | Raw formula text | surrounding_text, variable_definitions, latex_structure, combined_best | 272 | 30 |
| nfcorpus | Unstructured biomedical text | Raw biomedical passage chunk | generated_questions, keywords_entities, plain_summary, doc_summary_position, combined_best | 1324 | 30 |
| chartqa | Charts / graphs | Chart OCR/caption text | chart_to_table_data, axis_legend_title_metadata, chart_summary, vision_chart_description, combined_best | 40 | 30 |
| scifact | Structured scientific text | Raw abstract sentence | title_abstract_context, neighboring_context, llm_generated_chunk_context, doc_summary_position, generated_questions, combined_best | 654 | 30 |
| wikitablequestions | Tables | Linearized table row | column_headers_per_row, table_page_title, natural_language_row_summary, whole_table_summary, combined_best | 938 | 30 |

## 3. Results by dataset

### formulareasoning — Mathematical formulas

- **Best retrieval method:** `combined_best` (nDCG@10 0.079, baseline 0.019)
- **Worst retrieval method:** `latex_structure` (nDCG@10 0.016)
- **Best answer-quality method:** `variable_definitions` (faithfulness 0.100, baseline 0.033)
- Baseline → retrieval nDCG@10 0.019, MRR 0.012; answer faithfulness 0.033, relevance 0.233.

### nfcorpus — Unstructured biomedical text

- **Best retrieval method:** `combined_best` (nDCG@10 0.542, baseline 0.523)
- **Worst retrieval method:** `doc_summary_position` (nDCG@10 0.506)
- **Best answer-quality method:** `plain_summary` (faithfulness 0.533, baseline 0.433)
- Baseline → retrieval nDCG@10 0.523, MRR 0.831; answer faithfulness 0.433, relevance 0.433.

### chartqa — Charts / graphs

- **Best retrieval method:** `vision_chart_description` (nDCG@10 0.448, baseline 0.166)
- **Worst retrieval method:** `chart_to_table_data` (nDCG@10 0.178)
- **Best answer-quality method:** `vision_chart_description` (faithfulness 0.467, baseline 0.100)
- Baseline → retrieval nDCG@10 0.166, MRR 0.098; answer faithfulness 0.100, relevance 0.167.

### scifact — Structured scientific text

- **Best retrieval method:** `generated_questions` (nDCG@10 0.925, baseline 0.881)
- **Worst retrieval method:** `neighboring_context` (nDCG@10 0.892)
- **Best answer-quality method:** `title_abstract_context` (faithfulness 0.450, baseline 0.567)
- Baseline → retrieval nDCG@10 0.881, MRR 0.871; answer faithfulness 0.567, relevance 0.533.

### wikitablequestions — Tables

- **Best retrieval method:** `whole_table_summary` (nDCG@10 0.833, baseline 0.809)
- **Worst retrieval method:** `natural_language_row_summary` (nDCG@10 0.792)
- **Best answer-quality method:** `column_headers_per_row` (faithfulness 0.500, baseline 0.533)
- Baseline → retrieval nDCG@10 0.809, MRR 0.800; answer faithfulness 0.533, relevance 0.700.

## 4. Method comparison table

| Dataset | Method | Recall@5 | MRR | nDCG@10 | Hit@5 | Faithfulness | Answer rel. | Retr. latency (ms) | p95 (ms) | Token cost ($) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| formulareasoning | baseline | 0.000 | 0.012 | 0.019 | 0.000 | 0.033 | 0.233 | 726.696 | 1719.456 | 0.00156 | — |
| formulareasoning | surrounding_text | 0.044 | 0.033 | 0.037 | 0.100 | 0.033 | 0.100 | 939.155 | 1663.259 | 0.001921 | retrieval ↑ / answer ↓ |
| formulareasoning | variable_definitions | 0.128 | 0.060 | 0.078 | 0.167 | 0.100 | 0.200 | 601.644 | 1561.999 | 0.002838 | retrieval+answer ↑ |
| formulareasoning | latex_structure | 0.000 | 0.011 | 0.016 | 0.000 | 0.033 | 0.033 | 782.844 | 1621.100 | 0.002705 | both ↓ |
| formulareasoning | combined_best | 0.094 | 0.057 | 0.079 | 0.133 | 0.067 | 0.167 | 670.905 | 1640.364 | 0.003864 | retrieval+answer ↑ |
| nfcorpus | baseline | 0.124 | 0.831 | 0.523 | 0.900 | 0.433 | 0.433 | 681.355 | 2124.927 | 0.004579 | — |
| nfcorpus | generated_questions | 0.124 | 0.892 | 0.542 | 0.967 | 0.233 | 0.233 | 709.656 | 2264.088 | 0.005624 | retrieval ↑ / answer ↓ |
| nfcorpus | keywords_entities | 0.124 | 0.883 | 0.533 | 0.967 | 0.400 | 0.400 | 816.230 | 1960.859 | 0.005228 | retrieval ↑ / answer ↓ |
| nfcorpus | plain_summary | 0.121 | 0.836 | 0.526 | 0.933 | 0.533 | 0.533 | 636.159 | 2008.729 | 0.005731 | retrieval+answer ↑ |
| nfcorpus | doc_summary_position | 0.122 | 0.856 | 0.506 | 0.933 | 0.433 | 0.467 | 760.790 | 2413.487 | 0.006042 | both ↓ |
| nfcorpus | combined_best | 0.123 | 0.872 | 0.542 | 0.933 | 0.267 | 0.283 | 907.617 | 2776.602 | 0.007412 | retrieval ↑ / answer ↓ |
| chartqa | baseline | 0.133 | 0.098 | 0.166 | 0.133 | 0.100 | 0.167 | 280.132 | 635.405 | 0.000462 | — |
| chartqa | chart_to_table_data | 0.200 | 0.130 | 0.178 | 0.200 | 0.033 | 0.467 | 282.599 | 414.925 | 0.001126 | retrieval ↑ / answer ↓ |
| chartqa | axis_legend_title_metadata | 0.200 | 0.184 | 0.248 | 0.200 | 0.067 | 0.400 | 215.549 | 435.632 | 0.000901 | retrieval ↑ / answer ↓ |
| chartqa | chart_summary | 0.167 | 0.128 | 0.182 | 0.167 | 0.100 | 0.100 | 234.374 | 465.286 | 0.000988 | retrieval ↑ / answer ↓ |
| chartqa | vision_chart_description | 0.533 | 0.368 | 0.448 | 0.533 | 0.467 | 0.500 | 223.406 | 433.963 | 0.004003 | retrieval+answer ↑ |
| chartqa | combined_best | 0.233 | 0.140 | 0.192 | 0.233 | 0.000 | 0.233 | 215.352 | 421.673 | 0.001943 | retrieval ↑ / answer ↓ |
| scifact | baseline | 0.893 | 0.871 | 0.881 | 0.900 | 0.567 | 0.533 | 707.126 | 2108.585 | 0.001309 | — |
| scifact | title_abstract_context | 0.913 | 0.917 | 0.906 | 0.933 | 0.450 | 0.467 | 636.793 | 1887.794 | 0.004789 | retrieval ↑ / answer ↓ |
| scifact | neighboring_context | 0.920 | 0.894 | 0.892 | 0.933 | 0.383 | 0.400 | 676.175 | 2728.144 | 0.002867 | retrieval ↑ / answer ↓ |
| scifact | llm_generated_chunk_context | 0.913 | 0.917 | 0.906 | 0.933 | 0.417 | 0.467 | 740.146 | 2422.096 | 0.002504 | retrieval ↑ / answer ↓ |
| scifact | doc_summary_position | 0.913 | 0.933 | 0.918 | 0.933 | 0.383 | 0.400 | 672.615 | 1839.435 | 0.00318 | retrieval ↑ / answer ↓ |
| scifact | generated_questions | 0.993 | 0.907 | 0.925 | 1.000 | 0.317 | 0.367 | 538.545 | 1691.304 | 0.002482 | retrieval ↑ / answer ↓ |
| scifact | combined_best | 0.913 | 0.917 | 0.906 | 0.933 | 0.417 | 0.433 | 734.669 | 2116.811 | 0.006492 | retrieval ↑ / answer ↓ |
| wikitablequestions | baseline | 0.833 | 0.800 | 0.809 | 0.833 | 0.533 | 0.700 | 797.698 | 2587.934 | 0.001539 | — |
| wikitablequestions | column_headers_per_row | 0.833 | 0.790 | 0.801 | 0.833 | 0.500 | 0.667 | 928.497 | 2414.606 | 0.001925 | both ↓ |
| wikitablequestions | table_page_title | 0.867 | 0.808 | 0.823 | 0.867 | 0.433 | 0.600 | 775.818 | 2314.148 | 0.001803 | retrieval ↑ / answer ↓ |
| wikitablequestions | natural_language_row_summary | 0.833 | 0.778 | 0.792 | 0.833 | 0.433 | 0.567 | 609.183 | 1980.540 | 0.002804 | both ↓ |
| wikitablequestions | whole_table_summary | 0.833 | 0.833 | 0.833 | 0.833 | 0.267 | 0.400 | 594.007 | 1813.914 | 0.002792 | retrieval ↑ / answer ↓ |
| wikitablequestions | combined_best | 0.833 | 0.800 | 0.809 | 0.833 | 0.367 | 0.467 | 663.054 | 1938.786 | 0.00361 | both ↓ |

## 5. Overall ranking (best enrichment method per dataset)

| Dataset | Best retrieval Δ | Best answer Δ | Lowest added latency | Best overall trade-off |
|---|---|---|---|---|
| formulareasoning | combined_best (+0.060) | variable_definitions (+0.067) | variable_definitions (-125.052) | variable_definitions |
| nfcorpus | combined_best (+0.019) | plain_summary (+0.100) | plain_summary (-45.197) | plain_summary |
| chartqa | vision_chart_description (+0.282) | vision_chart_description (+0.367) | combined_best (-64.780) | vision_chart_description |
| scifact | generated_questions (+0.043) | title_abstract_context (-0.117) | generated_questions (-168.582) | title_abstract_context |
| wikitablequestions | whole_table_summary (+0.025) | column_headers_per_row (-0.033) | whole_table_summary (-203.691) | column_headers_per_row |

## 6. Main findings

- **Best method for Structured scientific text (SciFact):** `generated_questions`
- **Best method for Unstructured biomedical text (NFCorpus):** `combined_best`
- **Best method for Tables (WikiTableQuestions):** `whole_table_summary`
- **Best method for Charts (ChartQA):** `vision_chart_description`
- **Best method for Formulas (FormulaReasoning):** `combined_best`
- **Methods that hurt retrieval (nDCG@10 below baseline):** formulareasoning/latex_structure, nfcorpus/doc_summary_position, wikitablequestions/column_headers_per_row, wikitablequestions/natural_language_row_summary
- **Did better retrieval always mean better answers?** No. Cases where retrieval improved but answer quality dropped: nfcorpus/generated_questions, nfcorpus/keywords_entities, nfcorpus/combined_best, chartqa/chart_to_table_data, chartqa/axis_legend_title_metadata, chartqa/combined_best, scifact/title_abstract_context, scifact/neighboring_context, scifact/llm_generated_chunk_context, scifact/doc_summary_position, scifact/generated_questions, scifact/combined_best, wikitablequestions/table_page_title, wikitablequestions/whole_table_summary.
- **Highest indexing-latency condition:** nfcorpus/combined_best (187.7324s).

## 7. Final recommendation table

| Data type | Recommended method | Why | Trade-off | Use or avoid? |
|---|---|---|---|---|
| Mathematical formulas | variable_definitions | improves retrieval and/or answer grounding | adds encoding latency + token cost | Use |
| Unstructured biomedical text | plain_summary | improves retrieval and/or answer grounding | adds encoding latency + token cost | Use |
| Charts / graphs | vision_chart_description | improves retrieval and/or answer grounding | adds encoding latency + token cost | Use |
| Structured scientific text | title_abstract_context | no net gain over baseline in this run | adds encoding latency + token cost | Avoid (enrichment did not help here) |
| Tables | column_headers_per_row | no net gain over baseline in this run | adds encoding latency + token cost | Avoid (enrichment did not help here) |

## Notes & caveats

- Small debug sample; treat magnitudes as indicative, not final.
- ChartQA chart-to-table and axis metadata are documented placeholders (require OCR/vision). FormulaReasoning 'surrounding text' is a placeholder (formula DB stores standalone formulas).
- Answer grades come from an LLM judge (gpt-4o-mini) and are noisy.
