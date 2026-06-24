# Context-Enrichment Method Comparison — Summary

## 1. Executive summary

We tested whether different context-enrichment methods improve RAG retrieval and answer quality across structured scientific text, unstructured biomedical text, tables, charts, and formulas. For each document type we compared a **baseline** (raw content) against three targeted enrichment methods and a **combined_best** condition, measuring retrieval quality, answer quality, latency, token usage, and cost.

## 2. Dataset overview

| Dataset | Data type | Baseline representation | Enrichment methods tested | Docs (chunks) | Queries |
|---|---|---|---|---|---|
| scifact | Structured scientific text | Raw abstract sentence | title_abstract_context, neighboring_context, llm_generated_chunk_context, combined_best | 377 | 15 |
| nfcorpus | Unstructured biomedical text | Raw biomedical passage chunk | generated_questions, keywords_entities, plain_summary, combined_best | 887 | 15 |
| wikitablequestions | Tables | Linearized table row | column_headers_per_row, table_page_title, natural_language_row_summary, combined_best | 768 | 15 |
| chartqa | Charts / graphs | Chart OCR/caption text | chart_to_table_data, axis_legend_title_metadata, chart_summary, combined_best | 30 | 15 |
| formulareasoning | Mathematical formulas | Raw formula text | surrounding_text, variable_definitions, latex_structure, combined_best | 272 | 15 |

## 3. Results by dataset

### scifact — Structured scientific text

- **Best retrieval method:** `neighboring_context` (nDCG@10 0.970, baseline 0.951)
- **Worst retrieval method:** `combined_best` (nDCG@10 0.933)
- **Best answer-quality method:** `llm_generated_chunk_context` (faithfulness 0.433, baseline 0.533)
- Baseline → retrieval nDCG@10 0.951, MRR 0.933; answer faithfulness 0.533, relevance 0.467.

### nfcorpus — Unstructured biomedical text

- **Best retrieval method:** `plain_summary` (nDCG@10 0.651, baseline 0.654)
- **Worst retrieval method:** `keywords_entities` (nDCG@10 0.641)
- **Best answer-quality method:** `generated_questions` (faithfulness 0.587, baseline 0.400)
- Baseline → retrieval nDCG@10 0.654, MRR 0.911; answer faithfulness 0.400, relevance 0.400.

### wikitablequestions — Tables

- **Best retrieval method:** `natural_language_row_summary` (nDCG@10 0.867, baseline 0.800)
- **Worst retrieval method:** `column_headers_per_row` (nDCG@10 0.800)
- **Best answer-quality method:** `column_headers_per_row` (faithfulness 0.467, baseline 0.533)
- Baseline → retrieval nDCG@10 0.800, MRR 0.800; answer faithfulness 0.533, relevance 0.667.

### chartqa — Charts / graphs

- **Best retrieval method:** `axis_legend_title_metadata` (nDCG@10 0.322, baseline 0.209)
- **Worst retrieval method:** `chart_to_table_data` (nDCG@10 0.245)
- **Best answer-quality method:** `axis_legend_title_metadata` (faithfulness 0.267, baseline 0.067)
- Baseline → retrieval nDCG@10 0.209, MRR 0.112; answer faithfulness 0.067, relevance 0.133.

### formulareasoning — Mathematical formulas

- **Best retrieval method:** `variable_definitions` (nDCG@10 0.076, baseline 0.012)
- **Worst retrieval method:** `latex_structure` (nDCG@10 0.013)
- **Best answer-quality method:** `surrounding_text` (faithfulness 0.067, baseline 0.000)
- Baseline → retrieval nDCG@10 0.012, MRR 0.007; answer faithfulness 0.000, relevance 0.133.

## 4. Method comparison table

| Dataset | Method | Recall@5 | MRR | nDCG@10 | Hit@5 | Faithfulness | Answer rel. | Retr. latency (ms) | p95 (ms) | Token cost ($) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scifact | baseline | 1.000 | 0.933 | 0.951 | 1.000 | 0.533 | 0.467 | 1001.398 | 1686.993 | 0.000717 | — |
| scifact | title_abstract_context | 1.000 | 0.922 | 0.942 | 1.000 | 0.367 | 0.400 | 926.231 | 1503.718 | 0.002439 | both ↓ |
| scifact | neighboring_context | 1.000 | 0.967 | 0.970 | 1.000 | 0.367 | 0.400 | 893.798 | 1696.153 | 0.001507 | retrieval ↑ / answer ↓ |
| scifact | llm_generated_chunk_context | 1.000 | 0.950 | 0.962 | 1.000 | 0.433 | 0.467 | 923.483 | 1523.258 | 0.001164 | retrieval ↑ / answer ↓ |
| scifact | combined_best | 0.933 | 0.933 | 0.933 | 0.933 | 0.367 | 0.400 | 1167.431 | 2105.908 | 0.003185 | both ↓ |
| nfcorpus | baseline | 0.099 | 0.911 | 0.654 | 1.000 | 0.400 | 0.400 | 1152.101 | 2016.799 | 0.002267 | — |
| nfcorpus | generated_questions | 0.094 | 0.922 | 0.644 | 1.000 | 0.587 | 0.600 | 1004.824 | 1791.774 | 0.002523 | retrieval ↓ / answer ↑ |
| nfcorpus | keywords_entities | 0.102 | 0.922 | 0.641 | 1.000 | 0.400 | 0.400 | 1068.646 | 1919.124 | 0.002593 | both ↓ |
| nfcorpus | plain_summary | 0.097 | 0.933 | 0.651 | 1.000 | 0.467 | 0.400 | 1048.612 | 1798.386 | 0.002779 | retrieval ↓ / answer ↑ |
| nfcorpus | combined_best | 0.097 | 0.922 | 0.643 | 1.000 | 0.533 | 0.533 | 1148.921 | 2101.805 | 0.003275 | retrieval ↓ / answer ↑ |
| wikitablequestions | baseline | 0.800 | 0.800 | 0.800 | 0.800 | 0.533 | 0.667 | 857.297 | 1404.295 | 0.000718 | — |
| wikitablequestions | column_headers_per_row | 0.800 | 0.800 | 0.800 | 0.800 | 0.467 | 0.533 | 850.451 | 1498.541 | 0.000903 | both ↓ |
| wikitablequestions | table_page_title | 0.867 | 0.822 | 0.833 | 0.867 | 0.467 | 0.600 | 887.653 | 1553.864 | 0.000867 | retrieval ↑ / answer ↓ |
| wikitablequestions | natural_language_row_summary | 0.933 | 0.844 | 0.867 | 0.933 | 0.400 | 0.533 | 1076.545 | 1965.053 | 0.001306 | retrieval ↑ / answer ↓ |
| wikitablequestions | combined_best | 0.800 | 0.800 | 0.800 | 0.800 | 0.333 | 0.467 | 979.736 | 1887.139 | 0.001674 | both ↓ |
| chartqa | baseline | 0.333 | 0.112 | 0.209 | 0.333 | 0.067 | 0.133 | 313.350 | 433.684 | 0.000218 | — |
| chartqa | chart_to_table_data | 0.267 | 0.159 | 0.245 | 0.267 | 0.067 | 0.400 | 305.478 | 452.579 | 0.000548 | retrieval ↑ / answer ↓ |
| chartqa | axis_legend_title_metadata | 0.267 | 0.241 | 0.322 | 0.267 | 0.267 | 0.667 | 350.420 | 421.255 | 0.000434 | retrieval+answer ↑ |
| chartqa | chart_summary | 0.467 | 0.178 | 0.264 | 0.467 | 0.133 | 0.200 | 296.613 | 426.304 | 0.000333 | retrieval+answer ↑ |
| chartqa | combined_best | 0.333 | 0.220 | 0.306 | 0.333 | 0.067 | 0.733 | 298.687 | 425.296 | 0.000802 | retrieval ↑ / answer ↓ |
| formulareasoning | baseline | 0.000 | 0.007 | 0.012 | 0.000 | 0.000 | 0.133 | 1032.311 | 1520.861 | 0.000796 | — |
| formulareasoning | surrounding_text | 0.033 | 0.030 | 0.033 | 0.067 | 0.067 | 0.133 | 1022.111 | 1623.553 | 0.001018 | retrieval+answer ↑ |
| formulareasoning | variable_definitions | 0.089 | 0.064 | 0.076 | 0.133 | 0.000 | 0.200 | 975.041 | 1498.023 | 0.001373 | retrieval ↑ / answer ↓ |
| formulareasoning | latex_structure | 0.000 | 0.008 | 0.013 | 0.000 | 0.000 | 0.000 | 1014.501 | 1609.747 | 0.001383 | retrieval ↑ / answer ↓ |
| formulareasoning | combined_best | 0.022 | 0.059 | 0.064 | 0.067 | 0.067 | 0.133 | 999.501 | 1583.384 | 0.002028 | retrieval+answer ↑ |

## 5. Overall ranking (best enrichment method per dataset)

| Dataset | Best retrieval Δ | Best answer Δ | Lowest added latency | Best overall trade-off |
|---|---|---|---|---|
| scifact | neighboring_context (+0.019) | llm_generated_chunk_context (-0.100) | neighboring_context (-107.600) | llm_generated_chunk_context |
| nfcorpus | plain_summary (-0.004) | generated_questions (+0.187) | generated_questions (-147.277) | generated_questions |
| wikitablequestions | natural_language_row_summary (+0.067) | column_headers_per_row (-0.067) | column_headers_per_row (-6.846) | table_page_title |
| chartqa | axis_legend_title_metadata (+0.113) | axis_legend_title_metadata (+0.200) | chart_summary (-16.737) | axis_legend_title_metadata |
| formulareasoning | variable_definitions (+0.064) | surrounding_text (+0.067) | variable_definitions (-57.270) | combined_best |

## 6. Main findings

- **Best method for Structured scientific text (SciFact):** `neighboring_context`
- **Best method for Unstructured biomedical text (NFCorpus):** `plain_summary`
- **Best method for Tables (WikiTableQuestions):** `natural_language_row_summary`
- **Best method for Charts (ChartQA):** `axis_legend_title_metadata`
- **Best method for Formulas (FormulaReasoning):** `variable_definitions`
- **Methods that hurt retrieval (nDCG@10 below baseline):** scifact/title_abstract_context, scifact/combined_best, nfcorpus/generated_questions, nfcorpus/keywords_entities, nfcorpus/plain_summary, nfcorpus/combined_best
- **Did better retrieval always mean better answers?** No. Cases where retrieval improved but answer quality dropped: scifact/neighboring_context, scifact/llm_generated_chunk_context, wikitablequestions/table_page_title, wikitablequestions/natural_language_row_summary.
- **Highest indexing-latency condition:** nfcorpus/combined_best (113.857s).

## 7. Final recommendation table

| Data type | Recommended method | Why | Trade-off | Use or avoid? |
|---|---|---|---|---|
| Structured scientific text | llm_generated_chunk_context | no net gain over baseline in this run | adds encoding latency + token cost | Avoid (enrichment did not help here) |
| Unstructured biomedical text | generated_questions | improves retrieval and/or answer grounding | adds encoding latency + token cost | Use |
| Tables | table_page_title | no net gain over baseline in this run | adds encoding latency + token cost | Avoid (enrichment did not help here) |
| Charts / graphs | axis_legend_title_metadata | improves retrieval and/or answer grounding | adds encoding latency + token cost | Use |
| Mathematical formulas | combined_best | improves retrieval and/or answer grounding | adds encoding latency + token cost | Use |

## Notes & caveats

- Small debug sample; treat magnitudes as indicative, not final.
- ChartQA chart-to-table and axis metadata are documented placeholders (require OCR/vision). FormulaReasoning 'surrounding text' is a placeholder (formula DB stores standalone formulas).
- Answer grades come from an LLM judge (gpt-4o-mini) and are noisy.
