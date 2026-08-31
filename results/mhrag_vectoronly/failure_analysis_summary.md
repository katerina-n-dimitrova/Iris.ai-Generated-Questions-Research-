# Failure analysis — generated-question vs chunk retrieval (@k=5)

Queries: 84

## Query buckets

- **partial_multihop_generated**: 55
- **unchanged**: 51
- **complete_multihop_generated**: 20
- **hurt_by_generated**: 17
- **improved_by_generated**: 16
- **no_evidence_generated**: 9

## Generated-question diagnostics

- **top_question_points_to_gold_parent**: 44
- **top_question_points_to_nongold_parent**: 40
- **lower_ranked_question_hits_gold**: 38
- **gold_present_but_below_topk**: 7

## Representative contrast examples

### q00141 (temporal, 3 facts) — generated better
> After the Polygon report on Valve's updates to the Steam Deck hardware published on November 9, 2023, and the Engadget review of the Steam Deck OLED version published on the same date, was the reporting on Valve's improvements to the Steam Deck hardware consistent?
- gold answer: `Yes`  | gold chunks: ['a03::c7', 'a05::c4', 'a05::c5', 'a14::c13']
- baseline covered 1 / generated covered 2
- B top match question: _What does Valve estimate regarding hardware updates since the original Steam Deck launched?_

### q00215 (inference, 4 facts) — generated better
> Which company, recently mentioned in articles by both TechCrunch and The Verge, is not planning new measures for a major video platform in the next six months, secures default search engine positions through deals with other tech giants, and is accused of harming news publishers' revenue through anticompetitive practices?
- gold answer: `Google`  | gold chunks: ['a01::c5', 'a02::c0', 'a13::c2', 'a13::c3']
- baseline covered 1 / generated covered 2
- B top match question: _What is the main argument of the lawsuit regarding Google’s impact on news publishers?_

### q00307 (temporal, 3 facts) — baseline better
> Between the report from The Age on the Sydney Swans' position in the AFLW standings published on October 20, 2023, and the subsequent report from The Age on the Sydney Swans' standings published on November 3, 2023, was there no change in the Sydney Swans' ranking in the AFLW?
- gold answer: `no`  | gold chunks: ['a07::c9', 'a08::c0', 'a12::c0']
- baseline covered 2 / generated covered 1
- B top match question: _When did the Swans win their first AFLW match?_

### q00349 (inference, 3 facts) — generated better
> Which company is at the center of concerns from 'The Age' for manipulating search results to maximize ad revenue, from 'TechCrunch' for not planning additional measures on its video platform within six months, and is accused in another 'TechCrunch' article of anticompetitively affecting news publishers' content, readers, and advertising income?
- gold answer: `Google`  | gold chunks: ['a01::c5', 'a02::c0', 'a09::c4']
- baseline covered 1 / generated covered 2
- B top match question: _What is the main argument of the lawsuit regarding Google’s impact on news publishers?_

### q00581 (inference, 4 facts) — generated better
> Which company, covered by both Engadget and Polygon, is set to release an updated version of their hardware with numerous improvements and immediate availability starting November 16th?
- gold answer: `Valve`  | gold chunks: ['a03::c7', 'a04::c0', 'a05::c4', 'a05::c5']
- baseline covered 2 / generated covered 3
- B top match question: _What does Valve estimate regarding hardware updates since the original Steam Deck launched?_

### q00610 (temporal, 2 facts) — generated better
> After the Polygon report on the Steam Deck OLED improvements published at 18:00:00 on November 9, 2023, and the Engadget review of the Steam Deck OLED published at 18:00:38 on the same day, was there agreement between the two sources regarding the availability of the new iteration of the Steam Deck from Valve?
- gold answer: `Yes`  | gold chunks: ['a03::c7', 'a05::c5']
- baseline covered 1 / generated covered 2
- B top match question: _What is the release date for the Steam Deck OLED?_
