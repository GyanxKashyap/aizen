# Graph Report - tiny-llm  (2026-09-03)

## Corpus Check
- 0 files · ~0 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 462 nodes · 691 edges · 24 communities (22 shown, 2 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 109 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Transformer Architecture & Inference
- Pretraining, Scale & Known Bugs
- Phase Index & Eval Docs
- Phase 2 Data Generator
- Model Card & v0 Lineage
- BPE Tokenizer & Tests
- Fine-tune Data Lineage & Method
- Web Server & Endpoints
- Eval Harness & Method Findings
- Phase 3 Fine-tune Findings
- Web UI & Phase 8 Findings
- Eval Answer Checkers
- Eval Dataset Generator & Integrity
- Phase 3b Data Generator
- Phase 4 BPE Report
- Dataset Lineage & the Confound
- Phase 7 Conversation Report
- Phase 6 Data Generator
- Phase 3 Trainer
- Eval Dataset & Leakage Control
- Phase 6 Hybrid Report
- bAbI Dosage Findings
- Phase 2 Data Validator
- Inference Decorator

## God Nodes (most connected - your core abstractions)
1. `BPETokenizer` - 24 edges
2. `Repository file map` - 20 edges
3. `Aizen` - 19 edges
4. `Phase 3 Training Report — Instruction + Reasoning Fine-tune` - 13 edges
5. `From-scratch BPE tokenizer (tokenizer.py)` - 13 edges
6. `Phase 4 — From-Scratch BPE Tokenizer + 512-Token Context` - 12 edges
7. `Phase 6 — Hybrid Data: Curated Public + Synthetic` - 11 edges
8. `Frozen 400-question evaluation harness (eval.py)` - 11 edges
9. `Phase 8 — Aizen-40M: The Scale Finale` - 10 edges
10. `README — Aizen project front door` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Per-Mode Hero and Suggestion Chips` --semantically_similar_to--> `Frozen 400-question evaluation harness (eval.py)`  [INFERRED] [semantically similar]
  ui/index.html → docs/evaluation.md
- `Frozen 400-question benchmark` --references--> `main()`  [INFERRED]
  AIZEN.md → eval.py
- `Chunked, resumable training with per-chunk reseeding` --references--> `get_batch()`  [INFERRED]
  AIZEN.md → train_phase8_pretrain.py
- `The per-chunk reseeding bug` --references--> `get_batch()`  [INFERRED]
  AIZEN.md → train_phase5_finetune.py
- `Phase 5 stage 2 — 2,500-step fine-tune` --semantically_similar_to--> `Deliberately low stage-2 learning rate (1e-4 -> 1e-5)`  [INFERRED] [semantically similar]
  docs/phase5_pretraining.md → AIZEN.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **The eight phase reports that form the project's method narrative** — docs_evaluation, docs_phase2_dataset, docs_phase3_training, docs_phase4_bpe, docs_phase6_hybrid, docs_phase7_conversation, docs_phase8_scale [EXTRACTED 1.00]
- **Aizen build pipeline: corpus, tokenizer, architecture, two training stages, frozen benchmark** — aizen_tinystories_corpus, aizen_bpe_tokenizer, aizen_architecture, aizen_stage1_pretrain, aizen_stage2_finetune, aizen_frozen_benchmark [EXTRACTED 1.00]
- **Four real chat failures with v4b and the Phase 6c/7 data built to fix them** — docs_phase7_conversation_digit_drop_failure, docs_phase7_conversation_negatives_failure, docs_phase7_conversation_false_premise_failure, docs_phase7_conversation_not_a_question_failure, docs_phase7_conversation_multi_turn_blocks [EXTRACTED 1.00]
- **Verified Pool Lineage: 6b Is 6 Minus 4,000 bAbI Blocks, and 7 Descends from 6b** — docs_phase6_hybrid_dataset_lineage, docs_phase6_hybrid_subset_verification, docs_phase6_hybrid_babi_dose_cap, docs_phase6_hybrid_phase6_pool_dead_end, docs_phase6_hybrid_hybrid_dataset [EXTRACTED 1.00]
- **Scale/Sampling Confound: Evidence, Missing Ablation and Retraction** — docs_phase8_scale_sampling_confound, docs_phase8_scale_finetune_timeline_evidence, docs_phase8_scale_val_loss_evidence, docs_phase8_scale_missing_ablation, docs_phase8_scale_one_variable_retraction, docs_phase8_scale_reseeding_bug [EXTRACTED 1.00]
- **Q:/A: format contract shared by data generation, training, eval and serving** — docs_phase2_dataset_inline_reasoning_format, docs_phase3_training_answer_masked_loss, docs_phase3_training_example_aligned_windows, docs_phase4_bpe_newline_never_merges, docs_phase4_bpe_tokenization_parity [EXTRACTED 1.00]
- **Single Screen, Two Modes: Switch, Shared Composer, Per-Mode Threads and Theming** — ui_index_mode_switch, ui_index_chat_mode, ui_index_story_mode, ui_index_shared_composer, ui_index_per_mode_threads, ui_index_mode_theming [EXTRACTED 1.00]
- **Cumulative Aizen Phase Training Pool Lineage** — data_aizen_phase2_train_dataset, data_aizen_phase3b_train_dataset, data_aizen_phase6_train_dataset, data_aizen_phase6b_train_dataset, data_aizen_phase7_train_dataset [INFERRED 0.95]
- **Per-chunk reseeding bug and everything it silently distorted** — aizen_reseeding_bug, aizen_chunked_resumable_training, readme_resumable_training, docs_phase5_pretraining_reseeding_hindsight, aizen_phase8_confound [INFERRED 0.95]
- **Per-Phase Incremental Supplements Folded Into Pools** — data_phase3b_extra_dataset, data_phase6_extra_dataset, data_phase7_extra_dataset, data_aizen_phase3b_train_dataset, data_aizen_phase6_train_dataset, data_aizen_phase7_train_dataset [INFERRED 0.95]
- **TinyStories Raw To Cleaned Pretraining Pipeline** — data_tinystories_raw_dataset, data_tinystories_raw2_dataset, data_pretrain_dataset, data_pretrain2_dataset [INFERRED 0.95]

## Communities (24 total, 2 thin omitted)

### Community 0 - "Transformer Architecture & Inference"
Cohesion: 0.05
Nodes (37): Aizen architecture — decoder-only Transformer, GPT-2 layout, Radford et al., GPT-1 (2018) / GPT-2 (2019), Vaswani et al., 'Attention Is All You Need' (2017), answer(), answer(), no_grad, Chat with the BPE-tokenized Aizen (Phase 4). Does not touch chat.py. Run:…, no_grad (+29 more)

### Community 1 - "Pretraining, Scale & Known Bugs"
Cohesion: 0.07
Nodes (42): Chunked, resumable training with per-chunk reseeding, Honest limitations, Silent Metal GPU memory corruption, Phase 8 is confounded, Pretraining is not a scale trick, PyTorch with the MPS backend, The per-chunk reseeding bug, Stage 1 — TinyStories pretraining (16,000 steps) (+34 more)

### Community 2 - "Phase Index & Eval Docs"
Cohesion: 0.06
Nodes (34): The eight phases (evaluation -> scale finale), Some regressions are purchases, Aizen Evaluation System, How to run, Interpreting results, Known limitations of this evaluator, Methodology — `eval.py`, Why this exists (+26 more)

### Community 3 - "Phase 2 Data Generator"
Cohesion: 0.07
Nodes (10): add(), cand_two_step(), fill(), masked(), numtuple(), Phase 2 dataset generator: instruction-following + reasoning training data.…, Draw candidates until `target` examples are accepted., Try to add one example. Returns True if accepted. (+2 more)

### Community 4 - "Model Card & v0 Lineage"
Cohesion: 0.07
Nodes (20): AIZEN.md — Complete Model Card & Build Reference, What was deliberately not used, Learning order: spelling -> words -> grammar -> format -> facts -> arithmetic, Karpathy — nanoGPT and 'Let's build GPT', The 0.8M tiny-shakespeare predecessor (tinygpt.pt), Synthetic task data — computed, never authored, v0 — the original character-level Aizen (14,298,240 params), v0's exact training corpus is no longer reproducible (+12 more)

### Community 5 - "BPE Tokenizer & Tests"
Cohesion: 0.09
Nodes (18): Full printable-ASCII base vocabulary, Sennrich et al., 'Neural Machine Translation of Rare Words with Subword Units' (2016), From-scratch byte-pair tokenizer (4,096 vocab), Three tokenizer invariants (digits, newline, leading space), Old char tokenizer vs new BPE tokenizer - compression analysis (Phase 4 spec…, main(), check(), Tokenizer test suite (Phase 4 spec section 4). Run: python3 test_tokenizer.py… (+10 more)

### Community 6 - "Fine-tune Data Lineage & Method"
Cohesion: 0.12
Nodes (24): Answer-only loss masking, bAbI dose cap — 300 examples per task (2,400 total), Weston et al., 'Towards AI-Complete Question Answering' (2015), bAbI — the curated public task data, Fine-tuning pool lineage (3b + capped bAbI -> 6b -> 7), Deliberately low stage-2 learning rate (1e-4 -> 1e-5), With external data, proportion matters as much as presence, Stage 2 — answer-masked fine-tuning (5,000 steps) (+16 more)

### Community 7 - "Web Server & Endpoints"
Cohesion: 0.18
Nodes (17): Inference server — one Flask app, two models, port 8321, Binding host '::' for Safari's localhost resolution, Newest-first multi-turn history packing, Q:/A: prompt formatting at inference, Quick start — venv, Flask server on port 8321, One server, two models, one page (Chat and Story), route, chat() (+9 more)

### Community 8 - "Eval Harness & Method Findings"
Cohesion: 0.16
Nodes (16): Frozen 400-question evaluation harness (eval.py), Reproducible per-question seeding (1337 + index), Category-appropriate scoring checkers, Inline reasoning-on-answer-line format, Chain-format overgeneralization (arithmetic regression), Phase 3 recommended next experiments, Checkpoint-carried tokenizer auto-detection in eval.py, Every digit is its own token (+8 more)

### Community 9 - "Phase 3 Fine-tune Findings"
Cohesion: 0.14
Nodes (16): Per-category interpretation rules, Catastrophic forgetting warning (mix with qa.txt), Answer-only loss masking, 70/30 qa.txt + Phase 2 window mixture, Example-aligned training windows, Phase 3 fine-tune (v1, aizen_phase3.pt), Template-binding failure (format learned, function not), From-scratch BPE tokenizer (tokenizer.py) (+8 more)

### Community 10 - "Web UI & Phase 8 Findings"
Cohesion: 0.16
Nodes (16): Finding: 555-99 Now Parses — Attention-Precision Diagnosis Confirmed, Aizen-40M (v6) — 12 Layers / 512 Embedding / 8 Heads / 40.19M Params, Bug: Silent Metal GPU Memory Corruption Zeroing Gradients at Batch 24, Doubled Pretraining Corpus (229,640 Stories / 52.5M Tokens), Aizen Single-Screen Web UI, Chat Mode — /chat with Rolling History Pairs, /meta-Driven Footer Line and Live Indicator, Segmented Mode Switch (Chat / Story) (+8 more)

### Community 11 - "Eval Answer Checkers"
Cohesion: 0.18
Nodes (14): check_contains(), check_number(), check_one_word_correct(), check_three_items(), check_yes_no(), _norm(), _numbers(), Evaluation harness for Aizen (Phase 1). Run: python3 eval.py # evaluate… (+6 more)

### Community 12 - "Eval Dataset Generator & Integrity"
Cohesion: 0.15
Nodes (13): Build-your-own recipe, Two standing evaluation rules, Frozen 400-question benchmark, Eval leakage control, Noise floor of roughly +/-7 points at n=50, add(), Generate the held-out evaluation dataset for Aizen -> data/eval.json Design…, Returns True if added; False on collision/duplicate (caller may retry). Length… (+5 more)

### Community 13 - "Phase 3b Data Generator"
Cohesion: 0.21
Nodes (10): add(), cand_deduction2(), fill(), masked(), numtuple(), prop_q(), Phase 3b supplement generator - targeted fixes for the four failure modes…, Turn 'have wings' into question form 'does ... have wings? (+2 more)

### Community 14 - "Phase 4 BPE Report"
Cohesion: 0.17
Nodes (12): 10. Training ([train_phase4.py](../train_phase4.py)), 11. Evaluation (frozen 400 questions, identical scoring), 12. Improvements, 13. Regressions, 14. Limitations, 15. Next, 1. Why character-level tokenization was limiting Aizen, 2-5. How BPE works & how this implementation works ([tokenizer.py](../tokenizer.py)) (+4 more)

### Community 15 - "Dataset Lineage & the Confound"
Cohesion: 0.21
Nodes (12): Decision: Cap bAbI at 2,400 (300 per Task) — the 6b Pool, Verified Dataset Lineage: 3b ⊂ 6b ⊂ 6, and 7 = 6b + Phase-7 Blocks, File-Seam Artifact: One or Two Blocks Joined Without a Blank Line, Finding: the Over-Diluted Phase 6 Pool Is a Dead End, Evidence: Block-by-Block Set Comparison (22,398 of 22,400 Shared), v4b (aizen_phase6b) — 52.25% Overall, Evidence: Fine-Tune Timeline — v4b/v5 Buggy Sampling, v6 Fixed, Missing Ablation: 16M Model Re-Fine-Tuned with the Fixed Sampler (+4 more)

### Community 16 - "Phase 7 Conversation Report"
Cohesion: 0.17
Nodes (12): 1. Objective, 2. The data (`make_phase7_data.py`, 5,700 blocks), 3. Code changes, 4. Result, 5. Interpretation, 6. Limitations, Attention-precision hypothesis (555-99 is a capacity limit, not a data gap), Digit-drop failure: 555-99 read as 55 - 99 = 4 (+4 more)

### Community 17 - "Phase 6 Data Generator"
Cohesion: 0.25
Nodes (6): add(), fill(), masked(), numtuple(), Phase 6 HYBRID dataset generator. Two sources, one file: A. SYNTHETIC (Python-…, render()

### Community 18 - "Phase 3 Trainer"
Cohesion: 0.27
Nodes (9): estimate_loss(), get_batch(), masked_loss(), parse_examples(), no_grad, Phase 3: fine-tune Aizen on a 70/30 mixture of the original qa.txt and the…, Example-aligned packed windows with per-target loss masks., Split corpus into examples; per-char mask = 1 on ' answer\n' after '\nA:'. (+1 more)

### Community 19 - "Eval Dataset & Leakage Control"
Cohesion: 0.28
Nodes (9): data/eval.json (400 questions, 8 categories), Integrity guarantees, Per-category noise floor (~7pp at n=50), The dataset — `data/eval.json`, Whitelisted arithmetic skeletons (324 examples), Python-computed deterministic answers and chains, Eval leakage controls and disjoint vocabulary, Phase 2 reasoning + instruction dataset (10,000 examples) (+1 more)

### Community 20 - "Phase 6 Hybrid Report"
Cohesion: 0.25
Nodes (8): 1. Objective, 2. Starting checkpoint, 3. The hybrid dataset, 4. Training, 5. Result — and one principled iteration, 6. Lesson, 7. Limitations, Phase 6 — Hybrid Data: Curated Public + Synthetic

### Community 21 - "bAbI Dosage Findings"
Cohesion: 0.25
Nodes (8): bAbI Corpus (8 tasks, Muennighoff/babi mirror), Finding: bAbI at 24% of the Pool Crowded Out Logic and Reading, Limitation: the bAbI Dose Was Found by One Iteration, Not a Sweep, Phase 6 Hybrid Pool (26,400 blocks), patterns_v3 — Sequences Rebuilt for the BPE Representation, Lesson: With External Data, Proportion Matters as Much as Presence, Synthetic Phase 6 Generators (make_phase6_data.py, 5,000 blocks), Limitation: 40.19M Is the Last Gain Available from Scale in This Project

## Knowledge Gaps
- **70 isolated node(s):** `How to run`, `Interpreting results`, `Known limitations of this evaluator`, `Methodology — `eval.py``, `Why this exists` (+65 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Repository file map` connect `Transformer Architecture & Inference` to `Pretraining, Scale & Known Bugs`, `Phase 2 Data Generator`, `Model Card & v0 Lineage`, `BPE Tokenizer & Tests`, `Fine-tune Data Lineage & Method`, `Web Server & Endpoints`, `Web UI & Phase 8 Findings`, `Eval Answer Checkers`, `Eval Dataset Generator & Integrity`, `Phase 3b Data Generator`, `Phase 6 Data Generator`, `Phase 3 Trainer`?**
  _High betweenness centrality (0.473) - this node is a cross-community bridge._
- **Why does `Aizen Single-Screen Web UI` connect `Web UI & Phase 8 Findings` to `Transformer Architecture & Inference`?**
  _High betweenness centrality (0.173) - this node is a cross-community bridge._
- **Why does `AIZEN.md — Complete Model Card & Build Reference` connect `Model Card & v0 Lineage` to `Transformer Architecture & Inference`, `Pretraining, Scale & Known Bugs`, `Phase Index & Eval Docs`?**
  _High betweenness centrality (0.141) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `BPETokenizer` (e.g. with `main()` and `load()`) actually correct?**
  _`BPETokenizer` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `How to run`, `Interpreting results`, `Known limitations of this evaluator` to the rest of the system?**
  _70 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Transformer Architecture & Inference` be split into smaller, more focused modules?**
  _Cohesion score 0.05454545454545454 - nodes in this community are weakly interconnected._
- **Should `Pretraining, Scale & Known Bugs` be split into smaller, more focused modules?**
  _Cohesion score 0.06565656565656566 - nodes in this community are weakly interconnected._