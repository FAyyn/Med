# Proposed Additional Experiments (for author follow-up)

1. Background randomization
- Description: Keep lesion region fixed, replace background with images sampled from other patients or noise. Measure VQA-RAD closed accuracy and recall for lesion-related questions.
- Dataset: 500–1000 lesion-related test samples from VQA-RAD and SLAKE.
- Metric: Closed accuracy, open recall, and change relative to original inputs.
- Expected cost: evaluation-only, a few hours on a single GPU.

2. Lesion-removal consistency
- Description: For lesion-related queries, compare model predictions on original image, lesion-masked image, and background-only image. Expect CaMedPO answers to degrade when lesion is removed.
- Dataset: 200–500 curated samples with lesion-focused questions.
- Metric: Drop in correct lesion-related answers; statistical test of degradation.
- Expected cost: evaluation-only, a few hours.

3. Prompt ablation (optional)
- Description: Remove or rephrase the joint-query prompt T' and vary composite layout. Measure performance drop to rule out prompt-dependent heuristics.
- Dataset and cost: small subset, low cost.

Notes:
- These tests are diagnostic and do not require retraining. They can be run quickly and added to the supplement.

4. Structured factuality metrics
- Description: Compute CheXbert label F1 and RadGraph entity/relation overlap on existing test sets to quantify clinical factuality beyond n-gram metrics.
- Dataset: full test splits (VQA-RAD, SLAKE, IU-XRay sample for report generation).
- Metric: CheXbert label F1, RadGraph F1, and comparative delta versus BLEU/ROUGE.
- Expected cost: a few hours to one day depending on implementation readiness.

5. Semantic similarity metric
- Description: Compute BERTScore or a medical-domain semantic similarity metric to assess semantic fidelity.
- Expected cost: low.

6. Small human fact-check
- Description: Annotate 200 report outputs for error types: false positive, negation error, wrong location, missing finding.
- Expected cost: 1-2 person-days for annotation and summary.
