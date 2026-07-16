# Final Model Experiment Report

Generated: 2026-04-13T12:18:29.089021

## Finalized Configuration
- Model family: xgboost_depth6
- Training regime: weighted_min_confidence
- Target: inferred_joint_label
- Train rows: 45474, Val rows: 9773, Test rows: 11622
- Numeric feature count: 519

## Benchmark Context (Joint Label)
- target=inferred_joint_label, benchmark=unweighted, model=extra_trees_depth35_leaf2, test_f1_macro=0.8343, test_auc_macro_ovr=0.9998, test_balanced_accuracy=0.8037
- target=inferred_joint_label, benchmark=weighted_min_confidence, model=extra_trees_depth25, test_f1_macro=0.8265, test_auc_macro_ovr=0.9998, test_balanced_accuracy=0.8182

## Final Model Metrics (Joint Label)
- target=inferred_joint_label, split=test, accuracy=0.9951, balanced_accuracy=0.7983, f1_macro=0.7842, auc_macro_ovr=0.9993
- target=inferred_joint_label, split=val, accuracy=0.9958, balanced_accuracy=0.8464, f1_macro=0.8383, auc_macro_ovr=0.9998

## Unknown-Label Inference Review
- Unknown rows scored: 12527
- Review file: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/review/unknown_joint_label_inference_review.csv

## Saved Artifacts
- Metric table CSV: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/metrics/final_model_eval_metrics.csv
- Artifact index CSV: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/final_model_artifacts.csv
- [inferred_joint_label] [val] roc_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/roc/roc_micro_inferred_joint_label_val.png
- [inferred_joint_label] [val] pr_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/precision_recall/pr_micro_inferred_joint_label_val.png
- [inferred_joint_label] [test] confusion_matrix: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/confusion/confusion_top15_inferred_joint_label_test.png
- [inferred_joint_label] [test] roc_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/roc/roc_micro_inferred_joint_label_test.png
- [inferred_joint_label] [test] pr_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/precision_recall/pr_micro_inferred_joint_label_test.png

## Finalization Notes
- Final model choice fixed to xgboost_depth6 to simplify deployment and maintenance.
- Workflow narrowed to joint-label prediction only, as primary project objective.
- Weighted training retained because it improved minority-class behavior in benchmark diagnostics.
- PR diagnostics are emphasized for imbalance-sensitive evaluation; ROC retained as a secondary view.