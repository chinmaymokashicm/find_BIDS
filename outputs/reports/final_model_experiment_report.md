# Final Model Experiment Report

Generated: 2026-04-12T22:30:04.573624

## Finalized Configuration
- Model family: xgboost_depth6
- Training regime: weighted_min_confidence
- Targets: inferred_datatype, inferred_suffix, inferred_joint_label
- Train rows: 39271, Val rows: 8465, Test rows: 11623
- Numeric feature count: 507

## Benchmark Context (Best Candidate Per Target)
- target=inferred_datatype, benchmark=unweighted, model=xgboost_depth6, test_f1_macro=0.8663, test_auc_macro_ovr=0.9996, test_balanced_accuracy=0.9481
- target=inferred_datatype, benchmark=weighted_min_confidence, model=xgboost_depth6, test_f1_macro=0.8641, test_auc_macro_ovr=0.9996, test_balanced_accuracy=0.9450
- target=inferred_joint_label, benchmark=unweighted, model=extra_trees_depth25_balanced, test_f1_macro=0.8337, test_auc_macro_ovr=0.9879, test_balanced_accuracy=0.8278
- target=inferred_joint_label, benchmark=weighted_min_confidence, model=extra_trees_depth25, test_f1_macro=0.8335, test_auc_macro_ovr=0.9903, test_balanced_accuracy=0.8273
- target=inferred_suffix, benchmark=weighted_min_confidence, model=rf_depth20, test_f1_macro=0.8385, test_auc_macro_ovr=0.9998, test_balanced_accuracy=0.8282
- target=inferred_suffix, benchmark=unweighted, model=extra_trees_depth25_balanced, test_f1_macro=0.8334, test_auc_macro_ovr=0.9852, test_balanced_accuracy=0.8272

## Final Model Metrics
- target=inferred_datatype, split=test, accuracy=0.9972, balanced_accuracy=0.9482, f1_macro=0.8433, auc_macro_ovr=0.9996
- target=inferred_datatype, split=val, accuracy=0.9981, balanced_accuracy=0.9664, f1_macro=0.9765, auc_macro_ovr=0.9995
- target=inferred_joint_label, split=test, accuracy=0.9963, balanced_accuracy=0.8526, f1_macro=0.8539, auc_macro_ovr=0.9999
- target=inferred_joint_label, split=val, accuracy=0.9954, balanced_accuracy=0.8355, f1_macro=0.8546, auc_macro_ovr=0.9993
- target=inferred_suffix, split=test, accuracy=0.9963, balanced_accuracy=0.8547, f1_macro=0.8518, auc_macro_ovr=0.9999
- target=inferred_suffix, split=val, accuracy=0.9955, balanced_accuracy=0.8385, f1_macro=0.8539, auc_macro_ovr=0.9994

## Unknown-Label Inference Review
- Unknown rows scored: 21314
- Review file: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/review/unknown_label_inference_review.csv

## Saved Artifacts
- Metric table CSV: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/metrics/final_model_eval_metrics.csv
- Artifact index CSV: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/final_model_artifacts.csv
- [inferred_datatype] [val] roc_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/roc/roc_micro_inferred_datatype_val.png
- [inferred_datatype] [val] pr_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/precision_recall/pr_micro_inferred_datatype_val.png
- [inferred_datatype] [test] confusion_matrix: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/confusion/confusion_top15_inferred_datatype_test.png
- [inferred_datatype] [test] roc_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/roc/roc_micro_inferred_datatype_test.png
- [inferred_datatype] [test] pr_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/precision_recall/pr_micro_inferred_datatype_test.png
- [inferred_suffix] [val] roc_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/roc/roc_micro_inferred_suffix_val.png
- [inferred_suffix] [val] pr_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/precision_recall/pr_micro_inferred_suffix_val.png
- [inferred_suffix] [test] confusion_matrix: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/confusion/confusion_top15_inferred_suffix_test.png
- [inferred_suffix] [test] roc_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/roc/roc_micro_inferred_suffix_test.png
- [inferred_suffix] [test] pr_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/precision_recall/pr_micro_inferred_suffix_test.png
- [inferred_joint_label] [val] roc_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/roc/roc_micro_inferred_joint_label_val.png
- [inferred_joint_label] [val] pr_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/precision_recall/pr_micro_inferred_joint_label_val.png
- [inferred_joint_label] [test] confusion_matrix: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/confusion/confusion_top15_inferred_joint_label_test.png
- [inferred_joint_label] [test] roc_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/roc/roc_micro_inferred_joint_label_test.png
- [inferred_joint_label] [test] pr_micro: /Users/cmokashi/Documents/GitHub/find_BIDS/outputs/plots/finalized_model/precision_recall/pr_micro_inferred_joint_label_test.png

## Finalization Notes
- Final model choice fixed to xgboost_depth6 to simplify deployment and maintenance.
- Weighted training retained because it improved minority-class behavior in benchmark diagnostics.
- PR diagnostics are emphasized for imbalance-sensitive evaluation; ROC retained as a secondary view.