import unittest
from typing import cast

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.find_bids.models.classify.core import ml_prep_pipeline


def _build_series_features() -> pd.DataFrame:
    rows = []
    descriptions = [
        "Ax T1 MPRAGE precontrast",
        "Sag T1 MPRAGE postcontrast",
        "Ax T2 FLAIR",
        "Resting state BOLD MB EPI",
        "DWI trace b0",
        "ADC derived map",
        "ASL perfusion CBF",
        "Cor GRE SWI phase",
    ]
    protocols = [
        "t1_mprage",
        "t1_post_mprage",
        "t2_flair",
        "rsfmri_mb_epi",
        "diff_epi_b0",
        "adc_map",
        "asl_pcasl",
        "gre_swi_phase",
    ]
    sequences = [
        "mprage",
        "mprage",
        "flair",
        "epfid2d",
        "ep_b1000",
        "adc",
        "pcasl",
        "swi",
    ]
    for idx in range(8):
        rows.append(
            {
                "dataset": "demo",
                "study_uid": f"1.2.840.113619.2.55.3.{1000 + idx}",
                "subject_id": "sub-01" if idx < 4 else "sub-02",
                "session_id": "ses-01",
                "series_id": f"ser-{idx:02d}",
                "num_instances": 120 + idx,
                "num_unique_slices": 24,
                "num_volumes": 1 if idx < 3 else 32,
                "field_strength": 3.0,
                "rows": 128,
                "columns": 128,
                "repetition_time": 2000 + idx * 10,
                "echo_time": 30 + idx,
                "inversion_time": 900 if idx < 2 else None,
                "flip_angle": 90,
                "echo_train_length": 12,
                "slice_thickness": 1.0 + 0.1 * (idx % 3),
                "spacing_between_slices": 1.1 + 0.1 * (idx % 3),
                "echo_spacing": 0.6,
                "num_echoes": 1,
                "num_diffusion_volumes": 16 if idx == 4 else 0,
                "num_diffusion_directions": 15 if idx == 4 else 0,
                "num_b0": 1 if idx == 4 else 0,
                "temporal_variation": 0.2,
                "temporal_spacing": None,
                "num_timepoints": None,
                "multiband_factor": 4 if idx == 3 else 1,
                "manufacturer": "siemens",
                "model": "prisma",
                "scanning_sequence": "EP" if idx in {3, 4} else "GR",
                "sequence_variant": "SK",
                "scan_options": "FS",
                "mr_acquisition_type": "3D" if idx < 2 else "2D",
                "phase_encoding_direction": "j",
                "phase_encoding_axis": "y",
                "phase_encoding_polarity": "positive",
                "perfusion_labeling_type": "pcasl" if idx == 6 else None,
                "perfusion_series_type": "cbf" if idx == 6 else None,
                "contrast_agent": None,
                "tr_bucket": "short",
                "te_bucket": "short",
                "is_3D": idx < 2,
                "is_isotropic": idx < 2,
                "has_diffusion": idx == 4,
                "is_original": idx != 5,
                "is_primary": True,
                "is_localizer": False,
                "is_mpr": idx < 2,
                "is_mip": False,
                "is_projection": False,
                "is_reformatted": idx == 5,
                "has_angio": False,
                "has_diffusion_token": idx in {4, 5},
                "has_perfusion_token": idx == 6,
                "is_adc": idx == 5,
                "is_fa": False,
                "is_trace": idx == 4,
                "is_cbf": idx == 6,
                "is_cbv": False,
                "is_magnitude": idx != 7,
                "is_phase": idx == 7,
                "is_real": False,
                "is_imaginary": False,
                "series_description": descriptions[idx],
                "protocol_name": protocols[idx],
                "sequence_name": sequences[idx],
                "voxel_size": "1\\1\\1",
                "matrix_size": "128\\128",
                "pixel_spacing": "1\\1",
                "b_values": "0\\1000" if idx == 4 else None,
                "echo_times": "30",
                "echo_numbers": "1",
                "image_orientation": "1\\0\\0\\0\\1\\0",
                "acquisition_time": float(8 * 3600 + idx * 600),
                "series_time": f"2026-01-01T{8 + idx:02d}:00:00",
                "acquisition_order": float(1_700_000_000 + idx * 60),
                "source_image_sequences": "1.2.840.10008.5.1.4.1.1.4" if idx in {5, 7} else None,
            }
        )
    return pd.DataFrame(rows)


def _build_heuristic_scores() -> pd.DataFrame:
    labels = [
        ("anat", "T1w"),
        ("anat", "T1w"),
        ("anat", "FLAIR"),
        ("func", "bold"),
        ("dwi", "dwi"),
        ("dwi", "ADC"),
        ("perf", "cbf"),
        ("swi", "phase"),
    ]
    rows = []
    for idx, (datatype, suffix) in enumerate(labels):
        rows.append(
            {
                "dataset": "demo",
                "subject_id": "sub-01" if idx < 4 else "sub-02",
                "session_id": "ses-01",
                "series_id": f"ser-{idx:02d}",
                "inferred_datatype": datatype,
                "datatype_confidence": 0.95,
                "inferred_suffix": suffix,
                "suffix_confidence": 0.92,
                "is_derived": idx in {5, 7},
                "derived_confidence": 0.9,
                "label": f"{datatype}/{suffix}",
                "min_confidence": 0.92,
            }
        )
    return pd.DataFrame(rows)


def _build_heuristic_scores_with_unknown() -> pd.DataFrame:
    scores = _build_heuristic_scores()
    scores.loc[0, "inferred_suffix"] = "unknown"
    scores.loc[1, "inferred_datatype"] = "unknown"
    scores.loc[0, "min_confidence"] = 0.1
    scores.loc[1, "min_confidence"] = 0.05
    return scores


def _build_heuristic_scores_with_localizer_unknown_suffix() -> pd.DataFrame:
    scores = _build_heuristic_scores()
    scores.loc[0, "inferred_datatype"] = "localizer"
    scores.loc[0, "datatype_confidence"] = 0.87
    scores.loc[0, "inferred_suffix"] = "unknown"
    scores.loc[0, "suffix_confidence"] = 0.0
    scores.loc[0, "derived_confidence"] = 0.91
    scores.loc[0, "label"] = "localizer_unknown"
    scores.loc[0, "min_confidence"] = 0.0
    scores.loc[1, "inferred_datatype"] = "unknown"
    scores.loc[1, "min_confidence"] = 0.05
    return scores


class MlFeaturePipelineTests(unittest.TestCase):
    def test_pipeline_normalizes_localizer_unknown_suffix_before_filtering(self) -> None:
        sets = ml_prep_pipeline(
            _build_series_features(),
            _build_heuristic_scores_with_localizer_unknown_suffix(),
            random_state=19,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            tfidf_max_features=10,
            tfidf_min_df=1,
            sample_weighting_mode="none",
        )

        combined_y = sets["all"].y.reset_index(drop=True)
        self.assertTrue(
            ((combined_y["inferred_datatype"] == "localizer") & (combined_y["inferred_suffix"] == "localizer")).any()
        )
        self.assertFalse(
            ((combined_y["inferred_datatype"] == "localizer") & (combined_y["inferred_suffix"] == "unknown")).any()
        )

    def test_pipeline_drops_unknown_from_train_val(self) -> None:
        sets = ml_prep_pipeline(
            _build_series_features(),
            _build_heuristic_scores_with_unknown(),
            random_state=19,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            tfidf_max_features=10,
            tfidf_min_df=1,
            sample_weighting_mode="none",
        )

        train_y = sets["train"].y
        val_y = sets["val"].y
        self.assertFalse(train_y["inferred_datatype"].astype("string").str.lower().eq("unknown").any())
        self.assertFalse(train_y["inferred_suffix"].astype("string").str.lower().eq("unknown").any())
        self.assertFalse(val_y["inferred_datatype"].astype("string").str.lower().eq("unknown").any())
        self.assertFalse(val_y["inferred_suffix"].astype("string").str.lower().eq("unknown").any())

    def test_pipeline_keeps_non_localizer_unknown_suffix_unchanged(self) -> None:
        sets = ml_prep_pipeline(
            _build_series_features(),
            _build_heuristic_scores_with_localizer_unknown_suffix(),
            random_state=19,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            tfidf_max_features=10,
            tfidf_min_df=1,
            sample_weighting_mode="none",
            drop_unknown_in_train_val=False,
        )

        all_y = sets["all"].y.reset_index(drop=True)
        self.assertEqual(all_y.loc[1, "inferred_datatype"], "unknown")
        self.assertEqual(all_y.loc[1, "inferred_suffix"], "T1w")

    def test_pipeline_weighted_uses_min_confidence_with_floor(self) -> None:
        sets = ml_prep_pipeline(
            _build_series_features(),
            _build_heuristic_scores(),
            random_state=23,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            tfidf_max_features=8,
            tfidf_min_df=1,
            sample_weighting_mode="min_confidence",
            min_confidence_weight_floor=0.3,
        )

        train_split = sets["train"]
        self.assertIsNotNone(train_split.sample_weight)
        train_w = cast(pd.Series, train_split.sample_weight)
        self.assertTrue(train_w.between(0.3, 1.0).all())
        self.assertTrue((train_w >= 0.92).all())

    def test_pipeline_weighted_recomputes_localizer_min_confidence(self) -> None:
        sets = ml_prep_pipeline(
            _build_series_features(),
            _build_heuristic_scores_with_localizer_unknown_suffix(),
            random_state=19,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            tfidf_max_features=8,
            tfidf_min_df=1,
            sample_weighting_mode="min_confidence",
            min_confidence_weight_floor=0.3,
            drop_unknown_in_train_val=False,
        )

        all_y = sets["all"].y.reset_index(drop=True)
        all_w = cast(pd.Series, sets["all"].sample_weight).reset_index(drop=True)
        localizer_row = ((all_y["inferred_datatype"] == "localizer") & (all_y["inferred_suffix"] == "localizer"))
        self.assertTrue(localizer_row.any())
        self.assertAlmostEqual(float(all_w.loc[localizer_row].iloc[0]), 0.87, places=6)

    def test_pipeline_returns_named_model_ready_splits(self) -> None:
        sets = ml_prep_pipeline(
            _build_series_features(),
            _build_heuristic_scores(),
            random_state=5,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            tfidf_max_features=12,
            tfidf_min_df=1,
        )

        self.assertEqual({"train", "val", "test", "all"}, set(sets))

        train_split = sets["train"]
        self.assertTrue(hasattr(train_split, "X"))
        self.assertTrue(hasattr(train_split, "y"))
        self.assertEqual(train_split.X.shape[0], train_split.y.shape[0])

        train_X = cast(pd.DataFrame, train_split[0])
        train_y = cast(pd.DataFrame, train_split[1])
        assert_frame_equal(train_X, train_split.X)
        assert_frame_equal(train_y, train_split.y)
        assert_frame_equal(cast(pd.DataFrame, train_split[0]), train_split.X)
        assert_frame_equal(cast(pd.DataFrame, train_split[1]), train_split.y)

    def test_pipeline_adds_train_fitted_tfidf_and_drops_raw_text(self) -> None:
        sets = ml_prep_pipeline(
            _build_series_features(),
            _build_heuristic_scores(),
            random_state=7,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            tfidf_max_features=12,
            tfidf_min_df=1,
        )

        train_X, _ = sets["train"]
        val_X, _ = sets["val"]
        test_X, _ = sets["test"]

        tfidf_columns = [col for col in train_X.columns if col.startswith("tfidf_")]
        self.assertTrue(tfidf_columns)
        self.assertTrue(set(tfidf_columns).issubset(val_X.columns))
        self.assertTrue(set(tfidf_columns).issubset(test_X.columns))
        self.assertTrue(any(col.startswith("tfidf_series_description_") for col in tfidf_columns))
        self.assertTrue(any(col.startswith("tfidf_protocol_name_") for col in tfidf_columns))

        for forbidden in [
            "series_description",
            "protocol_name",
            "sequence_name",
            "combined_text",
        ]:
            self.assertNotIn(forbidden, train_X.columns)

    def test_pipeline_adds_leakage_safe_acquisition_features(self) -> None:
        sets = ml_prep_pipeline(
            _build_series_features(),
            _build_heuristic_scores(),
            random_state=11,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            tfidf_max_features=8,
            tfidf_min_df=1,
        )

        train_X, _ = sets["train"]
        self.assertIn("acquisition_hour", train_X.columns)
        self.assertIn("acquisition_order_rank", train_X.columns)
        self.assertIn("acquisition_order_fraction", train_X.columns)
        self.assertIn("source_image_sequences_present", train_X.columns)

        for forbidden in [
            "dataset",
            "study_uid",
            "subject_id",
            "session_id",
            "series_id",
            "acquisition_time",
            "acquisition_order",
            "series_time",
            "source_image_sequences",
            "label",
            "datatype_confidence",
            "suffix_confidence",
        ]:
            self.assertNotIn(forbidden, train_X.columns)

    def test_pipeline_supports_non_confidence_stratified_splits(self) -> None:
        sets = ml_prep_pipeline(
            _build_series_features(),
            _build_heuristic_scores(),
            random_state=13,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            sample_weighting_mode="none",
            drop_unknown_in_train_val=False,
            tfidf_max_features=10,
            tfidf_sequence_max_features=6,
            tfidf_min_df=1,
        )

        train_split = sets["train"]
        train_X = train_split.X
        self.assertIsNone(train_split.sample_weight)
        self.assertEqual(train_X.shape[0], train_split.y.shape[0])

    def test_pipeline_supports_confidence_weighted_training(self) -> None:
        sets = ml_prep_pipeline(
            _build_series_features(),
            _build_heuristic_scores(),
            random_state=17,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            sample_weighting_mode="min_confidence",
            min_confidence_weight_floor=0.2,
            drop_unknown_in_train_val=False,
            tfidf_max_features=10,
            tfidf_sequence_max_features=6,
            tfidf_min_df=1,
        )

        train_split = sets["train"]
        train_X = train_split.X
        self.assertIsNotNone(train_split.sample_weight)
        train_w = cast(pd.Series, train_split.sample_weight)
        self.assertEqual(len(train_X), len(train_w))
        self.assertTrue(train_w.between(0.2, 1.0).all())
        assert_series_equal(cast(pd.Series, train_split.sample_weight), train_w)


if __name__ == "__main__":
    unittest.main()