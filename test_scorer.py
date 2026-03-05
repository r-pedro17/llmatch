"""Unit tests for scorer.py"""
import unittest

from hardware import HardwareProfile
from models import Model
from scorer import (
    _model_memory_gb,
    _best_quant,
    _fit_level,
    _score_quality,
    _score_speed,
    _score_fit,
    _score_context,
    _run_mode,
    score_all,
)


def _make_model(**kwargs):
    defaults = dict(
        name="test-model",
        provider="test",
        parameter_count="7B",
        parameters_raw=7_000_000_000,
        min_ram_gb=4.0,
        recommended_ram_gb=8.0,
        min_vram_gb=4.0,
        quantization="Q4_K_M",
        context_length=4096,
        use_case="general",
        capabilities=["chat"],
        architecture="transformer",
        hf_downloads=1000,
        hf_likes=100,
        release_date="2025-01-01",
    )
    defaults.update(kwargs)
    return Model(**defaults)


def _make_hw(**kwargs):
    defaults = dict(
        ram_gb=32.0,
        cpu_cores=8,
        gpu_vram_gb=24.0,
        backend="cuda",
        gpu_name="RTX 4090",
    )
    defaults.update(kwargs)
    return HardwareProfile(**defaults)


class TestModelMemory(unittest.TestCase):
    def test_standard_model_q4(self):
        model = _make_model(parameters_raw=7_000_000_000)
        mem = _model_memory_gb(model, "Q4_K_M")
        # 7B * 4.8 bits / 8 / 1e9 = 4.2 GB
        self.assertAlmostEqual(mem, 4.2, places=1)

    def test_standard_model_q8(self):
        model = _make_model(parameters_raw=7_000_000_000)
        mem = _model_memory_gb(model, "Q8_0")
        # 7B * 8 bits / 8 / 1e9 = 7.0 GB
        self.assertAlmostEqual(mem, 7.0, places=1)

    def test_moe_model(self):
        model = _make_model(
            parameters_raw=47_000_000_000,
            is_moe=True,
            num_experts=8,
            active_experts=2,
            active_parameters=14_000_000_000,
        )
        mem = _model_memory_gb(model, "Q4_K_M")
        # active: 14B * 4.8/8/1e9 = 8.4 GB
        # inactive: 33B * 2.0/8/1e9 = 8.25 GB
        self.assertGreater(mem, 8.0)
        self.assertLess(mem, 20.0)


class TestBestQuant(unittest.TestCase):
    def test_large_memory_picks_q8(self):
        model = _make_model(parameters_raw=7_000_000_000)
        self.assertEqual(_best_quant(model, 24.0), "Q8_0")

    def test_tight_memory_picks_lower_quant(self):
        model = _make_model(parameters_raw=7_000_000_000)
        quant = _best_quant(model, 5.0)
        self.assertIn(quant, ["Q4_K_M", "Q5_K_M"])

    def test_no_fit_returns_none(self):
        model = _make_model(parameters_raw=70_000_000_000)
        self.assertIsNone(_best_quant(model, 1.0))


class TestFitLevel(unittest.TestCase):
    def test_perfect_range(self):
        self.assertEqual(_fit_level(0.6), "perfect")
        self.assertEqual(_fit_level(0.5), "perfect")
        self.assertEqual(_fit_level(0.8), "perfect")

    def test_good(self):
        self.assertEqual(_fit_level(0.3), "good")

    def test_marginal(self):
        self.assertEqual(_fit_level(0.1), "marginal")

    def test_too_tight(self):
        self.assertEqual(_fit_level(1.1), "too_tight")


class TestScoreQuality(unittest.TestCase):
    def test_zero_params(self):
        model = _make_model(parameters_raw=0)
        self.assertEqual(_score_quality(model, "Q4_K_M"), 0.0)

    def test_higher_params_higher_score(self):
        small = _make_model(parameters_raw=1_000_000_000)
        large = _make_model(parameters_raw=70_000_000_000)
        self.assertGreater(
            _score_quality(large, "Q4_K_M"),
            _score_quality(small, "Q4_K_M"),
        )

    def test_higher_quant_higher_score(self):
        model = _make_model(parameters_raw=7_000_000_000)
        self.assertGreater(
            _score_quality(model, "Q8_0"),
            _score_quality(model, "Q2_K"),
        )


class TestScoreContext(unittest.TestCase):
    def test_zero_context(self):
        self.assertEqual(_score_context(0), 0.0)

    def test_higher_context_higher_score(self):
        self.assertGreater(_score_context(32768), _score_context(4096))

    def test_bounded(self):
        score = _score_context(2**20)
        self.assertLessEqual(score, 100.0)
        self.assertGreaterEqual(score, 0.0)


class TestScoreFit(unittest.TestCase):
    def test_sweet_spot(self):
        self.assertEqual(_score_fit(0.65), 100.0)

    def test_over_capacity(self):
        self.assertEqual(_score_fit(1.5), 0.0)

    def test_decreasing_above_sweet_spot(self):
        self.assertGreater(_score_fit(0.8), _score_fit(0.95))


class TestRunMode(unittest.TestCase):
    def test_moe_model(self):
        model = _make_model(is_moe=True, active_parameters=14_000_000_000)
        hw = _make_hw()
        self.assertEqual(_run_mode(model, "Q4_K_M", hw), "moe")

    def test_gpu_model(self):
        model = _make_model(parameters_raw=7_000_000_000)
        hw = _make_hw(gpu_vram_gb=24.0)
        self.assertEqual(_run_mode(model, "Q4_K_M", hw), "gpu")

    def test_cpu_gpu_model(self):
        model = _make_model(parameters_raw=70_000_000_000)
        hw = _make_hw(gpu_vram_gb=8.0)
        self.assertEqual(_run_mode(model, "Q4_K_M", hw), "cpu+gpu")

    def test_cpu_only(self):
        model = _make_model(parameters_raw=7_000_000_000)
        hw = _make_hw(gpu_vram_gb=0.0, backend="cpu")
        self.assertEqual(_run_mode(model, "Q4_K_M", hw), "cpu")


class TestScoreAll(unittest.TestCase):
    def test_sorted_descending(self):
        models = [
            _make_model(name="small", parameters_raw=1_000_000_000),
            _make_model(name="large", parameters_raw=7_000_000_000),
        ]
        hw = _make_hw()
        scored = score_all(models, hw)
        self.assertEqual(len(scored), 2)
        self.assertGreaterEqual(scored[0].score_total, scored[1].score_total)

    def test_max_context_affects_context_score(self):
        models = [_make_model(context_length=128_000)]
        hw = _make_hw()
        uncapped = score_all(models, hw)
        capped = score_all(models, hw, max_context=4096)
        self.assertGreater(uncapped[0].score_context, capped[0].score_context)

    def test_models_too_large_are_skipped(self):
        models = [_make_model(parameters_raw=500_000_000_000)]
        hw = _make_hw(gpu_vram_gb=8.0, ram_gb=16.0)
        scored = score_all(models, hw)
        self.assertEqual(len(scored), 0)


if __name__ == "__main__":
    unittest.main()
