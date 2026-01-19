import importlib.util
import logging
from types import SimpleNamespace


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "gaslit_runner",
        "gaslit-af-runner.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_modules_all(tmp_path):
    logging.basicConfig(level=logging.INFO)
    runner = load_runner_module()

    args = SimpleNamespace(
        module="all",
        use_gpu=False,
        visualize=False,
        save_results=False,
        input_file=None,
    )

    config = {
        "biological": {
            "grid_size": 10,
            "time_steps": 3,
            "dt": 0.01,
            "noise_strength": 0.1,
            "diffusion_constant": 1.0,
            "reaction_rate": 1.0,
            "coupling_strength": 0.5,
            "initial_condition": "random",
            "boundary_condition": "periodic",
            "use_hardware_acceleration": False,
            "output_dir": str(tmp_path / "biological"),
            "random_seed": 1,
        },
        "genetic": {
            "output_dir": str(tmp_path / "genetic"),
            "risk_threshold": 0.7,
            "high_risk_threshold": 0.9,
            "use_hardware_acceleration": False,
            "random_seed": 1,
        },
        "institutional": {
            "output_dir": str(tmp_path / "institutional"),
            "simulation_steps": 3,
            "network_size": 5,
            "initial_evidence": 0.1,
            "evidence_growth_rate": 0.01,
            "denial_effectiveness": 0.8,
            "capture_spread_rate": 0.05,
            "random_seed": 1,
            "params": {
                "institutions": [
                    {"name": "CDC", "type": "regulatory", "influence": 0.9, "denial_bias": 0.7},
                    {"name": "FDA", "type": "regulatory", "influence": 0.85, "denial_bias": 0.65},
                    {"name": "NIH", "type": "research", "influence": 0.8, "denial_bias": 0.5},
                ]
            },
        },
        "legal": {
            "output_dir": str(tmp_path / "legal"),
            "simulation_steps": 3,
            "initial_evidence_level": 0.1,
            "evidence_growth_rate": 0.01,
            "shield_decay_rate": 0.005,
            "random_seed": 1,
            "timeline_start": "2019-01-01",
            "timeline_end": "2020-01-01",
        },
    }

    results = runner.run_modules(args, config)

    assert set(results.keys()) == {"biological", "genetic", "institutional", "legal"}
    assert results["biological"] is not None
    assert results["genetic"] is not None
    assert results["institutional"] is not None
    assert results["legal"] is not None
