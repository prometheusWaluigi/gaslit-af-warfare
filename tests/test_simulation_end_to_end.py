import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "gaslit_runner",
        "gaslit-af-runner.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_config(base_dir, *, time_steps, simulation_steps, seed):
    output_root = Path(base_dir)
    return {
        "biological": {
            "grid_size": 10,
            "time_steps": time_steps,
            "dt": 0.01,
            "noise_strength": 0.1,
            "diffusion_constant": 1.0,
            "reaction_rate": 1.0,
            "coupling_strength": 0.5,
            "initial_condition": "random",
            "boundary_condition": "periodic",
            "use_hardware_acceleration": False,
            "phase_portrait_points": 2,
            "phase_portrait_steps": 2,
            "output_dir": str(output_root / "biological"),
            "random_seed": seed,
        },
        "genetic": {
            "output_dir": str(output_root / "genetic"),
            "risk_threshold": 0.7,
            "high_risk_threshold": 0.9,
            "use_hardware_acceleration": False,
            "random_seed": seed,
        },
        "institutional": {
            "output_dir": str(output_root / "institutional"),
            "simulation_steps": simulation_steps,
            "network_size": 5,
            "initial_evidence": 0.1,
            "evidence_growth_rate": 0.01,
            "denial_effectiveness": 0.8,
            "capture_spread_rate": 0.05,
            "random_seed": seed,
            "params": {
                "institutions": [
                    {"name": "CDC", "type": "regulatory", "influence": 0.9, "denial_bias": 0.7},
                    {"name": "FDA", "type": "regulatory", "influence": 0.85, "denial_bias": 0.65},
                    {"name": "NIH", "type": "research", "influence": 0.8, "denial_bias": 0.5},
                    {"name": "Media", "type": "information", "influence": 0.6, "denial_bias": 0.4},
                ]
            },
        },
        "legal": {
            "output_dir": str(output_root / "legal"),
            "simulation_steps": simulation_steps,
            "initial_evidence_level": 0.1,
            "evidence_growth_rate": 0.01,
            "shield_decay_rate": 0.005,
            "random_seed": seed,
            "timeline_start": "2019-01-01",
            "timeline_end": "2020-01-01",
        },
    }


@pytest.mark.parametrize(
    "time_steps,simulation_steps,seed",
    [
        (3, 3, 1),
        (2, 4, 7),
    ],
)
def test_end_to_end_simulation_outputs(tmp_path, time_steps, simulation_steps, seed):
    runner = load_runner_module()
    args = SimpleNamespace(
        module="all",
        use_gpu=False,
        visualize=True,
        save_results=True,
        input_file=None,
    )

    config = build_config(
        tmp_path / f"run-{seed}",
        time_steps=time_steps,
        simulation_steps=simulation_steps,
        seed=seed,
    )

    results = runner.run_modules(args, config)

    assert set(results.keys()) == {"biological", "genetic", "institutional", "legal"}
    assert results["biological"] is not None
    assert results["genetic"] is not None
    assert results["institutional"] is not None
    assert results["legal"] is not None

    biological_dir = Path(config["biological"]["output_dir"])
    genetic_dir = Path(config["genetic"]["output_dir"])
    institutional_dir = Path(config["institutional"]["output_dir"])
    legal_dir = Path(config["legal"]["output_dir"])

    assert (biological_dir / "final_state.png").exists()
    assert (biological_dir / "phase_portrait.png").exists()
    assert any(path.suffix == ".json" for path in biological_dir.iterdir())

    assert (genetic_dir / "sample_heatmap.png").exists()
    assert (genetic_dir / "sample_risk_profile.json").exists()
    assert (genetic_dir / "sample_results.json").exists()

    assert (institutional_dir / "institutional_network.png").exists()
    assert (institutional_dir / "simulation_results.png").exists()
    assert any(path.suffix == ".json" for path in institutional_dir.iterdir())

    assert (legal_dir / "legal_simulation_results.png").exists()
    assert any(path.suffix == ".json" for path in legal_dir.iterdir())
