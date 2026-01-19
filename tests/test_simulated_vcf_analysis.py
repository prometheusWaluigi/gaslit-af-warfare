import json

import numpy as np

import simulate_vcf_analysis as sim


def test_simulate_vcf_data_structure():
    np.random.seed(0)
    vcf_data = sim.simulate_vcf_data()

    assert set(vcf_data.keys()) == {"callset", "variants", "samples", "genotypes"}
    assert vcf_data["samples"].tolist() == ["SAMPLE1"]
    assert len(vcf_data["variants"]["ID"]) == 7
    assert vcf_data["genotypes"].shape == (7, 1, 2)


def test_analyze_risk_genes_outputs():
    np.random.seed(1)
    vcf_data = sim.simulate_vcf_data()
    results = sim.analyze_risk_genes(vcf_data)

    assert set(results["risk_scores"].keys()) == set(sim.RISK_GENES)
    assert 0.0 <= results["fragility_gamma"] <= 1.0
    assert results["risk_category"] in {
        "High Risk - Allostatic Collapse",
        "Moderate Risk - Fragility",
        "Low Risk",
    }
    assert results["analyzed_variants_count"] == len(vcf_data["variants"]["ID"])


def test_generate_heatmap_data():
    risk_results = {
        "risk_scores": {"TNXB": 0.6, "COMT": 0.3},
    }
    heatmap = sim.generate_heatmap_data(risk_results)

    assert heatmap["genes"] == ["TNXB", "COMT"]
    assert heatmap["scores"] == [0.6, 0.3]
    assert heatmap["thresholds"]["risk"] == 0.7


def test_export_risk_profile(tmp_path):
    risk_results = {
        "risk_scores": {"TNXB": 0.6, "COMT": 0.3},
        "fragility_gamma": 0.5,
        "allostatic_lambda": 0.6,
        "allostatic_omega": 0.4,
        "risk_category": "Low Risk",
        "analyzed_variants_count": 2,
        "samples": ["SAMPLE1"],
    }
    output_file = tmp_path / "risk_profile.json"

    saved_path = sim.export_risk_profile(risk_results, str(output_file))
    assert saved_path == str(output_file)

    exported = json.loads(output_file.read_text())
    assert exported["risk_scores"] == risk_results["risk_scores"]
    assert exported["data_source"].startswith("Simulated VCF analysis")
