"""Independently check delivered summary arithmetic without raw recordings."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/review_corrected_evaluation"
RC = STUDY / "results/review_controls"


def main():
    manifest = json.loads((STUDY / "VERIFICATION.json").read_text())
    for document in manifest["documents"]:
        path = ROOT / document["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == document["sha256"], path
        assert len(PdfReader(path).pages) == document["pages"], path
    assert len(PdfReader(ROOT / "output/pdf/Interception_Movements_Final_Scientific_Report.pdf").pages) == 8
    metrics = ["mean_ks", "energy_distance", "mmd_rbf"]
    raw = pd.concat([pd.read_csv(p) for p in sorted((RC / "fingerprint").glob("z*.csv"))])
    assert len(raw) == 1344
    assert len(raw[["latent_dim", "outer_fold", "seed"]].drop_duplicates()) == 24
    seed = raw.groupby(["latent_dim", "seed", "subject", "arm"])[metrics].mean().reset_index()
    person = seed.groupby(["latent_dim", "subject", "arm"])[metrics].mean().reset_index()
    summary = person.groupby(["latent_dim", "arm"])[metrics].mean().sort_index()
    published = pd.read_csv(RC / "fingerprint_summary.csv").set_index(["latent_dim", "arm"]).sort_index()
    np.testing.assert_allclose(summary, published[metrics], atol=1e-12)
    paired = pd.read_csv(RC / "fingerprint_paired.csv")
    assert len(paired) == 12 and paired.n_subjects.eq(28).all()
    assert paired.control_minus_own.gt(0).all() and paired.p_fdr_bh.lt(.0004).all()
    # Verify corrections independently from stored raw p-values; reranking under
    # a different SciPy method='auto' version is deliberately a separate task.
    from scipy.stats import false_discovery_control
    np.testing.assert_allclose(false_discovery_control(paired.wilcoxon_p), paired.p_fdr_bh, atol=1e-12)
    for row in paired.itertuples():
        p = person[person.latent_dim == row.latent_dim].pivot(index="subject", columns="arm", values=row.metric)
        diff = p[row.control] - p["own"]
        np.testing.assert_allclose(diff.mean(), row.control_minus_own, atol=1e-12)
        assert int((diff > 0).sum()) == row.own_better_n
    direct = pd.read_csv(RC / "direct_context_summary.csv")
    for dim in (3, 8):
        p = direct[direct.latent_dim == dim].pivot(index="target", columns="arm", values="mae")
        assert len(p) == 14 and (p.direct_context < p.cvae_probe).all()
    components = pd.read_csv(RC / "matched_components.csv")
    assert components.job_id.is_unique and len(components) == 4056
    assert components.mj_fit_completed.all() and components.mj_selected_optimizer_converged.all()
    assert components.kind.value_counts().to_dict() == {"recorded": 2376, "generated": 1680}
    raw_summary = pd.read_csv(RC / "matched_component_participant_raw.csv")
    cols = [c for c in raw_summary if c.startswith(("ks_", "count_"))]
    derived = raw_summary.groupby(["latent_dim", "subject"])[cols].mean().groupby("latent_dim").mean()
    published = pd.read_csv(RC / "matched_component_summary.csv").set_index("latent_dim")
    np.testing.assert_allclose(derived[cols], published[cols], atol=1e-12)
    reference = pd.read_csv(RC / "matched_component_sampling.csv")
    assert len(reference) == 14 and (reference.observed > reference.reference_95).all()
    print("PASS: 3 PDF hashes/page counts; 24 fingerprint runs and 12 contrasts; 28 direct-context comparisons; 4,056 fits and component summaries.")
    print("These checks verify the delivered snapshot, not raw-data reproduction or scientific validity.")


if __name__ == "__main__":
    main()
