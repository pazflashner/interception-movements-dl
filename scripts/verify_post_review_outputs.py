"""Audit corrected run coverage/provenance and render delivery PDFs for review."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import fitz
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / "studies/review_corrected_evaluation"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args=parser.parse_args()
    expected={"cvae":48,"conditional_ae":24,"unconditional_vae":24,"spline_pca":16,"condition_ridge":12}
    references={}; inventory=[]; training=[]
    for family,count in expected.items():
        paths=sorted((STUDY/"runs"/family).glob("fold*/*/result.json"))
        assert len(paths)==count,(family,len(paths))
        for path in paths:
            r=json.loads(path.read_text()); folder=path.parent
            source=ROOT/r["source_run"]
            assert digest(source/"result.json")==r["source_result_sha256"]
            if family in {"cvae", "conditional_ae", "unconditional_vae"}:
                config=yaml.safe_load((source/"config.yaml").read_text())
                history=json.loads((source/"history.json").read_text())
                limit=int(config["train"]["epochs"])
                completed=len(history["train_loss"])
                assert completed==len(history["val_loss"])
                assert 0<completed<=limit
                training.append({"run":str(source.relative_to(ROOT)),
                    "epoch_limit":limit,"epochs_completed":completed,"reached_cap":completed==limit})
            assert r["neural_retraining"] is False and r["labels_changed"] is False
            reference=json.loads((folder/"distance_reference.json").read_text())
            fold=int(r["outer_fold"])
            if fold in references:
                assert reference==references[fold],(family,fold,"different distance geometry")
            references[fold]=reference
            assert len(reference["features"])==11
            assert reference["scale"][reference["features"].index("n_submovements")]>=1
            fidelity=pd.read_csv(folder/"context_query_fidelity.csv")
            assert len(fidelity)==7
            assert np.isfinite(fidelity[["energy_distance","mmd_rbf"]].to_numpy()).all()
            # Reconstruction and timing are invariant to the distance-only fix.
            for name in ("timing_predictions.csv","reconstruction_predictions.csv"):
                old,new=pd.read_csv(source/name),pd.read_csv(folder/name)
                assert list(old)==list(new) and len(old)==len(new)
                for col in old:
                    if pd.api.types.is_numeric_dtype(old[col]):
                        np.testing.assert_allclose(old[col],new[col],rtol=1e-5,atol=1e-7,equal_nan=True)
                    else:
                        assert old[col].equals(new[col])
            inventory.append({"run":str(folder.relative_to(ROOT)),"source_result_sha256":r["source_result_sha256"],
                "checkpoint_sha256":digest(source/"checkpoint.pt") if (source/"checkpoint.pt").exists() else None})
    assert len(training)==96
    assert {run["epoch_limit"] for run in training}=={150}
    assert sum(run["reached_cap"] for run in training)==5
    probe=pd.read_csv(STUDY/"results/behavioral_probes/pooled_by_seed.csv")
    assert probe.n_participants.eq(28).all()
    common=pd.read_csv(STUDY/"results/timing_fairness/predictions.csv")
    heads=common[common.comparison=="common_mlp"][["model_family","latent_dim","outer_fold","seed"]].drop_duplicates()
    assert len(heads)==48
    documents=[]
    render=ROOT/"output/pdf/rendered/review_corrected"
    render.mkdir(parents=True,exist_ok=True)
    for path in sorted((ROOT/"output/pdf").glob("*.pdf")):
        doc=fitz.open(path); thumbs=[]
        text="\n".join(p.get_text() for p in doc)
        assert "74.6%" in text and "marker 5)" not in text.lower()
        assert "common" in text.lower() and "negative" in text.lower()
        if path.name=="Interception_Movements_Final_Scientific_Report.pdf":
            normalized=" ".join(text.split())
            assert "150-epoch limit" in normalized and "200-epoch limit" not in normalized
            assert "Five of the 96 neural runs reached this limit." in normalized
        for i,page in enumerate(doc):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines",[]):
                    for span in line["spans"]:
                        rect=fitz.Rect(span["bbox"])
                        assert rect.x0>=20 and rect.x1<=page.rect.width-20,(path.name,i,span["text"])
                        assert rect.y0>=10 and rect.y1<=page.rect.height-8,(path.name,i,span["text"])
            if args.render:
                pix=page.get_pixmap(matrix=fitz.Matrix(1.5,1.5))
                pix.save(render/f"{path.stem}_{i+1}.png")
                im=Image.frombytes("RGB",[pix.width,pix.height],pix.samples); im.thumbnail((595,842))
                thumb=Image.new("RGB",(615,875),"#e5e5e5"); thumb.paste(im,((615-im.width)//2,20))
                ImageDraw.Draw(thumb).text((15,857),str(i+1),fill="black"); thumbs.append(thumb)
        for start in range(0,len(thumbs),4):
            sheet=Image.new("RGB",(1230,1750),"white")
            for j,thumb in enumerate(thumbs[start:start+4]):
                sheet.paste(thumb,((j%2)*615,(j//2)*875))
            sheet.save(render/f"{path.stem}_contact_{start//4+1}.png")
        documents.append({"path":str(path.relative_to(ROOT)),"pages":len(doc),"sha256":digest(path)})
    report={"run_counts":expected,"run_count":len(inventory),"shared_training_references":len(references),
        "prediction_invariance_checked":True,"common_timing_heads":len(heads),"documents":documents,"runs":inventory,
        "neural_training":{"run_count":len(training),"epoch_limit":150,"runs_reaching_cap":5,"runs":training}}
    (STUDY/"VERIFICATION.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    summary={k:v for k,v in report.items() if k not in {"runs","neural_training"}}
    summary["neural_training"]={k:v for k,v in report["neural_training"].items() if k!="runs"}
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
