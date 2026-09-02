"""Exercise every dashboard section at every latent dimension via AppTest."""
from pathlib import Path
import argparse
import json
import tempfile
import sys

import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    torch.set_num_threads(1)
    scratch=Path(__file__).resolve().parents[1]/".tmp/streamlit_delivery_test"
    scratch.mkdir(parents=True,exist_ok=True); tempfile.tempdir=str(scratch)
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(args.root/"src/confirmatory_dashboard.py"),default_timeout=90).run()
    assert not app.exception,list(app.exception)
    for name in ("config", "src.features", "src.vae_model", "src.submovements"):
        assert Path(sys.modules[name].__file__).resolve().is_relative_to(args.root.resolve()), name
    def select(label, value):
        # This installed AppTest version serializes even single-selection
        # segmented controls through an array-valued ButtonGroup widget.
        for widget in app.get("button_group"):
            current = value if widget.label == label else widget.value
            widget.set_value(current if isinstance(current,list) else ([] if current is None else [current]))
        app.run()
    checks=[]
    for n in (2,3,4,8):
        select("Latent dimension", n)
        for section in ("Generate","Held-out validation","Benchmarks","Latent associations","Diagnostics","Protocol & downloads"):
            select("Dashboard section", section)
            assert not app.exception,(n,section,list(app.exception))
            checks.append({"latent_dim":n,"section":section,"exceptions":0})
            print(f"PASS n={n}: {section}",flush=True)
    (scratch/"latest_check.json").write_text(json.dumps({"root":str(args.root),"checks":checks},indent=2))


if __name__=="__main__":
    main()
