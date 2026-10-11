# Third-party licenses & model-weight terms

This repository's own code is MIT (see `LICENSE`). It orchestrates external models/tools that
carry their **own** licenses and weight terms. Weights are never bundled in git — they are
fetched/placed under `data/weights/` on setup (see `data/weights/README.md`). Verify the
version-specific terms for anything you deploy commercially.

| Component | Code license | Weights | Commercial use | Notes |
|-----------|--------------|---------|----------------|-------|
| **This repo** | MIT | n/a | yes | `LICENSE` |
| RFdiffusion3 (RosettaCommons/foundry) | BSD-3-Clause | open (`foundry install rfd3`) | yes* | *foundry BSD-3 text does not explicitly name weights — confirm with IPD for commercial |
| RFdiffusion / RFdiffusionAA | BSD (covers weights) | open | yes | |
| ProteinMPNN / LigandMPNN / SolubleMPNN | MIT | open (in-repo) | yes | vendored under `packages/proteinmpnn` |
| **Boltz-1 / Boltz-2** (default backend) | MIT | open (auto-download) | **yes** | the open default; `aapg.toolkit.boltz` |
| BoltzGen | MIT | open (HF) | likely yes | confirm the weights license line |
| **Protenix v1** | Apache-2.0 | open | yes | use via `model_name=protenix_base_default_v1.0.0` |
| **Protenix v2** | (repo Apache-2.0) | **PROPRIETARY** | **no redistribution** | licensed users place weights in `data/weights/protenix_v2/`; auto-used by `structure_auto` |
| PyRosetta / Rosetta | dual: free non-commercial / paid commercial | binaries gated | paid UW license for commercial | optional `rosetta_interface` stage; user-installed |
| ESM-2 | MIT | open | yes | optional sequence scorer |
| ESM-3 (open weights) | Cambrian Non-Commercial | gated | **non-commercial only** | avoid for commercial; use ESM-2 |
| ipSAE (DunbrackLab/IPSAE) | MIT | n/a (metric) | yes | reimplemented in `aapg.toolkit.ipsae` |
| NetSolP-1.0 | BSD-3 (code) | DTU download (verify) | code yes; weights unclear | optional developability scorer |
| IEDB | data ~CC BY 4.0 (confirm); tools Non-Profit OSL 3.0 | data open | data yes w/ citation; tools NC | optional immunogenicity scorer |

## Default = fully open

Out of the box the pipeline uses **Boltz-2 (MIT)** for structure/ipSAE scoring, **RFdiffusion3
(BSD-3)** for generation, and **ProteinMPNN (MIT)** for sequences — all open with
commercially-usable weights. The only non-open pieces are **opt-in**: Protenix v2 (licensed
weights, auto-detected if present), PyRosetta (optional interface stage; academic-free /
commercial-paid), and ESM-3 (non-commercial; not used by default).

Sources: the upstream repos' `LICENSE` files and model cards (RosettaCommons/foundry,
dauparas/ProteinMPNN, jwohlwend/boltz, HannesStark/boltzgen, bytedance/Protenix,
facebookresearch/esm, evolutionaryscale/esm, DunbrackLab/IPSAE, RosettaCommons download page).
