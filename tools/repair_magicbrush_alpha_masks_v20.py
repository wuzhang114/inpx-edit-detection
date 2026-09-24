"""Repair MagicBrush GT from official RGBA alpha masks; reuse predictions.

Official author conversion (issue #10): image_bw = (alpha_channel != 255).
Original masks, outputs and extraction script are copied to an immutable backup.
No training, image inference, or other dataset metrics are changed.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "outputs/magicbrush_alpha_repair_2026-09-20"
ISSUE_API = "https://api.github.com/repos/osu-nlp-group/MagicBrush/issues/10/comments"
AUTHOR_URL = "https://github.com/OSU-NLP-Group/MagicBrush/issues/10#issuecomment-1924847468"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def main():
    # Fetch and verify the author's concrete conversion rule before mutations.
    comments_raw = subprocess.check_output(["curl.exe", "-sSL", "--max-time", "30", ISSUE_API])
    comments = json.loads(comments_raw)
    author = next(x for x in comments if x["html_url"] == AUTHOR_URL)
    assert author["user"]["login"] == "drogozhang" and author["author_association"] == "MEMBER"
    assert "image_bw = (alpha_channel != 255).astype(np.uint8)" in author["body"]
    if (BACKUP / "repair_report.json").exists():
        raise RuntimeError("Repair already completed; inspect the report rather than overwriting the backup.")
    BACKUP.mkdir(exist_ok=True)
    mask_dir = ROOT / "data/magicbrush_eval/masks"
    summary_path = ROOT / "outputs/external_final.json"
    scores_path = ROOT / "outputs/external_final_magicbrush_scores.npz"
    old_summary = json.loads(summary_path.read_text())
    with np.load(scores_path) as z:
        arrays = {k: z[k].copy() for k in z.files}
    for path in [summary_path, scores_path, ROOT / "src/evaluation/extract_magicbrush.py"]:
        dest = BACKUP / path.name
        assert not dest.exists(), str(dest)
        shutil.copy2(path, dest)
    assert not (BACKUP / "masks_before").exists()
    shutil.copytree(mask_dir, BACKUP / "masks_before")
    (BACKUP / "official_issue10_comments.json").write_bytes(comments_raw)
    image_names = {Path(x).stem for x in arrays["paths"]}
    gt_by_name, records = {}, []
    for shard in range(1,5):
        parquet = ROOT / f"data/magicbrush_dev_{shard}.parquet"
        table = pq.read_table(parquet, columns=["img_id","turn_index","mask_img","instruction"])
        for row in table.to_pylist():
            stem = f"{shard:02d}_{row['img_id']}_{row['turn_index']}"
            assert stem in image_names and stem not in gt_by_name
            raw = Image.open(io.BytesIO(row["mask_img"]["bytes"]))
            assert raw.mode == "RGBA", (stem,raw.mode)
            original_bad = np.asarray(raw.convert("RGB").convert("L"))
            old_path = BACKUP / "masks_before" / f"{stem}.png"
            np.testing.assert_array_equal(original_bad,np.asarray(Image.open(old_path)))
            corrected = Image.fromarray((np.asarray(raw)[:,:,3] != 255).astype(np.uint8)*255)
            corrected.save(mask_dir / f"{stem}.png")
            grid = np.asarray(corrected.resize((37,37),Image.Resampling.NEAREST)).ravel()>127
            gt_by_name[stem] = grid
            records.append({"stem":stem,"parquet":str(parquet),"source_mask":row["mask_img"]["path"],
                            "instruction":row["instruction"],"original_mode":raw.mode,
                            "old_mask_sha256":digest(old_path),"new_mask_sha256":digest(mask_dir/f"{stem}.png"),
                            "gt_positive_patches":int(grid.sum())})
    assert len(gt_by_name) == 528
    new_masks = np.stack([gt_by_name[Path(p).stem] for p in arrays["paths"]])
    old_masks = arrays["masks"].copy()
    arrays["masks"] = new_masks
    # Use the exact original evaluator, which only needs CPU for these metrics.
    from external_final import metrics
    updated = copy.deepcopy(old_summary)
    diffs = {}
    heads = old_summary["datasets"]["magicbrush"]["heads"]
    assert set(heads) == {"g2_s42","g2u_s42"}
    for tag, old in heads.items():
        reproduced = metrics(arrays[tag],old_masks,old["val_thr"])
        for field,value in old.items():
            np.testing.assert_allclose(reproduced[field],value,atol=1e-12)
        new = metrics(arrays[tag],new_masks,old["val_thr"])
        updated["datasets"]["magicbrush"]["heads"][tag] = new
        diffs[tag] = {k:{"old":old[k],"new":new[k],"delta":new[k]-old[k]} for k in old}
    note = {"rule":"edited = RGBA alpha != 255 (official author code)","source":AUTHOR_URL,
            "report":str(BACKUP / "repair_report.json"),"date":"2026-09-20",
            "predictions_unchanged":True,"masks_replaced":528,
            "prior_error":"RGBA converted to RGB then luminance, discarding alpha and treating source brightness as GT"}
    updated["datasets"]["magicbrush"]["mask_correction"] = note
    updated["datasets"]["magicbrush"]["legacy_parity_passed"] = False
    updated["datasets"]["magicbrush"]["legacy_parity_note"] = "Old g2 statistics used incorrect luminance GT; predictions are identical, corrected GT intentionally changes scores."
    for ds in old_summary["datasets"]:
        if ds != "magicbrush":
            assert updated["datasets"][ds] == old_summary["datasets"][ds]
    np.savez_compressed(scores_path,**arrays)
    summary_path.write_text(json.dumps(updated,indent=2),encoding="utf-8")
    with np.load(scores_path) as actual:
        for key,value in arrays.items():
            np.testing.assert_array_equal(actual[key],value)
    report = {**note,"official_comment":author,"backup_directory":str(BACKUP),"diff":diffs,
              "empty_grid_masks":int((new_masks.sum(1)==0).sum()),
              "old_positive_fraction":float(old_masks.mean()),"new_positive_fraction":float(new_masks.mean()),
              "changed_grid_entries":int((new_masks!=old_masks).sum()),"records":records,
              "historical_results":"Other MagicBrush localization JSONs and copied IMDL masks are historical and uncorrected; do not use them without regeneration.",
              "semantic_note":"Author prose docstring contradicts its executable rule; use explicit alpha!=255 code, independently confirmed by edit instruction and transparent region on the umbrella example."}
    (BACKUP / "repair_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (BACKUP / "README.md").write_text(
        "# MagicBrush alpha-mask correction\n\n"
        "The old extractor discarded the RGBA alpha channel and thresholded the masked image's luminance. "
        "All 528 original mask images are RGBA. The corrected positive class is alpha != 255, following "
        f"[the dataset author's conversion code]({AUTHOR_URL}). No predictions or training were changed.\n\n"
        "masks_before/ preserves all old local GT; external_final.json and external_final_magicbrush_scores.npz "
        "here preserve the old outputs. repair_report.json records per-sample sources and metric differences.\n\n"
        "Other historical MagicBrush result files and IMDL copied masks remain stale; use the corrected "
        "outputs/external_final.json and outputs/external_final_magicbrush_scores.npz for this paper.\n\n"
        + "```json\n" + json.dumps(diffs,indent=2) + "\n```\n",encoding="utf-8")
    print(json.dumps(diffs,indent=2),flush=True)


if __name__ == "__main__":
    main()
