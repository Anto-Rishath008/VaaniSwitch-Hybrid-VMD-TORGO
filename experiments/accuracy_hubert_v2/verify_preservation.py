"""Compare current protected original/V1 evidence with repair-session baselines."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from src.common import ROOT, preservation_snapshot, read, require, sha, write_json

before=read(ROOT/"logs/protected_before.json")
after=preservation_snapshot()
write_json(ROOT/"logs/protected_after.json",after)
require(before==after,"Original project or prior speaker analysis changed")
v1=ROOT.with_name("accuracy_hubert")
v1_before=read(ROOT/"logs/v1_before.json")
v1_after={p.relative_to(v1).as_posix():{"sha256":sha(p),"size":p.stat().st_size,
                                    "mtime_ns":p.stat().st_mtime_ns}
          for p in v1.rglob("*") if p.is_file()}
write_json(ROOT/"logs/v1_after.json",v1_after)
require(v1_before==v1_after,"V1 inventory/content/metadata changed")
report={"status":"PASS","original_files":len(after["original"]),
        "original_content_hashed_files":sum("sha256" in x for x in after["original"].values()),
        "original_corpus_metadata_files":sum("sha256" not in x for x in after["original"].values()),
        "tracked_experimental_files":len(after["experimental_tracked"]),
        "existing_speaker_analysis_files":len(after["speaker_analysis"]),
        "v1_files_sha256_size_mtime_checked":len(v1_after),
        "v1_frozen_run_sha256":sha(v1/"logs/frozen_run.json"),
        "v1_outer_artifacts":[str(p.relative_to(v1)) for p in (v1/"results").rglob("*") if p.is_file()],
        "original_git":after["original_git"],"study_git":after["study_git"],
        "scope":"Original raw/processed corpus: path/size/mtime; original project evidence and all V1 files: SHA256/size/mtime. Selected development source/RP separately hashed."}
require(not report["v1_outer_artifacts"],"Unexpected V1 outer artifact")
write_json(ROOT/"logs/preservation_report.json",report)
print(report)
