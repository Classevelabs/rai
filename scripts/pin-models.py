#!/usr/bin/env python3
"""Write launchers/models.tsv: the starter models rai-get-model knows by name.

Each model is pinned to one Hugging Face commit, and every file it needs is
listed with its size and SHA-256, so what a user downloads is exactly what was
tested, and a corrupt or swapped file is refused before conversion.

Run it from the repository root when the list changes:

    python3 scripts/pin-models.py

It only reads from huggingface.co. Small files (configs, tokenizers) have no
SHA-256 in the API, so they are downloaded once here and hashed.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

# name, repository, converter arguments, one-line description.
MODELS = [
    ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "",
     "Small and fast. Good for trying RAI out on any computer."),
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "",
     "Recommended. Clear answers at a good speed on a laptop."),
    ("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "",
     "Best answers, but slower. For computers with 8 GB of memory or more."),
]

# The files `rai convert` reads, plus the model's license to keep beside it.
WANTED_EXACT = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "model.safetensors.index.json",
    "LICENSE",
}

ENDPOINT = "https://huggingface.co"
HEADERS = {"User-Agent": "rai-pin-models"}


def get(url):
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def wanted(name):
    return name in WANTED_EXACT or (name.endswith(".safetensors") and "/" not in name)


def main():
    rows = ["# name\trepository\trevision\tfile\tbytes\tsha256\tconvert_args\tdescription"]
    for name, repo, convert_args, description in MODELS:
        info = json.loads(get(f"{ENDPOINT}/api/models/{repo}?blobs=true"))
        if info.get("gated"):
            sys.exit(f"{repo} is gated; a starter model must download without a login")
        revision = info["sha"]
        files = [s for s in info["siblings"] if wanted(s["rfilename"])]
        names = {s["rfilename"] for s in files}
        for required in ("config.json", "tokenizer.json"):
            if required not in names:
                sys.exit(f"{repo} has no {required}")
        if not any(n.endswith(".safetensors") for n in names):
            sys.exit(f"{repo} has no safetensors weights")
        total = 0
        for sibling in sorted(files, key=lambda s: s["rfilename"]):
            path = sibling["rfilename"]
            lfs = sibling.get("lfs")
            if lfs:
                size, digest = lfs["size"], lfs["sha256"]
            else:
                data = get(f"{ENDPOINT}/{repo}/resolve/{revision}/{path}")
                size, digest = len(data), hashlib.sha256(data).hexdigest()
            total += size
            rows.append("\t".join([name, repo, revision, path, str(size), digest,
                                   convert_args, description]))
        print(f"{name}: {repo}@{revision[:12]}, {len(files)} files, {total / 1e9:.2f} GB")
    out = Path(__file__).resolve().parent.parent / "launchers" / "models.tsv"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
