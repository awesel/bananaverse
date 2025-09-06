import os
import json
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Generator

from flask import Flask, request, Response, send_file, jsonify
from dotenv import load_dotenv

from main import run_pipeline, slugify, create_comic_layout, PANEL_SIZE

# Load environment variables
load_dotenv()


app = Flask(__name__, static_folder=None)


ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"


class RunState:
    def __init__(self):
        self.current_run_dir: Optional[Path] = None
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()


state = RunState()


def list_new_files(watch_dir: Path, known: set[str]) -> list[Path]:
    found = []
    for p in watch_dir.rglob("*.png"):
        rp = str(p.relative_to(watch_dir))
        if rp not in known:
            known.add(rp)
            found.append(p)
    # manifest
    mp = watch_dir / "manifest.json"
    if mp.exists():
        rp = str(mp.relative_to(watch_dir))
        if rp not in known:
            known.add(rp)
            found.append(mp)
    return found


def rebuild_live_comic(run_dir: Path) -> None:
    panels_dir = run_dir / "panels"
    if not panels_dir.exists():
        return
    panel_files = sorted(panels_dir.glob("panel-*.png"))
    if not panel_files:
        return
    try:
        panel_bytes_list = [p.read_bytes() for p in panel_files]
        live_img = create_comic_layout(
            panel_bytes_list, "Generating…", PANEL_SIZE)
        live_img.save(run_dir / "comic_final.png", "PNG")
    except Exception:
        # Best-effort; ignore transient errors while files are being written
        pass


def watcher(run_dir: Path, events: "queue.Queue[Dict[str, Any]]", stop_flag: threading.Event):
    known: set[str] = set()
    # Prime existing files in case run is already partway
    list_new_files(run_dir, known)
    last_manifest_snapshot: Optional[str] = None
    last_char_count = -1
    last_scene_count = -1
    while not stop_flag.is_set():
        try:
            new_files = list_new_files(run_dir, known)
            payload: Dict[str, Any] = {"type": "progress", "files": []}
            for p in new_files:
                rel = str(p.relative_to(run_dir))
                payload["files"].append(rel)
                if p.name == "manifest.json":
                    try:
                        manifest = p.read_text(encoding="utf-8")
                        if manifest != last_manifest_snapshot:
                            last_manifest_snapshot = manifest
                            events.put(
                                {"type": "manifest", "manifest": json.loads(manifest)})
                    except Exception:
                        pass
                if p.name == "comic_final.png":
                    payload["final"] = True
                # If a panel is added, rebuild live composite asap
                if p.parent.name == "panels" and p.name.startswith("panel-"):
                    rebuild_live_comic(run_dir)
            if payload["files"]:
                events.put(payload)

            # Emit lightweight assets listing even before manifest exists
            chars_dir = run_dir / "characters"
            scenes_dir = run_dir / "scenes"
            char_files = sorted([str(p.relative_to(run_dir)) for p in (
                chars_dir.glob("*.png") if chars_dir.exists() else [])])
            scene_files = sorted([str(p.relative_to(run_dir)) for p in (
                scenes_dir.glob("*.png") if scenes_dir.exists() else [])])
            if len(char_files) != last_char_count or len(scene_files) != last_scene_count:
                last_char_count = len(char_files)
                last_scene_count = len(scene_files)

                def label_from_path(rel_path: str) -> str:
                    name = Path(rel_path).stem.replace("-", " ")
                    return name.title()
                events.put({
                    "type": "assets",
                    "characters": [{"file": f, "name": label_from_path(f)} for f in char_files],
                    "scenes": [{"file": f, "name": label_from_path(f)} for f in scene_files]
                })
            time.sleep(0.5)
        except Exception as e:
            events.put({"type": "error", "message": str(e)})
            time.sleep(1)


def pipeline_worker(story_text: str, run_dir: Path, events: "queue.Queue[Dict[str, Any]]", stop_flag: threading.Event):
    try:
        # Emit start event
        events.put({"type": "start", "run": str(run_dir.name)})
        # Start watcher in same thread loop
        wt = threading.Thread(target=watcher, args=(
            run_dir, events, stop_flag), daemon=True)
        wt.start()
        run_pipeline(story_text, run_dir)
        events.put({"type": "done"})
    except Exception as e:
        events.put({"type": "error", "message": str(e)})
    finally:
        stop_flag.set()


@app.route("/")
def index() -> Response:
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    return Response(html, mimetype="text/html")


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(force=True)
    story_text: str = data.get("story", "").strip()
    if not story_text:
        return jsonify({"error": "Story text required"}), 400

    # Stop any previous run
    if state.thread and state.thread.is_alive():
        state.stop_flag.set()
        state.thread.join(timeout=1)

    # Create run directory
    slug = slugify(story_text.splitlines()[0] or "story")
    run_id = f"{slug}-{int(time.time())}"
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state.current_run_dir = run_dir
    state.events = queue.Queue()
    state.stop_flag = threading.Event()

    # Launch worker
    t = threading.Thread(target=pipeline_worker, args=(
        story_text, run_dir, state.events, state.stop_flag), daemon=True)
    t.start()
    state.thread = t
    return jsonify({"run": run_id})


@app.route("/api/stream")
def api_stream() -> Response:
    def gen() -> Generator[str, None, None]:
        yield "event: ping\n" "data: {}\n\n"
        while True:
            try:
                evt = state.events.get(timeout=60)
            except Exception:
                yield "event: ping\n" "data: {}\n\n"
                continue
            yield f"data: {json.dumps(evt)}\n\n"
            if evt.get("type") in {"done", "error"}:
                break
    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/manifest")
def api_manifest():
    run = request.args.get("run")
    if not run:
        return jsonify({"error": "Missing run"}), 400
    run_dir = OUTPUT_DIR / run
    mf = run_dir / "manifest.json"
    if not mf.exists():
        return jsonify({"manifest": None})
    return jsonify(json.loads(mf.read_text(encoding="utf-8")))


def safe_path(root: Path, rel: str) -> Optional[Path]:
    p = (root / rel).resolve()
    if root.resolve() in p.parents or p == root.resolve():
        return p if p.exists() else None
    return None


@app.route("/api/file")
def api_file():
    run = request.args.get("run")
    rel = request.args.get("path")
    if not run or not rel:
        return "Missing run or path", 400
    run_dir = OUTPUT_DIR / run
    p = safe_path(run_dir, rel)
    if not p:
        return "Not found", 404
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        return send_file(str(p))
    return Response(p.read_text(encoding="utf-8"), mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True, threaded=True)
