import os
import json
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Generator

from flask import Flask, request, Response, send_file, jsonify
from dotenv import load_dotenv

from main import run_pipeline, slugify, create_comic_layout, PANEL_SIZE, GAIC, API_KEY, FAL_API_KEY, PromptLogger, extract_characters, extract_scenes, panelize_story, audit_and_patch_cast, patch_panels_for_entities, validate_panel_speakers, build_char_ref_prompt, build_scene_ref_prompt, build_panel_base_instruction, pad_to_square, image_bytes_to_pil, pil_to_png_bytes, resize_to_fill, add_text_rectangles_to_panel, generate_comic_title, create_comic_layout, ensure_dir, USE_COMBINED_PANELS, generate_story_from_user_input

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
        # Cache for character reference images
        self.name_to_ref_bytes: Dict[str, bytes] = {}
        # Mapping from filename to character name
        self.filename_to_char_name: Dict[str, str] = {}
        # Event to control pipeline flow - wait for user to continue after sprite upload
        self.continue_generation_event = threading.Event()


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
        # For live rebuilding, use a simple fallback without AI generation
        # Create a basic title panel without GAIC/logger dependencies
        from PIL import Image, ImageDraw, ImageFont
        import io

        # Simple fallback title panel for live updates
        title = "Generating…"
        canvas = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), "white")
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.load_default()
        except:
            font = None

        if font:
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            title_height = title_bbox[3] - title_bbox[1]
            title_x = (PANEL_SIZE - title_width) // 2
            title_y = (PANEL_SIZE - title_height) // 2

            padding = 20
            draw.rectangle([title_x - padding, title_y - padding,
                           title_x + title_width + padding, title_y + title_height + padding],
                           fill="black")
            draw.text((title_x, title_y), title, fill="white", font=font)

        # Convert to bytes
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        title_panel_bytes = buf.getvalue()

        # Combine with panels manually
        all_panels = [title_panel_bytes] + panel_bytes_list

        # Create layout manually (simplified version)
        num_panels = len(all_panels)
        cols = 2
        rows = (num_panels + 1) // 2

        border_width = 8
        panel_spacing = 12
        margin = 20

        canvas_width = cols * PANEL_SIZE + \
            (cols - 1) * panel_spacing + 2 * margin
        canvas_height = rows * PANEL_SIZE + \
            (rows - 1) * panel_spacing + 2 * margin

        final_canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

        for i, panel_bytes in enumerate(all_panels):
            row = i // cols
            col = i % cols

            x = margin + col * (PANEL_SIZE + panel_spacing)
            y = margin + row * (PANEL_SIZE + panel_spacing)

            panel_img = Image.open(io.BytesIO(panel_bytes))
            panel_img = panel_img.resize(
                (PANEL_SIZE, PANEL_SIZE), Image.LANCZOS)

            draw_final = ImageDraw.Draw(final_canvas)
            draw_final.rectangle([x - border_width//2, y - border_width//2,
                                 x + PANEL_SIZE + border_width//2, y + PANEL_SIZE + border_width//2],
                                 fill="black")

            final_canvas.paste(panel_img, (x, y))

        final_canvas.save(run_dir / "comic_final.png", "PNG")
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
                    # Notify frontend that comic was rebuilt
                    events.put({"type": "comic_rebuilt"})
            if payload["files"]:
                events.put(payload)

            # Emit lightweight assets listing even before manifest exists
            chars_dir = run_dir / "characters"
            scenes_dir = run_dir / "scenes"

            # Get valid PNG files that are fully written and readable
            def get_valid_png_files(directory):
                if not directory.exists():
                    return []
                valid_files = []
                for p in directory.glob("*.png"):
                    try:
                        # Verify file is fully written by attempting to read a few bytes
                        with open(p, 'rb') as f:
                            # Check PNG header
                            header = f.read(8)
                            if len(header) == 8 and header.startswith(b'\x89PNG\r\n\x1a\n'):
                                valid_files.append(str(p.relative_to(run_dir)))
                    except (OSError, IOError):
                        # File not ready yet, skip it
                        continue
                return sorted(valid_files)

            char_files = get_valid_png_files(chars_dir)
            scene_files = get_valid_png_files(scenes_dir)

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
                # Rebuild comic when new assets are available (they may be used in existing panels)
                rebuild_live_comic(run_dir)
                events.put({"type": "comic_rebuilt"})
            time.sleep(0.5)
        except Exception as e:
            events.put({"type": "error", "message": str(e)})
            time.sleep(1)


def run_pipeline_with_events(story_text: str, out_root: Path, events: "queue.Queue[Dict[str, Any]]", user_input: str = None):
    """Run the pipeline with step progress events."""
    ensure_dir(out_root)
    logger = PromptLogger(out_root / "prompts_used.txt")
    g = GAIC(API_KEY)

    # Step 0: Generate story if needed
    if user_input:
        events.put({"type": "step", "step": "story_generation",
                   "message": "Generating story from your request..."})
        story_text = generate_story_from_user_input(g, user_input, logger)
        events.put({"type": "step_complete", "step": "story_generation",
                   "message": f"Story generated ({len(story_text)} characters)"})

    # Step 0.5: Generate comic title early
    events.put({"type": "step", "step": "title_generation",
               "message": "Generating comic title..."})
    comic_title = generate_comic_title(g, story_text)
    logger.log("COMIC_TITLE_GENERATION", f"Generated title: {comic_title}")
    events.put({"type": "step_complete", "step": "title_generation",
               "message": f"Title generated: {comic_title}"})

    # Step 1: Extract characters
    events.put({"type": "step", "step": "characters",
               "message": "Extracting characters..."})
    characters = extract_characters(g, story_text, logger)
    characters = audit_and_patch_cast(story_text, characters)
    events.put({"type": "step_complete", "step": "characters",
               "count": len(characters)})

    # Step 2: Extract scenes
    events.put({"type": "step", "step": "scenes",
               "message": "Extracting scenes..."})
    scenes = extract_scenes(g, story_text, logger)
    events.put({"type": "step_complete",
               "step": "scenes", "count": len(scenes)})

    # Panelize story (part of scenes step)
    panels = panelize_story(g, story_text, scenes, logger)
    patch_panels_for_entities(panels, characters)

    # Validate speakers
    cast_names = [c.name for c in characters]
    all_errors = []
    for panel in panels:
        errors = validate_panel_speakers(panel, cast_names)
        all_errors.extend(errors)

    # Create directories
    chars_dir = out_root / "characters"
    ensure_dir(chars_dir)
    scenes_dir = out_root / "scenes"
    ensure_dir(scenes_dir)
    panels_dir = out_root / "panels"
    ensure_dir(panels_dir)

    # Step 3: Generate character references
    events.put({"type": "step", "step": "character_refs",
               "message": "Generating character references..."})
    state.name_to_ref_bytes = {}  # Reset the cache for this run
    for i, c in enumerate(characters):
        events.put({"type": "step_progress", "step": "character_refs",
                   "current": i + 1, "total": len(characters), "item": c.name})
        prompt = build_char_ref_prompt(c)
        logger.log(f"CHARACTER_REF_NANO_PROMPT [{c.name}]", prompt)
        raw = g.generate_image_with_nano(prompt)
        img = pad_to_square(image_bytes_to_pil(raw), PANEL_SIZE)
        png = pil_to_png_bytes(img)
        fname = f"{slugify(c.name)}.png"
        (chars_dir / fname).write_bytes(png)
        state.name_to_ref_bytes[c.name.lower()] = png

    events.put({"type": "step_complete", "step": "character_refs",
               "count": len(characters)})

    # Trigger sprite upload window with character info
    character_sprites = [
        {"name": c.name, "filename": f"{slugify(c.name)}.png"} for c in characters]
    # Store filename to character name mapping for later use
    state.filename_to_char_name = {
        f"{slugify(c.name)}.png": c.name.lower() for c in characters}
    events.put({"type": "sprite_upload_ready",
               "characters": character_sprites})

    # Wait for user to continue generation (or timeout after 10 minutes)
    events.put({"type": "waiting_for_user",
               "message": "Waiting for user to continue generation..."})
    state.continue_generation_event.clear()  # Reset the event
    # 10 minute timeout
    if not state.continue_generation_event.wait(timeout=600):
        events.put({"type": "step", "step": "user_timeout",
                   "message": "Continuing automatically after timeout..."})
    else:
        events.put({"type": "step", "step": "user_continued",
                   "message": "User chose to continue generation"})

    # CRITICAL: Reload all character images from disk RIGHT AFTER continue/timeout
    # This ensures any uploaded custom sprites are loaded before ANY further generation
    events.put({"type": "step", "step": "reloading_characters",
               "message": "Loading character sprites from disk..."})

    for c in characters:
        fname = f"{slugify(c.name)}.png"
        char_file = chars_dir / fname
        if char_file.exists():
            # Reload the character image from disk (may be user-uploaded)
            char_png = char_file.read_bytes()
            state.name_to_ref_bytes[c.name.lower()] = char_png

    events.put({"type": "step_complete", "step": "reloading_characters",
               "message": "Character sprites reloaded from disk"})

    # Step 4: Generate scene references
    events.put({"type": "step", "step": "scene_refs",
               "message": "Generating scene references..."})
    scene_to_ref_bytes = {}
    for i, s in enumerate(scenes):
        events.put({"type": "step_progress", "step": "scene_refs",
                   "current": i + 1, "total": len(scenes), "item": s.name})
        prompt = build_scene_ref_prompt(s)
        logger.log(f"SCENE_REF_NANO_PROMPT [{s.name}]", prompt)
        raw = g.generate_image_with_nano(prompt)
        img = pad_to_square(image_bytes_to_pil(raw), PANEL_SIZE)
        png = pil_to_png_bytes(img)
        fname = f"{slugify(s.name)}.png"
        (scenes_dir / fname).write_bytes(png)
        scene_to_ref_bytes[s.name.lower()] = png

    events.put({"type": "step_complete",
               "step": "scene_refs", "count": len(scenes)})

    # Step 5: Generate panels
    events.put({"type": "step", "step": "panels",
               "message": "Generating comic panels..."})

    panel_manifest = []
    previous_panel_bytes = None

    for i, p in enumerate(panels):
        events.put({"type": "step_progress", "step": "panels", "current": i +
                   1, "total": len(panels), "item": f"Panel {p.index}"})

        # Get scene reference for this panel
        scene_ref_bytes = scene_to_ref_bytes.get(p.sceneName.lower())
        if not scene_ref_bytes:
            scene_ref_bytes = list(scene_to_ref_bytes.values())[
                0] if scene_to_ref_bytes else None

        if not scene_ref_bytes:
            raise RuntimeError(
                f"No scene reference available for panel {p.index}")

        # Gather ref images for characters in this panel
        char_ref_imgs = [state.name_to_ref_bytes[n.lower()]
                         for n in p.characterNames if n.lower() in state.name_to_ref_bytes]

        # Add previous panel for continuity if available
        ref_images_with_continuity = char_ref_imgs.copy()
        if previous_panel_bytes:
            ref_images_with_continuity.insert(0, previous_panel_bytes)

        # Generate base panel
        base_instruction = build_panel_base_instruction(p, characters)
        logger.log(f"PANEL_BASE_NANO_PROMPT [#{p.index}]", base_instruction)

        try:
            base_png = g.edit_image_with_nano(
                scene_ref_bytes, base_instruction, ref_images=ref_images_with_continuity)
            base_img = resize_to_fill(image_bytes_to_pil(base_png), PANEL_SIZE)
            base_png = pil_to_png_bytes(base_img)
        except Exception as e:
            base_img = resize_to_fill(
                image_bytes_to_pil(scene_ref_bytes), PANEL_SIZE)
            base_png = pil_to_png_bytes(base_img)

        # Save base image
        base_fname = f"base-panel-{p.index:03d}.png"
        (panels_dir / base_fname).write_bytes(base_png)

        # Add text if needed
        if p.dialogue or p.narration:
            base_img_pil = image_bytes_to_pil(base_png)
            text_img_pil = add_text_rectangles_to_panel(base_img_pil, p)
            text_png = pil_to_png_bytes(text_img_pil)

            text_fname = f"text-panel-{p.index:03d}.png"
            (panels_dir / text_fname).write_bytes(text_png)

            final_img = text_img_pil
            final_png = text_png
        else:
            final_img = base_img
            final_png = base_png

        # Save final panel image
        fname = f"panel-{p.index:03d}.png"
        (panels_dir / fname).write_bytes(final_png)

        # Convert structured dialogue back to simple format for manifest
        dialogue_strings = [f"{d.speaker}: {d.text}" for d in p.dialogue]

        # Determine text file name if dialogue/narration exists
        text_file = None
        if p.dialogue or p.narration:
            text_file = f"panels/text-panel-{p.index:03d}.png"

        panel_manifest.append({
            "index": p.index,
            "file": f"panels/{fname}",
            "base_file": f"panels/{base_fname}",
            "text_file": text_file,
            "prompt": p.prompt,  # Add the visual description prompt
            "dialogue": dialogue_strings,
            "narration": p.narration,
            "characters": p.characterNames,
            "scene": p.sceneName,
            "perspective": p.perspective,
            "sfx": p.sfx,
            "visualCues": p.visualCues
        })

        # Store this panel for continuity in next panel
        previous_panel_bytes = final_png

    # Create pages
    panel_manifest.sort(key=lambda x: x["index"])
    pages = []
    for i in range(0, len(panel_manifest), 6):
        pages.append({
            "index": i // 6 + 1,
            "panels": panel_manifest[i:i+6]
        })

    # Create manifest
    manifest = {
        "meta": {"panel_size": PANEL_SIZE},
        "characters": [{"name": c.name, "kind": c.kind, "aliases": c.aliases, "appearance": c.appearance, "summary": c.summary, "scenePresence": [{"sceneName": sp.sceneName, "isPresent": sp.isPresent, "action": sp.action} for sp in c.scenePresence], "file": f"characters/{slugify(c.name)}.png"} for c in characters],
        "scenes": [{"name": s.name, "description": s.description, "setting": s.setting, "file": f"scenes/{slugify(s.name)}.png"} for s in scenes],
        "panels": panel_manifest,
        "pages": pages,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate final comic
    events.put({"type": "step", "step": "final",
               "message": "Creating final comic layout..."})
    # Title was already generated earlier

    # Collect all final panel images
    final_panel_bytes = []
    for panel_info in panel_manifest:
        panel_path = out_root / panel_info["file"]
        if panel_path.exists():
            final_panel_bytes.append(panel_path.read_bytes())

    if final_panel_bytes:
        comic_layout = create_comic_layout(
            final_panel_bytes, comic_title, g, logger, PANEL_SIZE)
        comic_path = out_root / "comic_final.png"
        comic_layout.save(comic_path, "PNG")

    # Save all prompts used
    logger.flush()
    events.put({"type": "step_complete", "step": "final",
               "message": "Comic generation complete!"})


def pipeline_worker(story_text: str, run_dir: Path, events: "queue.Queue[Dict[str, Any]]", stop_flag: threading.Event, user_input: str = None):
    try:
        # Emit start event
        events.put({"type": "start", "run": str(run_dir.name)})
        # Start watcher in same thread loop
        wt = threading.Thread(target=watcher, args=(
            run_dir, events, stop_flag), daemon=True)
        wt.start()

        # Run pipeline with step tracking
        run_pipeline_with_events(story_text, run_dir, events, user_input)
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

    # Handle both story generation and direct story input
    generate_story = data.get("generate_story", False)

    if generate_story:
        user_input = data.get("user_input", "").strip()
        if not user_input:
            return jsonify({"error": "User input required for story generation"}), 400
        story_text = ""  # Will be generated in the pipeline
        slug_source = user_input
    else:
        story_text = data.get("story", "").strip()
        if not story_text:
            return jsonify({"error": "Story text required"}), 400
        user_input = None
        slug_source = story_text

    # Stop any previous run
    if state.thread and state.thread.is_alive():
        state.stop_flag.set()
        state.thread.join(timeout=1)

    # Create run directory
    slug = slugify(slug_source.splitlines()[0] or "story")
    run_id = f"{slug}-{int(time.time())}"
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state.current_run_dir = run_dir
    state.events = queue.Queue()
    state.stop_flag = threading.Event()
    state.continue_generation_event.clear()  # Reset continue event for new run

    # Launch worker
    t = threading.Thread(target=pipeline_worker, args=(
        story_text, run_dir, state.events, state.stop_flag, user_input), daemon=True)
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


@app.route("/api/continue_generation", methods=["POST"])
def api_continue_generation():
    """Signal the pipeline to continue after sprite upload window."""
    data = request.get_json(force=True)
    run = data.get("run")

    if not run:
        return jsonify({"error": "Missing run"}), 400

    run_dir = OUTPUT_DIR / run
    if not run_dir.exists():
        return jsonify({"error": "Run not found"}), 404

    # Only continue if this is the current run
    if state.current_run_dir == run_dir:
        state.continue_generation_event.set()  # Signal the pipeline to continue
        return jsonify({"success": True, "message": "Generation will continue"})
    else:
        return jsonify({"error": "Not the current run"}), 400


@app.route("/api/upload_sprite", methods=["POST"])
def api_upload_sprite():
    """Upload a custom sprite to replace an existing character sprite."""
    run = request.form.get("run")
    filename = request.form.get("filename")

    if not run or not filename:
        return jsonify({"error": "Missing run or filename"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "Only PNG and JPEG files are allowed"}), 400

    run_dir = OUTPUT_DIR / run
    if not run_dir.exists():
        return jsonify({"error": "Run not found"}), 404

    chars_dir = run_dir / "characters"
    target_file = chars_dir / filename

    # Check if the target file exists (must be replacing an existing sprite)
    if not target_file.exists():
        return jsonify({"error": f"Character sprite '{filename}' not found"}), 404

    try:
        # Process the uploaded image to match the expected format
        from PIL import Image
        import io

        # Read and process the uploaded image
        image_data = file.read()
        img = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary (removes alpha channel for JPEG compatibility)
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[-1])
            else:
                background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize to match panel size and pad to square
        img = pad_to_square(img, PANEL_SIZE)

        # Save as PNG
        png_bytes = pil_to_png_bytes(img)
        target_file.write_bytes(png_bytes)

        # Update the in-memory cache with the new sprite
        # Use the filename to character name mapping
        if filename in state.filename_to_char_name:
            char_name = state.filename_to_char_name[filename]
            state.name_to_ref_bytes[char_name] = png_bytes

        # Trigger a comic rebuild to use the new sprite
        if state.current_run_dir == run_dir:
            rebuild_live_comic(run_dir)
            state.events.put({"type": "comic_rebuilt"})
            state.events.put({"type": "sprite_uploaded", "filename": filename})

        return jsonify({"success": True, "message": f"Sprite '{filename}' uploaded successfully"})

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True, threaded=True)
