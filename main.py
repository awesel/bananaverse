# comic_e2e.py
import os
import io
import re
import sys
import json
import base64
import random
import string
from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont

# Google AI SDK
from google import genai
from google.genai import types

# ------------------ ENV & CONFIG ------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

# Models (override via env if your account uses different names)
# text planning
LLM_MODEL = os.getenv("PLANNING_MODEL", "gemini-2.5-flash")
# initial image (no text)
IMAGEN_MODEL = os.getenv("IMAGEN_MODEL", "imagen-4.0-generate-001")
IMAGEN_FALLBACK_MODEL = os.getenv(
    "IMAGEN_FALLBACK_MODEL", "gemini-2.5-flash-image-preview")
NANO_EDIT_MODEL = os.getenv(
    "NANO_EDIT_MODEL", "gemini-2.5-flash-image-preview")   # edit to add bubbles

PANEL_SIZE = int(os.getenv("PANEL_SIZE", "1024"))
STYLE_PRESET = "comic book style, bold outlines, vibrant colors, dynamic lighting, clean composition"
# if very long, still attempt edit; fallback to PIL if edit fails
MAX_BUBBLE_CHARS = int(os.getenv("MAX_BUBBLE_CHARS", "220"))
PRINT_PROMPTS = os.getenv("PRINT_PROMPTS", "1") == "1"

# ------------------ PROMPTS -----------------------
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    p = PROMPTS_DIR / f"{name}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8")


CHAR_PROMPT_TEMPLATE = load_prompt("character_extraction")
PANEL_PROMPT_TEMPLATE = load_prompt("panelization")

# Runtime templates
CHAR_REF_IMAGEN_TPL = (
    "Create a square character reference portrait, full-body 3/4 view, neutral pose, plain studio background.\n"
    f"Art direction: {STYLE_PRESET}.\n"
    "The character appearance: {appearance}.\n"
    "High detail, consistent identity for later panels.\n"
    "(No text in the image.)"
)

PANEL_BASE_IMAGEN_TPL = (
    "Square comic panel. Visual scene: {panel_visual}.\n"
    f"Art direction: {STYLE_PRESET}.\n"
    "Include these characters in the scene: {character_names}.\n"
    "Do not render any text, captions, or speech bubbles. Leave visual space for bubbles."
)

PANEL_EDIT_NANO_TPL = (
    "Edit the provided base panel image.\n"
    "Ensure each depicted character’s look matches the provided reference images (identity consistency).\n"
    "Add clean, readable comic speech bubble(s) with these line(s):\n"
    "{bubble_text_lines}\n"
    "Guidelines:\n"
    "- Place bubbles near the likely speaker; avoid covering faces.\n"
    "- Keep typography clear and high-contrast.\n"
    "- If there are multiple lines, use multiple bubbles or a stacked bubble.\n"
    "Only add bubbles and minor composition tweaks needed for readability."
)

# ------------------ DATA MODELS -------------------


class Character(BaseModel):
    name: str
    appearance: str
    summary: Optional[str] = ""


class Panel(BaseModel):
    index: int
    prompt: str
    dialogue: List[str] = Field(default_factory=list)       # ["Name: line"]
    characterNames: List[str] = Field(default_factory=list)

# ------------------ UTILITIES ---------------------


def fill(template: str, **kv):
    """Replace only specific placeholders, leaving JSON braces alone."""
    out = template
    for k, v in kv.items():
        out = out.replace(f"{{{k}}}", v)
    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def slugify(text: str, fallback: str = "item") -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or fallback


def first_json_block(s: str) -> str:
    # Find all potential JSON blocks and return the largest valid one
    starts = [m.start() for m in re.finditer(r"[\{\[]", s)]
    best_chunk = None
    best_size = 0

    for i in starts:
        for j in range(len(s), i+1, -1):
            chunk = s[i:j]
            try:
                parsed = json.loads(chunk)
                # Prefer larger, more complex JSON structures
                chunk_size = len(chunk)
                if chunk_size > best_size:
                    best_chunk = chunk
                    best_size = chunk_size
            except Exception:
                continue

    if best_chunk:
        return best_chunk
    raise ValueError("No valid JSON in model output")


def image_bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGBA")


def pil_to_png_bytes(img: Image.Image) -> bytes:
    """Converts a PIL Image object to PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def pad_to_square(img: Image.Image, size: int = PANEL_SIZE) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), resample=Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    canvas.paste(img, ((size - new_w)//2, (size - new_h)//2))
    return canvas


def to_inline_image_part(img_bytes: bytes, mime: str = "image/png") -> Dict[str, Any]:
    return {"inline_data": {"mime_type": mime, "data": base64.b64encode(img_bytes).decode("utf-8")}}

# --- Simple prompt logger (stdout + file) ---


class PromptLogger:
    def __init__(self, out_file: Path):
        self.out_file = out_file
        self.lines: List[str] = []

    def log(self, title: str, content: str):
        block = f"\n===== {title} =====\n{content.strip()}\n"
        self.lines.append(block)
        if PRINT_PROMPTS:
            print(block)

    def flush(self):
        self.out_file.write_text("".join(self.lines), encoding="utf-8")

# ------------------ GENAI WRAPPER ----------------


class GAIC:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    # Text planning (Gemini 2.5)
    def generate_text(self, prompt: str, model: str = LLM_MODEL) -> str:
        resp = self.client.models.generate_content(
            model=model, contents=prompt)
        if getattr(resp, "text", ""):
            return resp.text
        out = []
        for c in getattr(resp, "candidates", []) or []:
            for p in getattr(c, "content", {}).parts or []:
                if getattr(p, "text", None):
                    out.append(p.text)
        return "\n".join(out).strip()

    def generate_image_imagen(self, prompt: str, model: str = None) -> bytes:
        """
        Generate a base image using Imagen 4.0 via models.generate_images.
        Returns PNG bytes.
        """
        if model is None:
            model = os.getenv("IMAGEN_MODEL", "imagen-4.0-generate-001")

        # Call the Imagen 4 endpoint
        resp = self.client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,   # we just need the first
                # You can add more here if your account supports them, e.g.:
                # aspect_ratio="1:1",  # if available to your tenant
            ),
        )

        if not getattr(resp, "generated_images", None):
            # Optional: fallback to a Gemini image model if available
            fallback = os.getenv("IMAGEN_FALLBACK_MODEL",
                                 "gemini-2.5-flash-image-preview")
            print(
                f"[warn] Imagen returned no images; falling back to {fallback}")
            alt = self.client.models.generate_content(
                model=fallback, contents=prompt)
            for cand in alt.candidates:
                for part in cand.content.parts:
                    if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                        return part.inline_data.data
            raise RuntimeError("No image returned by Imagen or fallback.")

        # The SDK's .image object has an image_bytes property with the raw image data.
        # We'll use this to create a standard PIL Image object.
        sdk_image = resp.generated_images[0].image

        # 1. Get the raw image bytes from the SDK object.
        image_bytes = sdk_image.image_bytes

        # 2. Create a BytesIO buffer from the raw bytes.
        buffer = io.BytesIO(image_bytes)

        # 3. Open the image data from the buffer into a standard PIL Image.
        true_pil_image = Image.open(buffer)

        # 4. Now, convert this standard PIL Image to PNG bytes.
        return pil_to_png_bytes(true_pil_image)

    def edit_image_with_nano(self, base_image: bytes, instruction: str, ref_images: Optional[List[bytes]] = None,
                             model: str = NANO_EDIT_MODEL) -> bytes:
        parts: List[Any] = []
        parts.append({"inline_data": {"mime_type": "image/png",
                     "data": base64.b64encode(base_image).decode("utf-8")}})
        if ref_images:
            for b in ref_images:
                parts.append({"inline_data": {"mime_type": "image/png",
                             "data": base64.b64encode(b).decode("utf-8")}})
        parts.append(instruction)
        resp = self.client.models.generate_content(model=model, contents=parts)
        for cand in resp.candidates:
            for part in cand.content.parts:
                if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                    return part.inline_data.data
        raise RuntimeError("Nano edit returned no image data.")

# ------------------ PIPELINE STEPS ---------------


def extract_characters(g: GAIC, story: str, log: PromptLogger) -> List[Character]:
    prompt = fill(CHAR_PROMPT_TEMPLATE, story=story)
    log.log("CHARACTER_EXTRACTION_PROMPT", prompt)
    txt = g.generate_text(prompt)
    log.log("CHARACTER_EXTRACTION_RAW_RESPONSE", txt)
    data = json.loads(first_json_block(txt))

    # Handle both cases: {"characters": [...]} or just [...]
    if isinstance(data, list):
        chars_data = data
    elif isinstance(data, dict) and "characters" in data:
        chars_data = data["characters"]
    else:
        raise ValueError(f"Unexpected JSON structure: {type(data)}")

    try:
        chars = [Character(**c) for c in chars_data]
    except Exception as e:
        raise ValueError(f"Character JSON validation error: {e}")
    # Dedup by name
    seen, uniq = set(), []
    for c in chars:
        k = c.name.strip().lower()
        if k not in seen:
            uniq.append(c)
            seen.add(k)
    return uniq


def panelize_story(g: GAIC, story: str, log: PromptLogger) -> List[Panel]:
    prompt = fill(PANEL_PROMPT_TEMPLATE, story=story)
    log.log("PANELIZATION_PROMPT", prompt)
    txt = g.generate_text(prompt)
    log.log("PANELIZATION_RAW_RESPONSE", txt)
    data = json.loads(first_json_block(txt))

    # Handle both cases: {"panels": [...]} or just [...]
    if isinstance(data, list):
        panels_data = data
    elif isinstance(data, dict) and "panels" in data:
        panels_data = data["panels"]
    else:
        raise ValueError(f"Unexpected JSON structure: {type(data)}")

    try:
        panels = [Panel(**p) for p in panels_data]
    except Exception as e:
        raise ValueError(f"Panel JSON validation error: {e}")
    panels.sort(key=lambda x: x.index)
    return panels


def build_char_ref_prompt(c: Character) -> str:
    return CHAR_REF_IMAGEN_TPL.format(appearance=c.appearance)


def build_panel_base_prompt(p: Panel) -> str:
    names = ", ".join(
        p.characterNames) if p.characterNames else "as applicable"
    return PANEL_BASE_IMAGEN_TPL.format(panel_visual=p.prompt, character_names=names)


def build_nano_edit_instruction(p: Panel) -> str:
    if not p.dialogue:
        lines = "(no text)"
    else:
        lines = "\n".join(f"- {d}" for d in p.dialogue)
    return PANEL_EDIT_NANO_TPL.format(bubble_text_lines=lines)


# ------------------ MAIN ORCHESTRATION -----------
DEMO_STORY = """\
[Scene: A quiet city rooftop at dusk.]

Alex: We finally made it.
Mina: The signal's weak, but it's there.
(Narration) A faint glow pulses from the distant tower.

A sudden gust sends papers flying.

Alex: Hold on—do you hear that?
Mina: ...Wings?

A shadow passes overhead as a small mechanical hawk lands nearby, blinking.

Alex: Our courier. Let's see what the Council sent.
Mina: If it's another warning, I'm going to scream.
"""


def run_pipeline(story_text: str, out_root: Path):
    ensure_dir(out_root)
    logger = PromptLogger(out_root / "prompts_used.txt")
    g = GAIC(API_KEY)

    print(">> Planning with Gemini 2.5...")
    characters = extract_characters(g, story_text, logger)
    print(f"   Characters: {[c.name for c in characters]}")
    panels = panelize_story(g, story_text, logger)
    print(f"   Panels: {len(panels)}")

    chars_dir = out_root / "characters"
    ensure_dir(chars_dir)
    panels_dir = out_root / "panels"
    ensure_dir(panels_dir)

    # Character refs via Imagen 4
    print(">> Generating character references (Imagen 4)...")
    name_to_ref_bytes: Dict[str, bytes] = {}
    for c in characters:
        prompt = build_char_ref_prompt(c)
        logger.log(f"CHARACTER_REF_IMAGEN_PROMPT [{c.name}]", prompt)
        raw = g.generate_image_imagen(prompt)
        img = pad_to_square(image_bytes_to_pil(raw), PANEL_SIZE)
        png = pil_to_png_bytes(img)
        fname = f"{slugify(c.name)}.png"
        (chars_dir / fname).write_bytes(png)
        name_to_ref_bytes[c.name.lower()] = png
        print(f"   ✓ {c.name} -> {fname}")

    # Panels: base via Imagen 4, then edit via Nano to add bubbles (and enforce likeness)
    print(">> Rendering panels (Imagen 4 base → Nano edit for bubbles)...")
    panel_manifest = []
    for p in panels:
        base_prompt = build_panel_base_prompt(p)
        logger.log(f"PANEL_BASE_IMAGEN_PROMPT [#{p.index}]", base_prompt)
        base_raw = g.generate_image_imagen(base_prompt)
        base_img = pad_to_square(image_bytes_to_pil(base_raw), PANEL_SIZE)
        base_png = pil_to_png_bytes(base_img)

        # Gather ref images for characters in this panel
        ref_imgs = [name_to_ref_bytes[n.lower()]
                    for n in p.characterNames if n.lower() in name_to_ref_bytes]

        # Edit instruction for Nano Banana
        edit_instruction = build_nano_edit_instruction(p)
        logger.log(f"PANEL_EDIT_NANO_PROMPT [#{p.index}]", edit_instruction)

        try:
            edited_png = g.edit_image_with_nano(
                base_png, edit_instruction, ref_images=ref_imgs)
            final_img = pad_to_square(
                image_bytes_to_pil(edited_png), PANEL_SIZE)
        except Exception as e:
            # Fallback: keep base if edit fails
            print(
                f"   ! Nano edit failed on panel {p.index}: {e}. Keeping base image.")
            final_img = image_bytes_to_pil(base_png)

        fname = f"panel-{p.index:03d}.png"
        (panels_dir / fname).write_bytes(pil_to_png_bytes(final_img))
        panel_manifest.append({"index": p.index, "file": f"panels/{fname}",
                              "dialogue": p.dialogue, "characters": p.characterNames})
        print(f"   ✓ Panel {p.index} -> {fname}")

    # Pages (6 per page)
    panel_manifest.sort(key=lambda x: x["index"])
    pages = []
    for i in range(0, len(panel_manifest), 6):
        pages.append({
            "index": i // 6 + 1,
            "panels": panel_manifest[i:i+6]
        })

    manifest = {
        "meta": {"panel_size": PANEL_SIZE, "style": STYLE_PRESET},
        "characters": [{"name": c.name, "appearance": c.appearance, "file": f"characters/{slugify(c.name)}.png"} for c in characters],
        "panels": panel_manifest,
        "pages": pages,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Save all prompts used
    logger.flush()
    print(f">> Done. Output at: {out_root}")


# ------------------ CLI -------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        story_text = Path(sys.argv[1]).read_text(encoding="utf-8")
        slug = slugify(Path(sys.argv[1]).stem)
    else:
        print("No input file given; using DEMO_STORY.")
        story_text = DEMO_STORY
        slug = "demo-story"

    run_id = "".join(random.choices(
        string.ascii_lowercase + string.digits, k=6))
    out_dir = Path("output") / f"{slug}-{run_id}"
    run_pipeline(story_text, out_dir)
