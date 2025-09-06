# comic_e2e.py
import textwrap
from typing import Literal
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
# Use combined single-step panel generation (cost optimization)
USE_COMBINED_PANELS = os.getenv("USE_COMBINED_PANELS", "1") == "1"

# ------------------ PROMPTS -----------------------
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    p = PROMPTS_DIR / f"{name}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8")


CHAR_PROMPT_TEMPLATE = load_prompt("character_extraction")
SCENE_PROMPT_TEMPLATE = load_prompt("scene_extraction")
PANEL_PROMPT_TEMPLATE = load_prompt("panelization")

# Runtime templates
CHAR_REF_IMAGEN_TPL = load_prompt("char_ref_imagen")
SCENE_REF_IMAGEN_TPL = load_prompt("scene_ref_imagen")
PANEL_BASE_NANO_TPL = load_prompt("panel_base_nano")
PANEL_EDIT_NANO_TPL = load_prompt("panel_edit_nano")
PANEL_COMBINED_NANO_TPL = load_prompt("panel_combined_nano")

# ------------------ DATA MODELS -------------------


class ScenePresence(BaseModel):
    sceneName: str
    isPresent: bool
    action: str


class Character(BaseModel):
    name: str
    kind: Literal["human", "creature",
                  "robotic_creature", "vehicle", "prop"] = "human"
    aliases: List[str] = Field(default_factory=list)
    appearance: str
    summary: Optional[str] = ""
    scenePresence: List[ScenePresence] = Field(default_factory=list)


class Scene(BaseModel):
    name: str
    description: str
    setting: str


class SpeechLine(BaseModel):
    speaker: str
    text: str
    whisper: bool = False


class Panel(BaseModel):
    index: int
    prompt: str
    dialogue: List[SpeechLine] = Field(default_factory=list)
    # Narration text for caption boxes
    narration: Optional[str] = None
    characterNames: List[str] = Field(default_factory=list)
    sceneName: str = ""
    perspective: str = ""
    sfx: List[str] = Field(default_factory=list)
    visualCues: List[str] = Field(default_factory=list)

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


def resize_to_fill(img: Image.Image, size: int = PANEL_SIZE) -> Image.Image:
    """Resize image to fill the entire square without padding/whitespace."""
    w, h = img.size
    scale = max(size / w, size / h)  # Scale to fill, not fit
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), resample=Image.LANCZOS)

    # Crop to exact size if needed
    if new_w > size or new_h > size:
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        img = img.crop((left, top, left + size, top + size))

    return img


def to_inline_image_part(img_bytes: bytes, mime: str = "image/png") -> Dict[str, Any]:
    return {"inline_data": {"mime_type": mime, "data": base64.b64encode(img_bytes).decode("utf-8")}}


def normalize_dialogue(p: Panel) -> None:
    """Repair stray newlines like 'Hold on\n—do you hear that?'"""
    for line in p.dialogue:
        line.text = line.text.replace("\n—", "—").strip()


def validate_panel_speakers(p: Panel, cast_names: List[str]) -> List[str]:
    """Validate that all speakers exist in cast and are listed in characterNames"""
    errors = []
    for line in p.dialogue:
        if line.speaker not in cast_names:
            errors.append(
                f"Unknown speaker '{line.speaker}' in panel {p.index}")
        if line.speaker not in p.characterNames:
            errors.append(
                f"Speaker '{line.speaker}' missing from characterNames in panel {p.index}")
    return errors


def audit_and_patch_cast(story: str, characters: List[Character]) -> List[Character]:
    """Guarantee non-human entities appear in cast"""
    text = story.lower()
    def has(term): return term in text
    names = {c.name.lower() for c in characters}

    # Heuristic: courier/hawk keywords => ensure Mechanical Hawk exists
    if any(has(k) for k in ["mechanical hawk", "courier", "hawk"]):
        if "mechanical hawk" not in names:
            characters.append(Character(
                name="Mechanical Hawk",
                kind="robotic_creature",
                aliases=["courier", "hawk", "mechanical bird"],
                appearance="Small sleek automaton resembling a hawk; metallic feather panels, red blinking sensor eye, leg-mounted message tube; silent servo-driven wings.",
                summary="Courier from the Council",
                scenePresence=[ScenePresence(
                    sceneName="Quiet City Rooftop",
                    isPresent=True,
                    action="circles overhead, lands, blinks, delivers payload"
                )]
            ))
    return characters


def patch_panels_for_entities(panels: List[Panel], characters: List[Character]):
    """Ensure entities appear in the right panels with proper visual cues"""
    cast = {c.name for c in characters}
    for p in panels:
        # Normalize dialogue
        normalize_dialogue(p)

        # If courier/hawk is mentioned in lines or prompt, add it
        panel_text = " ".join([l.text.lower()
                              for l in p.dialogue]) + " " + p.prompt.lower()
        if any(k in panel_text for k in ["courier", "hawk"]):
            if "Mechanical Hawk" in cast and "Mechanical Hawk" not in p.characterNames:
                p.characterNames.append("Mechanical Hawk")

        # Add recommended cues for this specific story
        if p.index == 2 and "WHOOSH" not in p.sfx:
            p.sfx.append("WHOOSH")
            p.visualCues.append("papers_fly_directional")
        if p.index == 3 and "wing_shadow_passing_overhead" not in p.visualCues:
            p.visualCues.append("wing_shadow_passing_overhead")
        if p.index == 5:
            p.sfx.extend([s for s in ["CLACK", "WHIRR"] if s not in p.sfx])
            for cue in ["red_eye_blink", "leg_tube_payload"]:
                if cue not in p.visualCues:
                    p.visualCues.append(cue)


def build_character_descriptors(p: Panel, all_chars: List[Character]) -> str:
    """Build rich character descriptions with appearance fallbacks"""
    descs = []
    for name in p.characterNames:
        c = next((x for x in all_chars if x.name.lower() == name.lower()), None)
        if not c:
            continue
        action = next((sp.action for sp in c.scenePresence
                      if sp.sceneName.lower() == p.sceneName.lower() and sp.isPresent),
                      "present")
        descs.append(f"{c.name}: {c.appearance}. In this panel: {action}.")
    return " | ".join(descs) if descs else "As applicable."


def draw_caption_box(img: Image.Image, text: str, position: str = "top") -> Image.Image:
    """
    Draw a styled caption box on the image with the given text.
    Position can be "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right"
    """
    if not text or not text.strip():
        return img

    # Create a copy to avoid modifying the original
    result = img.copy()
    draw = ImageDraw.Draw(result)

    # Try to load a font, fallback to default
    try:
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            "arial.ttf"  # Windows
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 24)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Word wrap text to prevent overflow
    max_width = int(result.size[0] * 0.8)
    wrapped = textwrap.fill(text, width=42)  # Coarse font-independent fallback

    # Calculate text dimensions using multiline
    text_bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=4)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Add padding around text
    padding = 12
    box_width = text_width + 2 * padding
    box_height = text_height + 2 * padding

    # Get image dimensions
    img_width, img_height = result.size

    # Calculate position based on the position parameter
    if position == "top":
        x = (img_width - box_width) // 2
        y = 20
    elif position == "bottom":
        x = (img_width - box_width) // 2
        y = img_height - box_height - 20
    elif position == "top_left":
        x = 20
        y = 20
    elif position == "top_right":
        x = img_width - box_width - 20
        y = 20
    elif position == "bottom_left":
        x = 20
        y = img_height - box_height - 20
    elif position == "bottom_right":
        x = img_width - box_width - 20
        y = img_height - box_height - 20
    else:
        # Default to top center
        x = (img_width - box_width) // 2
        y = 20

    # Ensure the box stays within image bounds
    x = max(0, min(x, img_width - box_width))
    y = max(0, min(y, img_height - box_height))

    # Draw caption box with rounded corners
    corner_radius = 8
    draw.rounded_rectangle(
        [x, y, x + box_width, y + box_height],
        radius=corner_radius,
        fill="white",
        outline="black",
        width=2
    )

    # Draw text using multiline
    text_x = x + padding
    text_y = y + padding
    draw.multiline_text((text_x, text_y), wrapped,
                        fill="black", font=font, spacing=4)

    return result


def create_comic_layout(panels: List[bytes], title: str, panel_size: int = PANEL_SIZE) -> Image.Image:
    """Stitch panels together into a comic book layout with title."""
    if not panels:
        raise ValueError("No panels to stitch together")

    # Calculate layout - 2 columns max
    num_panels = len(panels)
    cols = 2
    rows = (num_panels + 1) // 2  # Ceiling division

    # Comic styling
    border_width = 8
    panel_spacing = 12
    title_height = 120  # Increased from 80
    margin = 20

    # Calculate canvas size
    canvas_width = cols * panel_size + (cols - 1) * panel_spacing + 2 * margin
    canvas_height = title_height + rows * panel_size + \
        (rows - 1) * panel_spacing + 2 * margin

    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    # Try to load a font, fallback to default
    try:
        # Try to find a bold font
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            "arial.ttf"  # Windows
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 48)  # Increased from 36
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Draw title
    title_bbox = draw.textbbox((0, 0), title, font=font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (canvas_width - title_width) // 2
    title_y = margin

    # Title background
    draw.rectangle([title_x - 15, title_y - 10, title_x + title_width + 15, title_y + title_height - 15],
                   fill="black")
    draw.text((title_x, title_y), title, fill="white", font=font)

    # Place panels
    y_offset = margin + title_height
    for i, panel_bytes in enumerate(panels):
        row = i // cols
        col = i % cols

        x = margin + col * (panel_size + panel_spacing)
        y = y_offset + row * (panel_size + panel_spacing)

        # Load and resize panel
        panel_img = image_bytes_to_pil(panel_bytes)
        panel_img = resize_to_fill(panel_img, panel_size)

        # Draw border
        draw.rectangle([x - border_width//2, y - border_width//2,
                       x + panel_size + border_width//2, y + panel_size + border_width//2],
                       fill="black")

        # Paste panel
        canvas.paste(panel_img, (x, y))

    return canvas

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

    # Structured output generation
    def generate_structured(self, prompt: str, response_schema, model: str = LLM_MODEL):
        resp = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            }
        )
        return resp.parsed

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


def generate_comic_title(g: GAIC, story: str) -> str:
    """Generate a compelling comic book title for the story."""
    prompt = f"""Based on this story, generate a compelling comic book title that captures the essence and tone of the narrative. Return only the title, nothing else.

Story:
{story}

Generate a title that is:
- Catchy and memorable
- Appropriate for a comic book
- 1-5 words maximum
- Captures the main theme or mood"""

    return g.generate_text(prompt).strip().strip('"').strip("'")

# ------------------ PIPELINE STEPS ---------------


class CharacterResponse(BaseModel):
    characters: List[Character]


class SceneResponse(BaseModel):
    scenes: List[Scene]


class PanelResponse(BaseModel):
    panels: List[Panel]


def extract_characters(g: GAIC, story: str, log: PromptLogger) -> List[Character]:
    prompt = fill(CHAR_PROMPT_TEMPLATE, story=story)
    log.log("CHARACTER_EXTRACTION_PROMPT", prompt)
    response = g.generate_structured(prompt, CharacterResponse)
    log.log("CHARACTER_EXTRACTION_RESPONSE", str(response))

    # Dedup by name
    seen, uniq = set(), []
    for c in response.characters:
        k = c.name.strip().lower()
        if k not in seen:
            uniq.append(c)
            seen.add(k)
    return uniq


def extract_scenes(g: GAIC, story: str, log: PromptLogger) -> List[Scene]:
    prompt = fill(SCENE_PROMPT_TEMPLATE, story=story)
    log.log("SCENE_EXTRACTION_PROMPT", prompt)
    response = g.generate_structured(prompt, SceneResponse)
    log.log("SCENE_EXTRACTION_RESPONSE", str(response))
    return response.scenes


def panelize_story(g: GAIC, story: str, scenes: List[Scene], log: PromptLogger) -> List[Panel]:
    scene_info = "\n".join([f"- {s.name}: {s.description}" for s in scenes])
    prompt = fill(PANEL_PROMPT_TEMPLATE, story=story, scenes=scene_info)
    log.log("PANELIZATION_PROMPT", prompt)
    response = g.generate_structured(prompt, PanelResponse)
    log.log("PANELIZATION_RESPONSE", str(response))

    panels = response.panels
    panels.sort(key=lambda x: x.index)
    return panels


def build_char_ref_prompt(c: Character) -> str:
    return CHAR_REF_IMAGEN_TPL.format(appearance=c.appearance, STYLE_PRESET=STYLE_PRESET)


def build_scene_ref_prompt(s: Scene) -> str:
    return SCENE_REF_IMAGEN_TPL.format(setting=s.setting, description=s.description, STYLE_PRESET=STYLE_PRESET)


def build_panel_base_instruction(p: Panel, characters: List[Character]) -> str:
    # Use rich character descriptions with appearance fallbacks
    character_info = build_character_descriptors(p, characters)

    # Add perspective information to the instruction
    perspective_instruction = f"Camera perspective: {p.perspective}. "
    if p.perspective == "close_up":
        perspective_instruction += "Focus tightly on the character's face or specific detail."
    elif p.perspective == "character_focus":
        perspective_instruction += "Emphasize one character while others are in the background."
    elif p.perspective == "action_shot":
        perspective_instruction += "Use a dynamic angle to capture movement or action."
    elif p.perspective == "establishing_shot":
        perspective_instruction += "Show a wide view to establish the location."
    elif p.perspective == "reaction_shot":
        perspective_instruction += "Close-up showing character's emotional response."
    elif p.perspective == "full_scene":
        perspective_instruction += "Show the entire scene with all characters visible."
    elif p.perspective == "medium_shot":
        perspective_instruction += "Show character from waist up."
    elif p.perspective == "wide_shot":
        perspective_instruction += "Show characters in context of larger environment."

    return PANEL_BASE_NANO_TPL.format(
        panel_visual=f"{perspective_instruction}{p.prompt}",
        character_names=character_info,
        STYLE_PRESET=STYLE_PRESET
    )


def build_nano_edit_instruction(p: Panel) -> str:
    parts = []
    if p.dialogue:
        lines = "\n".join(f"- {d.speaker}: {d.text}" + (" (whisper)" if d.whisper else "")
                          for d in p.dialogue)
        parts.append(
            f"Add clean, readable speech bubbles with this text:\n{lines}")
    if p.narration:
        parts.append(
            f"Add a narration caption box with this text: {p.narration}")
    if p.sfx:
        parts.append("Place small SFX labels: " + ", ".join(p.sfx))

    if parts:
        return PANEL_EDIT_NANO_TPL.format(bubble_text_lines="\n\n".join(parts))
    else:
        return "Make minor composition adjustments; do not add text."


def build_panel_combined_instruction(p: Panel, characters: List[Character]) -> str:
    """Build a combined instruction for single-step panel generation with dialogue."""
    # Use rich character descriptions with appearance fallbacks
    character_info = build_character_descriptors(p, characters)

    # Add perspective information to the instruction
    perspective_instruction = f"Camera perspective: {p.perspective}. "
    if p.perspective == "close_up":
        perspective_instruction += "Focus tightly on the character's face or specific detail."
    elif p.perspective == "character_focus":
        perspective_instruction += "Emphasize one character while others are in the background."
    elif p.perspective == "action_shot":
        perspective_instruction += "Use a dynamic angle to capture movement or action."
    elif p.perspective == "establishing_shot":
        perspective_instruction += "Show a wide view to establish the location."
    elif p.perspective == "reaction_shot":
        perspective_instruction += "Close-up showing character's emotional response."
    elif p.perspective == "full_scene":
        perspective_instruction += "Show the entire scene with all characters visible."
    elif p.perspective == "medium_shot":
        perspective_instruction += "Show character from waist up."
    elif p.perspective == "wide_shot":
        perspective_instruction += "Show characters in context of larger environment."

    # Add SFX and visual cues to the instruction
    fx_bits = []
    if p.sfx:
        fx_bits.append("Add small onomatopoeia SFX labels: " +
                       ", ".join(p.sfx))
    if p.visualCues:
        fx_bits.append("Render visual cues: " + ", ".join(p.visualCues))
    fx_text = (" " + ". ".join(fx_bits) + ".") if fx_bits else ""

    # Handle dialogue and narration instruction
    text_instructions = []
    if p.dialogue:
        lines = "\n".join(f"- {d.speaker}: {d.text}" + (" (whisper)" if d.whisper else "")
                          for d in p.dialogue)
        text_instructions.append(
            f"Add clean, readable comic speech bubble(s) with this text:\n{lines}")
    if p.narration:
        text_instructions.append(
            f"Add a narration caption box with this text: {p.narration}")

    if text_instructions:
        dialogue_instruction = f"Finally, {'. '.join(text_instructions)}."
    else:
        dialogue_instruction = "There is no dialogue or narration for this panel."

    return PANEL_COMBINED_NANO_TPL.format(
        panel_visual=f"{perspective_instruction}{p.prompt}{fx_text}",
        character_names=character_info,
        STYLE_PRESET=STYLE_PRESET,
        dialogue_instruction=dialogue_instruction
    )


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

    # Audit and patch cast to ensure non-human entities are included
    characters = audit_and_patch_cast(story_text, characters)
    print(f"   Characters: {[c.name for c in characters]}")

    scenes = extract_scenes(g, story_text, logger)
    print(f"   Scenes: {[s.name for s in scenes]}")
    panels = panelize_story(g, story_text, scenes, logger)

    # Patch panels to ensure entities appear correctly with visual cues
    patch_panels_for_entities(panels, characters)

    # Validate speakers and show any issues
    cast_names = [c.name for c in characters]
    all_errors = []
    for panel in panels:
        errors = validate_panel_speakers(panel, cast_names)
        all_errors.extend(errors)

    if all_errors:
        print("   Validation warnings:")
        for error in all_errors:
            print(f"     - {error}")

    print(f"   Panels: {len(panels)}")

    chars_dir = out_root / "characters"
    ensure_dir(chars_dir)
    scenes_dir = out_root / "scenes"
    ensure_dir(scenes_dir)
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

    # Scene refs via Imagen 4
    print(">> Generating scene references (Imagen 4)...")
    scene_to_ref_bytes: Dict[str, bytes] = {}
    for s in scenes:
        prompt = build_scene_ref_prompt(s)
        logger.log(f"SCENE_REF_IMAGEN_PROMPT [{s.name}]", prompt)
        raw = g.generate_image_imagen(prompt)
        img = pad_to_square(image_bytes_to_pil(raw), PANEL_SIZE)
        png = pil_to_png_bytes(img)
        fname = f"{slugify(s.name)}.png"
        (scenes_dir / fname).write_bytes(png)
        scene_to_ref_bytes[s.name.lower()] = png
        print(f"   ✓ {s.name} -> {fname}")

    # Panels: Generate panels using either combined or two-step approach
    if USE_COMBINED_PANELS:
        print(">> Rendering panels (Combined approach: scene + characters + dialogue in one call)...")
    else:
        print(">> Rendering panels (Two-step approach: scene + characters → speech bubbles)...")

    panel_manifest = []
    previous_panel_bytes = None  # For continuity feeding

    for p in panels:
        # Get scene reference for this panel
        scene_ref_bytes = scene_to_ref_bytes.get(p.sceneName.lower())
        if not scene_ref_bytes:
            # Fallback to first scene if panel scene not found
            scene_ref_bytes = list(scene_to_ref_bytes.values())[
                0] if scene_to_ref_bytes else None

        if not scene_ref_bytes:
            raise RuntimeError(
                f"No scene reference available for panel {p.index}")

        # Gather ref images for characters in this panel
        char_ref_imgs = [name_to_ref_bytes[n.lower()]
                         for n in p.characterNames if n.lower() in name_to_ref_bytes]

        # Add previous panel for continuity if available
        ref_images_with_continuity = char_ref_imgs.copy()
        if previous_panel_bytes:
            ref_images_with_continuity.insert(
                0, previous_panel_bytes)  # Put previous panel first

        if USE_COMBINED_PANELS:
            # Single combined call for cost optimization
            combined_instruction = build_panel_combined_instruction(
                p, characters)
            logger.log(
                f"PANEL_COMBINED_NANO_PROMPT [#{p.index}]", combined_instruction)

            try:
                final_png = g.edit_image_with_nano(
                    scene_ref_bytes, combined_instruction, ref_images=ref_images_with_continuity)
                final_img = resize_to_fill(
                    image_bytes_to_pil(final_png), PANEL_SIZE)
                final_png = pil_to_png_bytes(final_img)
                print(
                    f"   ✓ Combined generation {p.index} -> scene + characters + dialogue")
            except Exception as e:
                print(
                    f"   ! Combined generation failed on panel {p.index}: {e}. Using scene reference.")
                final_img = resize_to_fill(
                    image_bytes_to_pil(scene_ref_bytes), PANEL_SIZE)
                final_png = pil_to_png_bytes(final_img)

            # For combined approach, we don't have a separate base image
            base_fname = f"base-panel-{p.index:03d}.png"
            # Save same as base for compatibility
            (panels_dir / base_fname).write_bytes(final_png)
        else:
            # Original two-step approach
            # First Nano Banana round: place characters in scene
            base_instruction = build_panel_base_instruction(p, characters)
            logger.log(
                f"PANEL_BASE_NANO_PROMPT [#{p.index}]", base_instruction)

            try:
                base_png = g.edit_image_with_nano(
                    scene_ref_bytes, base_instruction, ref_images=ref_images_with_continuity)
                # Use resize_to_fill instead of pad_to_square
                base_img = resize_to_fill(
                    image_bytes_to_pil(base_png), PANEL_SIZE)
                base_png = pil_to_png_bytes(base_img)
            except Exception as e:
                print(
                    f"   ! Nano base edit failed on panel {p.index}: {e}. Using scene reference.")
                base_img = resize_to_fill(image_bytes_to_pil(
                    scene_ref_bytes), PANEL_SIZE)  # Use resize_to_fill
                base_png = pil_to_png_bytes(base_img)

            # Save base Nano Banana generation (scene + characters)
            base_fname = f"base-panel-{p.index:03d}.png"
            (panels_dir / base_fname).write_bytes(base_png)
            print(f"   ✓ Nano base {p.index} -> {base_fname}")

            # Second Nano Banana round: add speech bubbles (only if dialogue exists)
            if p.dialogue:
                edit_instruction = build_nano_edit_instruction(p)
                logger.log(
                    f"PANEL_EDIT_NANO_PROMPT [#{p.index}]", edit_instruction)

                try:
                    edited_png = g.edit_image_with_nano(
                        base_png, edit_instruction, ref_images=char_ref_imgs)
                    final_img = resize_to_fill(
                        # Use resize_to_fill
                        image_bytes_to_pil(edited_png), PANEL_SIZE)
                    final_png = pil_to_png_bytes(final_img)
                    print(f"   ✓ Nano edit {p.index} -> added speech bubbles")
                except Exception as e:
                    # Fallback: keep base if edit fails
                    print(
                        f"   ! Nano edit failed on panel {p.index}: {e}. Keeping base image.")
                    final_img = image_bytes_to_pil(base_png)
                    final_png = base_png
            else:
                # No dialogue, skip nano edit entirely to save API costs
                final_img = image_bytes_to_pil(base_png)
                final_png = base_png
                print(
                    f"   ✓ Panel {p.index} -> no dialogue, skipping nano edit (cost savings)")

        # Add narration caption box if needed (using PIL as fallback)
        if p.narration:
            try:
                final_img = image_bytes_to_pil(final_png)
                final_img_with_caption = draw_caption_box(
                    final_img, p.narration, "top")
                final_png = pil_to_png_bytes(final_img_with_caption)
                print(f"   ✓ Added narration caption to panel {p.index}")
            except Exception as e:
                print(f"   ! Failed to add caption to panel {p.index}: {e}")

        # Save final panel image
        fname = f"panel-{p.index:03d}.png"
        (panels_dir / fname).write_bytes(final_png)
        # Convert structured dialogue back to simple format for manifest
        dialogue_strings = [f"{d.speaker}: {d.text}" for d in p.dialogue]
        panel_manifest.append({"index": p.index, "file": f"panels/{fname}",
                              "base_file": f"panels/{base_fname}",
                               "dialogue": dialogue_strings, "narration": p.narration,
                               "characters": p.characterNames, "scene": p.sceneName,
                               "perspective": p.perspective, "sfx": p.sfx,
                               "visualCues": p.visualCues})
        print(f"   ✓ Panel {p.index} -> {fname}")

        # Store this panel for continuity in next panel
        previous_panel_bytes = final_png

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
        "characters": [{"name": c.name, "kind": c.kind, "aliases": c.aliases, "appearance": c.appearance, "summary": c.summary, "scenePresence": [{"sceneName": sp.sceneName, "isPresent": sp.isPresent, "action": sp.action} for sp in c.scenePresence], "file": f"characters/{slugify(c.name)}.png"} for c in characters],
        "scenes": [{"name": s.name, "description": s.description, "setting": s.setting, "file": f"scenes/{slugify(s.name)}.png"} for s in scenes],
        "panels": panel_manifest,
        "pages": pages,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Generate comic title and create stitched layout
    print(">> Creating final comic layout...")
    comic_title = generate_comic_title(g, story_text)
    logger.log("COMIC_TITLE_GENERATION", f"Generated title: {comic_title}")
    print(f"   ✓ Generated title: {comic_title}")

    # Collect all final panel images
    final_panel_bytes = []
    for panel_info in panel_manifest:
        panel_path = out_root / panel_info["file"]
        if panel_path.exists():
            final_panel_bytes.append(panel_path.read_bytes())

    if final_panel_bytes:
        comic_layout = create_comic_layout(
            final_panel_bytes, comic_title, PANEL_SIZE)
        comic_path = out_root / "comic_final.png"
        comic_layout.save(comic_path, "PNG")
        print(f"   ✓ Final comic saved: {comic_path}")

    # Save all prompts used
    logger.flush()
    print(f">> Done. Output at: {out_root}")
    print(f">> Final comic: {out_root / 'comic_final.png'}")


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
