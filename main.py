# main.py
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
# nano banana flash for all image generation
NANO_IMAGE_MODEL = os.getenv(
    "NANO_IMAGE_MODEL", "gemini-2.5-flash-image-preview")
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
CHAR_REF_NANO_TPL = load_prompt("char_ref_nano")
SCENE_REF_NANO_TPL = load_prompt("scene_ref_nano")
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


class CharacterPosition(BaseModel):
    name: str
    x: float  # 0.0 to 1.0, left to right
    y: float  # 0.0 to 1.0, top to bottom
    prominence: str  # "foreground", "background", "off_screen"


class Panel(BaseModel):
    index: int
    prompt: str
    dialogue: List[SpeechLine] = Field(default_factory=list)
    # Narration text for caption boxes
    narration: Optional[str] = None
    characterNames: List[str] = Field(default_factory=list)
    # Character positioning information for text placement
    characterPositions: List[CharacterPosition] = Field(default_factory=list)
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


def wrap_text(text: str, font, max_width: int, draw) -> List[str]:
    """
    Wrap text to fit within a maximum width, breaking at word boundaries.
    """
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]

        if test_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # Single word is too long, just add it anyway
                lines.append(word)
                current_line = ""

    if current_line:
        lines.append(current_line)

    return lines


def find_best_text_position_from_panel_data(speaker: str, panel: Panel,
                                            used_areas: List[tuple], rect_width: int, rect_height: int,
                                            img_width: int, img_height: int) -> tuple:
    """
    Find the best position for a text box based on the speaker's position from panel data.
    """
    # Find the speaker's position in the panel data
    speaker_position = None
    for char_pos in panel.characterPositions:
        if char_pos.name == speaker:
            speaker_position = char_pos
            break

    if not speaker_position or speaker_position.prominence == "off_screen":
        # Fallback to default positions if character position unknown or off-screen
        default_positions = [
            (img_width * 0.1, img_height * 0.1),
            (img_width * 0.6, img_height * 0.1),
            (img_width * 0.1, img_height * 0.4),
            (img_width * 0.6, img_height * 0.4),
            (img_width * 0.1, img_height * 0.7),
            (img_width * 0.6, img_height * 0.7),
        ]
        return default_positions[0]

    char_pixel_x = speaker_position.x * img_width
    char_pixel_y = speaker_position.y * img_height

    # Adjust positioning based on prominence
    if speaker_position.prominence == "background":
        # For background characters, place text more conservatively
        offset_distance = 30
    else:
        # For foreground characters, can place text closer
        offset_distance = 50

    # Try positions around the character, preferring above and to sides
    candidate_positions = [
        # Above character
        (char_pixel_x - rect_width/2, char_pixel_y - rect_height - offset_distance),
        # Above and to the left
        (char_pixel_x - rect_width - 20, char_pixel_y -
         rect_height - offset_distance + 20),
        # Above and to the right
        (char_pixel_x + 20, char_pixel_y - rect_height - offset_distance + 20),
        # To the side (left)
        (char_pixel_x - rect_width - 40, char_pixel_y - rect_height/2),
        # To the side (right)
        (char_pixel_x + 40, char_pixel_y - rect_height/2),
        # Below (as last resort)
        (char_pixel_x - rect_width/2, char_pixel_y + offset_distance),
    ]

    # Find the first position that doesn't overlap significantly with used areas
    for pos_x, pos_y in candidate_positions:
        # Clamp to image bounds
        pos_x = max(10, min(pos_x, img_width - rect_width - 10))
        pos_y = max(10, min(pos_y, img_height - rect_height - 10))

        # Check for overlap with used areas
        new_rect = (pos_x, pos_y, pos_x + rect_width, pos_y + rect_height)

        has_significant_overlap = False
        for used_rect in used_areas:
            if rectangles_overlap_significantly(new_rect, used_rect):
                has_significant_overlap = True
                break

        if not has_significant_overlap:
            return (pos_x, pos_y)

    # If all positions overlap, use the first one anyway (better than random placement)
    pos_x, pos_y = candidate_positions[0]
    pos_x = max(10, min(pos_x, img_width - rect_width - 10))
    pos_y = max(10, min(pos_y, img_height - rect_height - 10))
    return (pos_x, pos_y)


def rectangles_overlap_significantly(rect1: tuple, rect2: tuple, threshold: float = 0.3) -> bool:
    """
    Check if two rectangles overlap by more than the threshold (as fraction of smaller area).
    """
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2

    # Calculate intersection
    left = max(x1, x3)
    top = max(y1, y3)
    right = min(x2, x4)
    bottom = min(y2, y4)

    if left >= right or top >= bottom:
        return False  # No overlap

    overlap_area = (right - left) * (bottom - top)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    smaller_area = min(area1, area2)

    return (overlap_area / smaller_area) > threshold


def add_text_rectangles_to_panel(panel_img: Image.Image, panel: Panel) -> Image.Image:
    """
    Mechanically add text to a panel using PIL with simple rectangles.
    This creates clean, readable text that Nano can later convert to speech bubbles.
    Uses character position data from the panel for intelligent text placement.
    """
    if not panel.dialogue and not panel.narration:
        return panel_img

    # Work on a copy
    img = panel_img.copy()
    draw = ImageDraw.Draw(img)

    # Try to load a good font for speech bubbles
    font = None
    font_size = 20  # Slightly smaller for better fit
    try:
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            "arial.ttf"  # Windows
        ]
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Panel dimensions
    width, height = img.size

    # Maximum width for text (about 40% of panel width)
    max_text_width = int(width * 0.35)

    # Track used areas to avoid overlap
    used_areas = []

    # Add dialogue rectangles
    for i, dialogue_line in enumerate(panel.dialogue):
        # Format text with speaker name
        text = f"{dialogue_line.speaker}: {dialogue_line.text}"

        # Wrap text to fit within reasonable width
        wrapped_lines = wrap_text(text, font, max_text_width, draw)

        # Calculate dimensions for wrapped text
        line_height = draw.textbbox((0, 0), "Ay", font=font)[
            3] - draw.textbbox((0, 0), "Ay", font=font)[1]
        total_text_height = line_height * len(wrapped_lines)

        # Find the widest line for rectangle width
        max_line_width = 0
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            max_line_width = max(max_line_width, line_width)

        # Add padding for rectangle
        padding = 8
        rect_width = max_line_width + 2 * padding
        rect_height = total_text_height + 2 * padding

        # Find best position based on speaker's position from panel data
        rect_x, rect_y = find_best_text_position_from_panel_data(
            dialogue_line.speaker, panel, used_areas,
            rect_width, rect_height, width, height
        )

        # Draw white rectangle with black border
        draw.rectangle([rect_x, rect_y, rect_x + rect_width, rect_y + rect_height],
                       fill="white", outline="black", width=2)

        # Draw wrapped text in rectangle
        text_x = rect_x + padding
        text_y = rect_y + padding

        for line_idx, line in enumerate(wrapped_lines):
            line_y = text_y + (line_idx * line_height)
            draw.text((text_x, line_y), line, fill="black", font=font)

        # Track this area as used
        used_areas.append(
            (rect_x, rect_y, rect_x + rect_width, rect_y + rect_height))

    # Add narration rectangle if present
    if panel.narration:
        text = f"Narration: {panel.narration}"

        # Wrap narration text as well
        # Slightly wider for narration
        wrapped_lines = wrap_text(text, font, max_text_width * 1.5, draw)

        # Calculate dimensions for wrapped text
        line_height = draw.textbbox((0, 0), "Ay", font=font)[
            3] - draw.textbbox((0, 0), "Ay", font=font)[1]
        total_text_height = line_height * len(wrapped_lines)

        # Find the widest line for rectangle width
        max_line_width = 0
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            max_line_width = max(max_line_width, line_width)

        # Add padding for rectangle
        padding = 8
        rect_width = max_line_width + 2 * padding
        rect_height = total_text_height + 2 * padding

        # Position at bottom center for narration
        rect_x = (width - rect_width) // 2
        rect_y = height - rect_height - 20  # 20px from bottom

        # Ensure rectangle fits within image bounds
        rect_x = max(0, min(rect_x, width - rect_width))
        rect_y = max(0, min(rect_y, height - rect_height))

        # Draw yellow rectangle with black border for narration
        draw.rectangle([rect_x, rect_y, rect_x + rect_width, rect_y + rect_height],
                       fill="yellow", outline="black", width=2)

        # Draw wrapped narration text
        text_x = rect_x + padding
        text_y = rect_y + padding

        for line_idx, line in enumerate(wrapped_lines):
            line_y = text_y + (line_idx * line_height)
            draw.text((text_x, line_y), line, fill="black", font=font)

    return img


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

    def generate_image_with_nano(self, prompt: str, base_image: Optional[bytes] = None, model: str = NANO_IMAGE_MODEL) -> bytes:
        """
        Generate an image using nano banana flash model.
        If base_image is provided, edit that image. Otherwise, generate from scratch using a white canvas.
        Returns PNG bytes.
        """
        parts: List[Any] = []

        if base_image:
            # Edit mode: start with base image
            parts.append({"inline_data": {"mime_type": "image/png",
                         "data": base64.b64encode(base_image).decode("utf-8")}})
        else:
            # Generate from scratch: create a white canvas as base
            white_canvas = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), "white")
            white_canvas_bytes = pil_to_png_bytes(white_canvas)
            parts.append({"inline_data": {"mime_type": "image/png",
                         "data": base64.b64encode(white_canvas_bytes).decode("utf-8")}})

        parts.append(prompt)

        try:
            resp = self.client.models.generate_content(
                model=model, contents=parts)
        except Exception as e:
            print(f"[ERROR] API call failed: {e}")
            raise RuntimeError(f"Nano image generation API call failed: {e}")

        # Debug: print response structure
        print(f"[DEBUG] Response type: {type(resp)}")
        print(
            f"[DEBUG] Response hasattr candidates: {hasattr(resp, 'candidates')}")

        if not hasattr(resp, 'candidates') or not resp.candidates:
            print(f"[ERROR] No candidates in response: {resp}")
            raise RuntimeError("Nano image generation returned no candidates.")

        print(f"[DEBUG] Candidates: {len(resp.candidates)}")

        for i, cand in enumerate(resp.candidates):
            print(f"[DEBUG] Candidate {i}: type={type(cand)}")
            if not hasattr(cand, 'content'):
                print(f"[DEBUG] Candidate {i} has no content attribute")
                continue

            print(f"[DEBUG] Candidate {i} content: type={type(cand.content)}")
            if not hasattr(cand.content, 'parts') or not cand.content.parts:
                print(f"[DEBUG] Candidate {i} content has no parts")
                continue

            print(f"[DEBUG] Candidate {i} has {len(cand.content.parts)} parts")
            for j, part in enumerate(cand.content.parts):
                print(f"[DEBUG] Part {j}: type={type(part)}")
                print(
                    f"[DEBUG] Part {j} hasattr inline_data: {hasattr(part, 'inline_data')}")

                if hasattr(part, "inline_data") and part.inline_data:
                    print(
                        f"[DEBUG] Part {j} inline_data: type={type(part.inline_data)}")
                    print(
                        f"[DEBUG] Part {j} inline_data hasattr data: {hasattr(part.inline_data, 'data')}")

                    if hasattr(part.inline_data, "data") and part.inline_data.data:
                        try:
                            # Check if data is already bytes or if it's base64 string
                            if isinstance(part.inline_data.data, bytes):
                                image_data = part.inline_data.data
                                print(
                                    f"[DEBUG] Got raw bytes data length: {len(image_data)} bytes")
                            else:
                                image_data = base64.b64decode(
                                    part.inline_data.data)
                                print(
                                    f"[DEBUG] Decoded base64 data length: {len(image_data)} bytes")

                            print(
                                f"[DEBUG] First 20 bytes as hex: {image_data[:20].hex()}")

                            # Check for common image format headers
                            if image_data.startswith(b'\x89PNG'):
                                print(f"[DEBUG] Detected PNG format")
                            elif image_data.startswith(b'\xff\xd8\xff'):
                                print(f"[DEBUG] Detected JPEG format")
                            elif image_data.startswith(b'GIF'):
                                print(f"[DEBUG] Detected GIF format")
                            elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:20]:
                                print(f"[DEBUG] Detected WebP format")
                            else:
                                print(
                                    f"[DEBUG] Unknown format - first 100 bytes: {image_data[:100]}")

                            # Validate that it's actually image data by trying to open it
                            try:
                                test_img = Image.open(io.BytesIO(image_data))
                                print(
                                    f"[DEBUG] Successfully validated image: {test_img.size}, {test_img.mode}")
                                return image_data
                            except Exception as img_error:
                                print(
                                    f"[ERROR] Invalid image data: {img_error}")
                                print(
                                    f"[DEBUG] Data length: {len(image_data)} bytes")
                                continue

                        except Exception as decode_error:
                            print(
                                f"[ERROR] Failed to process data: {decode_error}")
                            print(
                                f"[DEBUG] Raw data type: {type(part.inline_data.data)}")
                            print(
                                f"[DEBUG] Raw data length: {len(part.inline_data.data) if hasattr(part.inline_data.data, '__len__') else 'no length'}")
                            continue
                else:
                    print(
                        f"[DEBUG] Part {j} has no inline_data or inline_data is None")
                    if hasattr(part, 'text'):
                        print(
                            f"[DEBUG] Part {j} text content: {part.text[:200]}...")

        raise RuntimeError(
            "Nano image generation returned no valid image data.")

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

        try:
            resp = self.client.models.generate_content(
                model=model, contents=parts)
        except Exception as e:
            print(f"[ERROR] Nano edit API call failed: {e}")
            raise RuntimeError(f"Nano edit API call failed: {e}")

        if not hasattr(resp, 'candidates') or not resp.candidates:
            raise RuntimeError("Nano edit returned no candidates.")

        for cand in resp.candidates:
            if not hasattr(cand, 'content') or not cand.content.parts:
                continue

            for part in cand.content.parts:
                if hasattr(part, "inline_data") and part.inline_data and hasattr(part.inline_data, "data") and part.inline_data.data:
                    try:
                        # Handle both bytes and base64 string data
                        if isinstance(part.inline_data.data, bytes):
                            image_data = part.inline_data.data
                        else:
                            image_data = base64.b64decode(
                                part.inline_data.data)
                        # Validate image data
                        test_img = Image.open(io.BytesIO(image_data))
                        return image_data
                    except Exception as e:
                        print(
                            f"[ERROR] Invalid image data in edit response: {e}")
                        continue

        raise RuntimeError("Nano edit returned no valid image data.")


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
    return CHAR_REF_NANO_TPL.format(appearance=c.appearance, STYLE_PRESET=STYLE_PRESET)


def build_scene_ref_prompt(s: Scene) -> str:
    return SCENE_REF_NANO_TPL.format(setting=s.setting, description=s.description, STYLE_PRESET=STYLE_PRESET)


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
    """
    Build instruction for Nano to convert PIL text rectangles into proper speech bubbles.
    This assumes the image already has mechanically-placed text in rectangles.
    Includes character positioning context for better bubble tail placement.
    """
    parts = []
    if p.dialogue:
        speaker_list = ", ".join([d.speaker for d in p.dialogue])
        parts.append(
            f"Convert the white text rectangles into proper comic speech bubbles. The speakers are: {speaker_list}")

        # Add character positioning context
        if p.characterPositions:
            position_info = []
            for char_pos in p.characterPositions:
                if char_pos.prominence != "off_screen":
                    # Convert positions to descriptive terms
                    x_desc = "center"
                    if char_pos.x < 0.33:
                        x_desc = "left side"
                    elif char_pos.x > 0.67:
                        x_desc = "right side"

                    y_desc = "middle"
                    if char_pos.y < 0.33:
                        y_desc = "upper area"
                    elif char_pos.y > 0.67:
                        y_desc = "lower area"

                    position_info.append(
                        f"{char_pos.name} is positioned in the {x_desc}, {y_desc} of the frame")

            if position_info:
                parts.append(
                    "Character positions for bubble tail reference: " + "; ".join(position_info))

        parts.append(
            "Add bubble tails pointing toward the appropriate characters when possible")
        parts.append(
            "DO NOT move or reposition the text - keep the text in exactly the same location")
        parts.append(
            "Only change the white rectangular border into a rounded speech bubble shape and add a tail")

    if p.narration:
        parts.append(
            "Convert the yellow narration rectangle into a proper comic caption box")
        parts.append("Keep the narration text in exactly the same position")
        parts.append(
            "Only change the rectangular border into a caption box style")

    if p.sfx:
        parts.append("Add small SFX labels: " + ", ".join(p.sfx))

    if parts:
        instruction = "Transform the text rectangles in this image into proper comic book elements:\n\n" + \
            "\n".join(f"- {part}" for part in parts)
        instruction += "\n\nCRITICAL REQUIREMENTS:"
        instruction += "\n- Keep ALL existing text in exactly the same position and formatting"
        instruction += "\n- Do NOT move, resize, or reformat any text"
        instruction += "\n- Only change the rectangular containers into bubble/caption shapes"
        instruction += "\n- Add bubble tails/pointers pointing toward the correct speaker based on their position"
        instruction += "\n- Preserve all line breaks and text wrapping exactly as shown"
        instruction += "\n- Ensure bubble tails don't obscure important visual elements"
        return instruction
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

    # Character refs via Nano banana Flash
    print(">> Generating character references (Nano Banana Flash)...")
    name_to_ref_bytes: Dict[str, bytes] = {}
    for c in characters:
        prompt = build_char_ref_prompt(c)
        logger.log(f"CHARACTER_REF_NANO_PROMPT [{c.name}]", prompt)
        raw = g.generate_image_with_nano(prompt)
        img = pad_to_square(image_bytes_to_pil(raw), PANEL_SIZE)
        png = pil_to_png_bytes(img)
        fname = f"{slugify(c.name)}.png"
        (chars_dir / fname).write_bytes(png)
        name_to_ref_bytes[c.name.lower()] = png
        print(f"   ✓ {c.name} -> {fname}")

    # Scene refs via Nano banana Flash
    print(">> Generating scene references (Nano bananaFlash)...")
    scene_to_ref_bytes: Dict[str, bytes] = {}
    for s in scenes:
        prompt = build_scene_ref_prompt(s)
        logger.log(f"SCENE_REF_NANO_PROMPT [{s.name}]", prompt)
        raw = g.generate_image_with_nano(prompt)
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
            # Combined approach: Generate scene + characters without dialogue first
            base_instruction = build_panel_base_instruction(p, characters)
            logger.log(
                f"PANEL_BASE_NANO_PROMPT [#{p.index}]", base_instruction)

            try:
                base_png = g.edit_image_with_nano(
                    scene_ref_bytes, base_instruction, ref_images=ref_images_with_continuity)
                base_img = resize_to_fill(
                    image_bytes_to_pil(base_png), PANEL_SIZE)
                base_png = pil_to_png_bytes(base_img)
                print(f"   ✓ Combined base {p.index} -> scene + characters")
            except Exception as e:
                print(
                    f"   ! Combined base generation failed on panel {p.index}: {e}. Using scene reference.")
                base_img = resize_to_fill(
                    image_bytes_to_pil(scene_ref_bytes), PANEL_SIZE)
                base_png = pil_to_png_bytes(base_img)

            # Save base image for compatibility
            base_fname = f"base-panel-{p.index:03d}.png"
            (panels_dir / base_fname).write_bytes(base_png)

            # Step 1: Add PIL text rectangles to base panel (if dialogue/narration exists)
            if p.dialogue or p.narration:
                # Add mechanical text rectangles using PIL
                base_img_pil = image_bytes_to_pil(base_png)
                text_img_pil = add_text_rectangles_to_panel(base_img_pil, p)
                text_png = pil_to_png_bytes(text_img_pil)

                # Save intermediate version with PIL text rectangles
                text_fname = f"text-panel-{p.index:03d}.png"
                (panels_dir / text_fname).write_bytes(text_png)
                print(f"   ✓ Combined PIL text {p.index} -> {text_fname}")

                # Step 2: Pass to Nano to convert rectangles to speech bubbles
                edit_instruction = build_nano_edit_instruction(p)
                logger.log(
                    f"PANEL_EDIT_NANO_PROMPT [#{p.index}]", edit_instruction)

                try:
                    edited_png = g.edit_image_with_nano(
                        text_png, edit_instruction, ref_images=char_ref_imgs)
                    final_img = resize_to_fill(
                        image_bytes_to_pil(edited_png), PANEL_SIZE)
                    final_png = pil_to_png_bytes(final_img)
                    print(
                        f"   ✓ Combined bubble conversion {p.index} -> converted rectangles to speech bubbles")
                except Exception as e:
                    # Fallback: keep PIL text version if Nano edit fails
                    print(
                        f"   ! Combined bubble conversion failed on panel {p.index}: {e}. Keeping PIL text version.")
                    final_img = text_img_pil
                    final_png = text_png
            else:
                # No dialogue or narration, use base image
                final_img = base_img
                final_png = base_png
                print(
                    f"   ✓ Combined panel {p.index} -> no dialogue/narration, using base image")
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

            # Step 1: Add PIL text rectangles to base panel (if dialogue/narration exists)
            if p.dialogue or p.narration:
                # Add mechanical text rectangles using PIL
                base_img_pil = image_bytes_to_pil(base_png)
                text_img_pil = add_text_rectangles_to_panel(base_img_pil, p)
                text_png = pil_to_png_bytes(text_img_pil)

                # Save intermediate version with PIL text rectangles
                text_fname = f"text-panel-{p.index:03d}.png"
                (panels_dir / text_fname).write_bytes(text_png)
                print(f"   ✓ PIL text {p.index} -> {text_fname}")

                # Step 2: Pass to Nano to convert rectangles to speech bubbles
                edit_instruction = build_nano_edit_instruction(p)
                logger.log(
                    f"PANEL_EDIT_NANO_PROMPT [#{p.index}]", edit_instruction)

                try:
                    edited_png = g.edit_image_with_nano(
                        text_png, edit_instruction, ref_images=char_ref_imgs)
                    final_img = resize_to_fill(
                        image_bytes_to_pil(edited_png), PANEL_SIZE)
                    final_png = pil_to_png_bytes(final_img)
                    print(
                        f"   ✓ Nano bubble conversion {p.index} -> converted rectangles to speech bubbles")
                except Exception as e:
                    # Fallback: keep PIL text version if Nano edit fails
                    print(
                        f"   ! Nano bubble conversion failed on panel {p.index}: {e}. Keeping PIL text version.")
                    final_img = text_img_pil
                    final_png = text_png
            else:
                # No dialogue or narration, skip text processing entirely to save costs
                final_img = image_bytes_to_pil(base_png)
                final_png = base_png
                print(
                    f"   ✓ Panel {p.index} -> no dialogue/narration, skipping text processing (cost savings)")

        # Save final panel image
        fname = f"panel-{p.index:03d}.png"
        (panels_dir / fname).write_bytes(final_png)
        # Convert structured dialogue back to simple format for manifest
        dialogue_strings = [f"{d.speaker}: {d.text}" for d in p.dialogue]

        # Determine text file name if dialogue/narration exists
        text_file = None
        if p.dialogue or p.narration:
            text_file = f"panels/text-panel-{p.index:03d}.png"

        panel_manifest.append({"index": p.index, "file": f"panels/{fname}",
                              "base_file": f"panels/{base_fname}",
                               "text_file": text_file,  # New field for PIL text version
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
