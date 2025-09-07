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

# Google AI SDK (for text generation only)
from google import genai
from google.genai import types

# Fal AI SDK for image generation
import fal_client
import requests

# ------------------ ENV & CONFIG ------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

# Fal API setup
FAL_API_KEY = os.getenv("FAL_API_KEY")
if not FAL_API_KEY:
    raise RuntimeError("Missing FAL_API_KEY in .env")
os.environ["FAL_KEY"] = FAL_API_KEY

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

# Load custom story generation template
CUSTOM_STORY_TEMPLATE = (Path(__file__).parent / "stories" /
                         "custom_story.txt").read_text(encoding="utf-8")

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


def find_right_side_position(dialog_index: int, used_areas: List[tuple],
                             rect_width: int, rect_height: int,
                             img_width: int, img_height: int) -> tuple:
    """
    Find position for dialog on the right side, stacking multiple dialogs vertically.
    First dialogue appears at top, subsequent ones below for natural reading order.
    """
    # Base position: top right with some margin (changed from bottom right)
    margin = 10
    base_x = img_width - rect_width - margin
    base_y = margin  # Start from top instead of bottom

    # For multiple dialogs, stack them vertically downward
    vertical_spacing = 5  # Small gap between dialog boxes

    # Calculate vertical offset based on dialog index
    # Each subsequent dialog goes lower down (natural reading order)
    vertical_offset = dialog_index * (rect_height + vertical_spacing)

    pos_x = base_x
    pos_y = base_y + vertical_offset  # Add offset instead of subtract

    # Ensure the dialog doesn't go off the bottom of the panel
    pos_y = min(pos_y, img_height - rect_height - margin)

    # Ensure it doesn't go off the left side either
    pos_x = max(margin, min(pos_x, img_width - rect_width - margin))

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
    font_size = 32  # Reduced to 70% of previous size (46) for better fit
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

        # Position dialog on right side (diagonal from narrator text at top left)
        rect_x, rect_y = find_right_side_position(
            i, used_areas, rect_width, rect_height, width, height
        )

        # Draw white rectangle with rounded appearance using black border
        draw.rectangle([rect_x, rect_y, rect_x + rect_width, rect_y + rect_height],
                       fill="white", outline="black", width=3)
        # Add inner shadow effect for depth
        draw.rectangle([rect_x+1, rect_y+1, rect_x + rect_width-1, rect_y + rect_height-1],
                       fill=None, outline="lightgray", width=1)

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

        # Position at top left for narration
        rect_x = 10  # 10px from left edge
        rect_y = 10  # 10px from top edge

        # Ensure rectangle fits within image bounds
        rect_x = max(0, min(rect_x, width - rect_width))
        rect_y = max(0, min(rect_y, height - rect_height))

        # Draw light gray rectangle with black border for narration
        draw.rectangle([rect_x, rect_y, rect_x + rect_width, rect_y + rect_height],
                       fill="lightgray", outline="black", width=2)

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


def create_title_panel(g: "GAIC", title: str, logger: "PromptLogger", panel_size: int = PANEL_SIZE) -> bytes:
    """Create a title panel with the comic title using Nano Banana image generation."""
    # Build the title generation prompt
    title_prompt = f"""Please generate a comic book title image with the text "{title}". 
    
Style requirements:
- {STYLE_PRESET}
- Bold, dramatic comic book title design
- Eye-catching typography that fits the comic theme
- Professional comic book cover style
- The title text should be prominent and clearly readable
- Use dynamic composition and striking visual design
- Make it look like a proper comic book title panel

The title text is: "{title}"

Create a visually striking title image that would work as the opening panel of a comic book."""

    logger.log("TITLE_PANEL_NANO_PROMPT", title_prompt)

    try:
        # Generate title panel using Nano Banana
        title_image_bytes = g.generate_image_with_nano(title_prompt)

        # Resize to proper panel size
        title_img = resize_to_fill(
            image_bytes_to_pil(title_image_bytes), panel_size)
        title_png = pil_to_png_bytes(title_img)

        return title_png

    except Exception as e:
        print(
            f"   ! Nano title generation failed: {e}. Using fallback plain text title.")
        logger.log("TITLE_PANEL_FALLBACK",
                   f"Nano generation failed: {e}, using fallback")

        # Fallback to original plain text implementation
        canvas = Image.new("RGB", (panel_size, panel_size), "white")
        draw = ImageDraw.Draw(canvas)

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
                    font = ImageFont.truetype(font_path, 72)
                    break
                except:
                    continue
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        # Calculate title position (centered)
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]
        title_x = (panel_size - title_width) // 2
        title_y = (panel_size - title_height) // 2

        # Draw title background
        padding = 20
        draw.rectangle([title_x - padding, title_y - padding,
                       title_x + title_width + padding, title_y + title_height + padding],
                       fill="black")
        draw.text((title_x, title_y), title, fill="white", font=font)

        return pil_to_png_bytes(canvas)


def create_comic_layout(panels: List[bytes], title: str, g: "GAIC", logger: "PromptLogger", panel_size: int = PANEL_SIZE) -> Image.Image:
    """Stitch panels together into a comic book layout with title panel as first panel."""
    if not panels:
        raise ValueError("No panels to stitch together")

    # Create title panel and add it as the first panel
    title_panel_bytes = create_title_panel(g, title, logger, panel_size)
    all_panels = [title_panel_bytes] + panels

    # Calculate layout - 2 columns max
    num_panels = len(all_panels)
    cols = 2
    rows = (num_panels + 1) // 2  # Ceiling division

    # Comic styling
    border_width = 8
    panel_spacing = 12
    margin = 20

    # Calculate canvas size (no separate title area needed)
    canvas_width = cols * panel_size + (cols - 1) * panel_spacing + 2 * margin
    canvas_height = rows * panel_size + (rows - 1) * panel_spacing + 2 * margin

    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    # Place panels
    for i, panel_bytes in enumerate(all_panels):
        row = i // cols
        col = i % cols

        x = margin + col * (panel_size + panel_spacing)
        y = margin + row * (panel_size + panel_spacing)

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
        Generate an image using Fal AI nano banana model.
        If base_image is provided, edit that image. Otherwise, generate from scratch.
        Returns PNG bytes.
        """
        try:
            if base_image:
                # Edit mode: use image-to-image editing
                base64_data = base64.b64encode(base_image).decode('utf-8')
                data_uri = f"data:image/png;base64,{base64_data}"

                result = fal_client.subscribe(
                    "fal-ai/nano-banana/edit",
                    arguments={
                        "prompt": prompt,
                        "image_urls": [data_uri],
                        "num_images": 1,
                        "output_format": "png"
                    },
                    with_logs=True,
                )
            else:
                # Generate from scratch: text-to-image
                result = fal_client.subscribe(
                    "fal-ai/nano-banana",
                    arguments={
                        "prompt": prompt,
                        "num_images": 1,
                        "output_format": "png"
                    },
                    with_logs=True,
                )

            if result.get('images') and len(result['images']) > 0:
                image_url = result['images'][0]['url']

                # Download the image
                response = requests.get(image_url)
                if response.status_code == 200:
                    # Validate that it's actually image data
                    test_img = Image.open(io.BytesIO(response.content))
                    print(
                        f"[DEBUG] Fal image generated: {test_img.size}, {test_img.mode}")
                    return response.content
                else:
                    raise RuntimeError(
                        f"Failed to download image from Fal: {response.status_code}")
            else:
                raise RuntimeError("Fal API returned no images")

        except Exception as e:
            print(f"[ERROR] Fal API call failed: {e}")
            raise RuntimeError(f"Fal image generation failed: {e}")

    def edit_image_with_nano(self, base_image: bytes, instruction: str, ref_images: Optional[List[bytes]] = None,
                             model: str = NANO_EDIT_MODEL) -> bytes:
        """
        Edit an image using Fal AI nano banana edit model.
        ref_images are additional reference images to guide the editing.
        Returns PNG bytes.
        """
        try:
            # Optimize image sizes to avoid request body size limits
            def optimize_image_for_api(img_bytes: bytes, max_size: int = 512) -> bytes:
                """Compress image to reduce API payload size"""
                img = Image.open(io.BytesIO(img_bytes))
                if img.size[0] > max_size or img.size[1] > max_size:
                    img.thumbnail((max_size, max_size), Image.LANCZOS)

                # Convert to JPEG for smaller size (API accepts JPEG)
                output = io.BytesIO()
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Convert to RGB for JPEG
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split(
                    )[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                img.save(output, format='JPEG', quality=85, optimize=True)
                return output.getvalue()

            # Optimize base image
            optimized_base = optimize_image_for_api(base_image)
            base64_data = base64.b64encode(optimized_base).decode('utf-8')
            image_urls = [f"data:image/jpeg;base64,{base64_data}"]

            # Add reference images if provided, but limit to 1-2 most important ones
            if ref_images:
                # Limit to first 2 reference images to avoid payload size issues
                for ref_img in ref_images[:2]:
                    optimized_ref = optimize_image_for_api(ref_img)
                    ref_base64 = base64.b64encode(
                        optimized_ref).decode('utf-8')
                    image_urls.append(f"data:image/jpeg;base64,{ref_base64}")

            # Enhanced prompt with reference image information
            enhanced_instruction = instruction
            if ref_images:
                enhanced_instruction += " Use the additional reference images to maintain character consistency and visual style."

            result = fal_client.subscribe(
                "fal-ai/nano-banana/edit",
                arguments={
                    "prompt": enhanced_instruction,
                    "image_urls": image_urls,
                    "num_images": 1,
                    "output_format": "png"
                },
                with_logs=True,
            )

            if result.get('images') and len(result['images']) > 0:
                image_url = result['images'][0]['url']

                # Download the edited image
                response = requests.get(image_url)
                if response.status_code == 200:
                    # Validate image data
                    test_img = Image.open(io.BytesIO(response.content))
                    print(
                        f"[DEBUG] Fal image edited: {test_img.size}, {test_img.mode}")
                    return response.content
                else:
                    raise RuntimeError(
                        f"Failed to download edited image from Fal: {response.status_code}")
            else:
                raise RuntimeError("Fal API returned no edited images")

        except Exception as e:
            print(f"[ERROR] Fal edit API call failed: {e}")
            raise RuntimeError(f"Fal image editing failed: {e}")


def generate_story_from_user_input(g: GAIC, user_input: str, log: PromptLogger) -> str:
    """Generate a story based on user input using the custom story template."""
    prompt = fill(CUSTOM_STORY_TEMPLATE, user_input=user_input)
    log.log("STORY_GENERATION_PROMPT", prompt)

    story = g.generate_text(prompt)
    log.log("STORY_GENERATION_RESPONSE", story)

    # Extract just the story part if there's a title
    lines = story.strip().split('\n')
    # Skip title line if it looks like a title (short line followed by empty line)
    if len(lines) > 2 and len(lines[0].strip()) < 100 and lines[1].strip() == '':
        story = '\n'.join(lines[2:]).strip()

    return story


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


def run_pipeline(story_text: str, out_root: Path, user_input: str = None):
    """
    Run the comic generation pipeline.

    Args:
        story_text: The story text to use (if user_input is None)
        out_root: Output directory
        user_input: If provided, generate story from this input instead of using story_text
    """
    ensure_dir(out_root)
    logger = PromptLogger(out_root / "prompts_used.txt")
    g = GAIC(API_KEY)

    # Generate story if user_input is provided
    if user_input:
        print(">> Generating story from user input...")
        story_text = generate_story_from_user_input(g, user_input, logger)
        print(f"   Generated story: {len(story_text)} characters")

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

    # CRITICAL: Reload all character images from disk before panel generation
    # This ensures any uploaded custom sprites are used instead of cached originals
    print(">> Reloading character sprites from disk...")
    for c in characters:
        fname = f"{slugify(c.name)}.png"
        char_file = chars_dir / fname
        if char_file.exists():
            # Reload the character image from disk (may be user-uploaded)
            char_png = char_file.read_bytes()
            name_to_ref_bytes[c.name.lower()] = char_png
            print(f"   ✓ Reloaded {c.name} -> {fname}")

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

            # Add PIL text rectangles to base panel (if dialogue/narration exists)
            if p.dialogue or p.narration:
                # Add mechanical text rectangles using PIL - these will be the final text boxes
                base_img_pil = image_bytes_to_pil(base_png)
                text_img_pil = add_text_rectangles_to_panel(base_img_pil, p)
                text_png = pil_to_png_bytes(text_img_pil)

                # Save intermediate version with PIL text rectangles
                text_fname = f"text-panel-{p.index:03d}.png"
                (panels_dir / text_fname).write_bytes(text_png)
                print(f"   ✓ Combined PIL text {p.index} -> {text_fname}")

                # Use PIL text version as final output (no nano editing)
                final_img = text_img_pil
                final_png = text_png
                print(
                    f"   ✓ Combined panel {p.index} -> using auto-positioned text boxes")
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

            # Add PIL text rectangles to base panel (if dialogue/narration exists)
            if p.dialogue or p.narration:
                # Add mechanical text rectangles using PIL - these will be the final text boxes
                base_img_pil = image_bytes_to_pil(base_png)
                text_img_pil = add_text_rectangles_to_panel(base_img_pil, p)
                text_png = pil_to_png_bytes(text_img_pil)

                # Save intermediate version with PIL text rectangles
                text_fname = f"text-panel-{p.index:03d}.png"
                (panels_dir / text_fname).write_bytes(text_png)
                print(f"   ✓ PIL text {p.index} -> {text_fname}")

                # Use PIL text version as final output (no nano editing)
                final_img = text_img_pil
                final_png = text_png
                print(
                    f"   ✓ Panel {p.index} -> using auto-positioned text boxes")
            else:
                # No dialogue or narration, skip text processing entirely
                final_img = image_bytes_to_pil(base_png)
                final_png = base_png
                print(
                    f"   ✓ Panel {p.index} -> no dialogue/narration, skipping text processing")

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
                               "prompt": p.prompt,  # Add the visual description prompt
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
            final_panel_bytes, comic_title, g, logger, PANEL_SIZE)
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
