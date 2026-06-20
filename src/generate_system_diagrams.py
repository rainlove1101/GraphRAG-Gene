from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 2400, 1450
BLUE = "#2F6F9F"
GREEN = "#4F8A5B"
GOLD = "#C9952E"
RED = "#A94E4E"
GRAY = "#4B5563"
INK = "#111827"
LIGHT_BLUE = "#E8F1F8"
LIGHT_GREEN = "#EAF4ED"
LIGHT_GOLD = "#FFF7E6"
LIGHT_RED = "#FCEEEE"
LIGHT_GRAY = "#F3F4F6"


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE = font(44, True)
HEAD = font(29, True)
BODY = font(25)
SMALL = font(22)


def rounded_box(draw, xy, fill, outline, radius=24, width=4):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(draw, box, text, fill=INK, fnt=BODY, max_chars=22):
    x1, y1, x2, y2 = box
    lines = []
    for raw in text.split("\n"):
        lines.extend(textwrap.wrap(raw, max_chars) or [""])
    line_heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total_h = sum(line_heights) + (len(lines) - 1) * 8
    y = y1 + (y2 - y1 - total_h) / 2
    for line, lh in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=fnt)
        x = x1 + (x2 - x1 - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), line, fill=fill, font=fnt)
        y += lh + 8


def box(draw, xy, text, fill=LIGHT_GRAY, outline=GRAY):
    rounded_box(draw, xy, fill, outline)
    centered_text(draw, xy, text)


def arrow(draw, start, end, fill=GRAY, width=5):
    draw.line([start, end], fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        points = [(ex, ey), (ex - 24 * direction, ey - 14), (ex - 24 * direction, ey + 14)]
    else:
        direction = 1 if ey > sy else -1
        points = [(ex, ey), (ex - 14, ey - 24 * direction), (ex + 14, ey - 24 * direction)]
    draw.polygon(points, fill=fill)


def save_image(img, name):
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    img.save(png)
    try:
        img.convert("RGB").save(pdf)
    except Exception:
        # PNG is the canonical source used by the manuscript builder; PDF export is best-effort.
        pass


def architecture_diagram():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((W / 2, 80), "GraphRAG-Gene System Architecture", anchor="mm", font=TITLE, fill=INK)

    boxes = {
        "raw": (130, 250, 660, 460),
        "kg": (940, 250, 1490, 460),
        "index": (1780, 250, 2310, 460),
        "query": (130, 760, 660, 970),
        "retrieval": (940, 760, 1490, 970),
        "ranking": (1780, 760, 2310, 970),
        "llm": (940, 1130, 1490, 1340),
        "output": (1780, 1130, 2310, 1340),
    }

    box(d, boxes["raw"], "Reactome-derived\nraw tables\nGenes / pathways /\nrelationships", LIGHT_BLUE, BLUE)
    box(d, boxes["kg"], "Knowledge graph\nconstruction\nNode keys / filtering /\nconfidence scores", LIGHT_BLUE, BLUE)
    box(d, boxes["index"], "Graph index\nNodes / edges /\ncommunities /\nsummaries", LIGHT_GREEN, GREEN)
    box(d, boxes["query"], "User query\nMulti-gene list", LIGHT_GOLD, GOLD)
    box(d, boxes["retrieval"], "Graph retrieval\nGene matching /\npathway candidates /\ndirect relations", LIGHT_GOLD, GOLD)
    box(d, boxes["ranking"], "Specificity-aware\nranking\nTop pathways /\nevidence context", LIGHT_GOLD, GOLD)
    box(d, boxes["llm"], "Optional LLM\nsummarization\nEvidence-grounded\nreporting", LIGHT_RED, RED)
    box(d, boxes["output"], "Output\nPathway-level\ninterpretation with\nlimitations", LIGHT_RED, RED)

    arrow(d, (660, 355), (940, 355), BLUE)
    arrow(d, (1490, 355), (1780, 355), BLUE)
    arrow(d, (2045, 460), (2045, 760), GREEN)
    arrow(d, (660, 865), (940, 865), GOLD)
    arrow(d, (1490, 865), (1780, 865), GOLD)
    arrow(d, (1215, 970), (1215, 1130), RED)
    arrow(d, (1490, 1235), (1780, 1235), RED)
    arrow(d, (2045, 970), (2045, 1130), RED)

    d.text((395, 575), "Offline index construction", anchor="mm", font=HEAD, fill=BLUE)
    d.text((395, 1075), "Online query-time retrieval", anchor="mm", font=HEAD, fill=GOLD)
    d.text((W / 2, 1400), "Clinical diagnosis, personal risk prediction, variant pathogenicity assessment, and treatment recommendations are excluded.", anchor="mm", font=SMALL, fill=RED)
    save_image(img, "system_architecture")


def workflow_diagram():
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((W / 2, 80), "Experimental and Reproducibility Workflow", anchor="mm", font=TITLE, fill=INK)

    coords = [
        (130, 230, 660, 460),
        (940, 230, 1490, 460),
        (1780, 230, 2310, 460),
        (130, 735, 660, 1035),
        (940, 735, 1490, 1035),
        (1780, 735, 2310, 1035),
    ]
    steps = [
        ("1. Build index\nParse entities and relationships\nfilter by confidence\nrun Leiden clustering", LIGHT_BLUE, BLUE),
        ("2. Query cases\n36 curated multi-gene cases\nexpected pathway labels\nsynonym-aware matching", LIGHT_GOLD, GOLD),
        ("3. Baselines\nGene-only retrieval\nPathway-only retrieval\nGraphRAG-Gene", LIGHT_GREEN, GREEN),
        ("4. Ablation\nFull model\nno direct-relation bonus\nno specificity weighting\noverlap only", LIGHT_GREEN, GREEN),
        ("5. Sensitivity\nLeiden resolution sweep\ncommunity statistics\nstrict matching control", LIGHT_BLUE, BLUE),
        ("6. Outputs\nCSV result tables\ncase reports\npublication figures\nWord manuscript draft", LIGHT_RED, RED),
    ]
    for xy, (text, fill, outline) in zip(coords, steps):
        box(d, xy, text, fill, outline)

    arrow(d, (660, 345), (940, 345))
    arrow(d, (1490, 345), (1780, 345))
    arrow(d, (2045, 460), (2045, 735))
    arrow(d, (1780, 885), (1490, 885))
    arrow(d, (940, 885), (660, 885))

    d.text((W / 2, 1210), "Main benchmark: synonym-aware pathway matching. Sensitivity control: strict substring matching.", anchor="mm", font=HEAD, fill=GRAY)
    d.text((W / 2, 1280), "Reproducibility command: scripts/run_submission_experiments.ps1", anchor="mm", font=SMALL, fill=GRAY)
    save_image(img, "experimental_workflow")


def main():
    architecture_diagram()
    workflow_diagram()
    print(f"Saved diagrams to {OUT_DIR}")


if __name__ == "__main__":
    main()
