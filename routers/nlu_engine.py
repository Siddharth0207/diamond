import re

def extract_entities(text: str) -> dict:
    entities = {}
    # Handle updates like "change clarity to SI1"
    if match := re.search(r"(change|set|update)\s+(clarity|carat|color|price|shape)\s+(to)?\s*([a-zA-Z0-9.]+)", text.lower()):
        key = match.group(2)
        value = match.group(4)
        entities[key] = value.upper() if key in {"clarity", "color"} else value

    # Carat (float)
    if match := re.search(r"(\d+(\.\d+)?)\s*carat", text.lower()):
        entities["carat"] = float(match.group(1))

    # Clarity
    if match := re.search(r"\b(IF|VVS1|VVS2|VS1|VS2|SI1|SI2)\b", text.upper()):
        entities["clarity"] = match.group(1)

    # Color
    if match := re.search(r"\b([D-K])\b", text.upper()):
        entities["color"] = match.group(1)

    # Price
    if match := re.search(r"under\s*\$?(\d+)", text.lower()):
        entities["price"] = {"max": int(match.group(1))}
    elif match := re.search(r"over\s*\$?(\d+)", text.lower()):
        entities["price"] = {"min": int(match.group(1))}

    # Shape
    if match := re.search(r"\b(round|oval|princess|emerald|pear|marquise|cushion|radiant)\b", text.lower()):
        entities["shape"] = match.group(1)

    return entities
