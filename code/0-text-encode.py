"""Encode the battery CSV files as text.

Each sensor row becomes one line "<column> is <value> and ...", rounded to two
decimals, and a second variant with numbers spelled out by num2words. The
outputs are the files in data/ used by the following steps.
"""
import re
from pathlib import Path

import pandas as pd
from num2words import num2words

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEPARATOR = " is "


def text_encoder(separator, data):
    encoded = pd.DataFrame([data.columns] * len(data), columns=data.columns) + separator + data.astype(str)
    return encoded.apply(lambda x: " and ".join(x), axis=1)


def text_encode(data, file, prefix):
    stem = str(file).replace(".csv", "")
    suffix = SEPARATOR.strip()
    encoded = text_encoder(SEPARATOR, data)
    encoded.to_csv(f"{stem}_{prefix}_{suffix}.csv", index=False, header=False, float_format="%.2f")
    encoded.apply(
        lambda x: re.sub(r"[-+]?(?:\d*\.*\d+)", lambda m: num2words(m.group()), x)
    ).to_csv(f"{stem}_{prefix}_{suffix}_num2words.csv", index=False, header=False, float_format="%.2f")


if __name__ == "__main__":
    # The healthy file keeps the spelling "roundend" of the published artifact.
    for name, prefix in [
        ("battery_healthy.csv", "roundend_and_text_encoded_with"),
        ("battery_runtime_fault1.csv", "rounded_and_text_encoded_with"),
    ]:
        file = DATA_DIR / name
        text_encode(pd.read_csv(file).dropna().round(2), file, prefix)
