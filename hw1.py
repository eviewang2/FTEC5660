#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek
    from langchain_core.output_parsers import JsonOutputParser

    system_text = """You read Hong Kong supermarket receipts. Output ONLY JSON in this format:
{{
  "items": [{{"name": "<text>", "amount": <number>}}],
  "discounts": [{{"label": "<text>", "amount": <number>}}],
  "subtotal": <number>,
  "rounding": <number>,
  "payment_amount": <number>
}}
Rules:
- items: EVERY positive product line above the subtotal (including $0.00 lines and
  plastic bag charges). Use the line total on the right (already multiplied by QTY/數量).
- discounts: EVERY negative line above the subtotal (Buy N Save, % OFF, coupons,
  member offers, 包裝變形 ...). Write amount as a POSITIVE number. Empty list if none.
- subtotal: the number on the SUBTOTAL / 小計 line.
- rounding: the ROUNDING line with its sign (e.g. -0.01). 0 if there is none.
- payment_amount: the final amount paid after rounding (OCTOPUS / 八達通 / VISA / CASH line).
  If cash was given with change, use subtotal + rounding.
- Ignore card numbers, remaining balance (餘額), 扣除金額, change (找續), dates and item codes.
- Check: sum(items) - sum(discounts) must equal subtotal."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", [
            {"type": "text", "text": "Read this receipt and output the JSON.{feedback}"},
            {"type": "image_url", "image_url": {"url": "{image_url}"}},
        ]),
    ])
    llm = ChatDeepSeek(model="deepseek-v4-flash-vision-exp", temperature=0)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    return chain

def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
     ### YOUR CODE HERE
    def is_consistent(data):
        """自我检查：商品之和 - 折扣之和 应该等于 小计。"""
        try:
            items_sum = sum(Decimal(str(x["amount"])) for x in data["items"])
            discount_sum = sum(Decimal(str(d["amount"])) for d in data["discounts"])
            subtotal = Decimal(str(data["subtotal"]))
            return abs(items_sum - discount_sum - subtotal) < Decimal("0.01")
        except Exception:
            return False

    urls = [image_data_url(p) for p in images]

    # 第一轮：所有小票一起读
    results = chain.batch(
        [{"image_url": u, "feedback": ""} for u in urls],
        return_exceptions=True,
    )

    # 反思：检查没通过的小票，带着提示重读，最多重读 2 轮
    feedback = (
        "\nYour previous answer failed the check: sum(items) - sum(discounts) "
        "did not equal subtotal. Re-read every line and every digit carefully."
    )
    for _ in range(2):
        bad = [i for i, d in enumerate(results) if not is_consistent(d)]
        if not bad:
            break
        retry = chain.batch(
            [{"image_url": urls[i], "feedback": feedback} for i in bad],
            return_exceptions=True,
        )
        for i, d in zip(bad, retry):
            results[i] = d

    total_paid = Decimal("0")
    total_original = Decimal("0")
    for data in results:
        if not isinstance(data, dict):
            continue
        paid = Decimal(str(data["payment_amount"]))
        subtotal = Decimal(str(data["subtotal"]))
        discount_sum = sum(Decimal(str(d["amount"])) for d in data["discounts"])
        total_paid += paid
        total_original += subtotal + discount_sum

    return {QUERY_1: f"HK${total_paid:.2f}", QUERY_2: f"HK${total_original:.2f}"}

# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
