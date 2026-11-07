
import json
from pathlib import Path

BASE = Path(__file__).parent

questions = []

with open(BASE / "questions.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))

html = [
    """
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
    <meta charset="UTF-8">
    <title>VLM Benchmark Review</title>
    <style>
        body { font-family: sans-serif; margin: 30px; }
        .item { border: 1px solid #ccc; padding: 16px;
                margin-bottom: 24px; border-radius: 8px; }
        img { max-width: 480px; max-height: 360px;
              display: block; margin: 12px 0; }
        .answer { color: #176b37; }
    </style>
    </head>
    <body>
    <h1>VLM Benchmark Ground Truth Review</h1>
    """
]

for q in questions:
    image_path = q["image"]

    gt = q.get("ground_truth", q.get("reference_captions", []))

    html.append(f"""
    <div class="item">
        <h2>Question {q['id']} - {q['category']}</h2>
        <img src="{image_path}" alt="Question {q['id']}">
        <p><b>Prompt:</b> {q['prompt']}</p>
        <p class="answer"><b>Ground Truth:</b> {json.dumps(gt, ensure_ascii=False)}</p>
        <p><b>Verified:</b> {q.get('ground_truth_verified', False)}</p>
    </div>
    """)

html.append("</body></html>")

output = BASE / "review_benchmark.html"
output.write_text("\n".join(html), encoding="utf-8")

print("Review page created:", output)
