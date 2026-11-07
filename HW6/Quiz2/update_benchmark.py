import json
import shutil
from pathlib import Path

path = Path("benchmark/questions.jsonl")
backup = path.with_suffix(".jsonl.bak")
shutil.copy2(path, backup)

questions = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

replacements = {
    "coco_000000565776.jpg": {
        "new_image": "images/coco_000000558213.jpg",
        "question": "請判斷這張照片主要呈現什麼活動場景？",
        "ground_truth": "滑板運動場景",
        "category": "image_classification",
    },
    "coco_000000034257.jpg": {
        "new_image": "images/coco_000000257169.jpg",
        "question": "照片中人物手持的主要物品是什麼？",
        "ground_truth": "吹風機",
        "category": "object_recognition",
    },
}

def get_category(q):
    for key in ("category", "task_type", "type"):
        if key in q:
            return q[key]
    return None

def get_question_key(q):
    for key in ("question", "prompt", "instruction"):
        if key in q:
            return key
    raise KeyError("找不到題目文字欄位")

for old_image, data in replacements.items():
    target = [
        q for q in questions
        if q.get("image", "").endswith(old_image)
        and get_category(q) == data["category"]
    ]

    if len(target) != 1:
        raise ValueError(
            f"{old_image} 找到 {len(target)} 題，預期恰好 1 題，未修改檔案。"
        )

    q = target[0]
    q["image"] = data["new_image"]
    q[get_question_key(q)] = data["question"]
    q["ground_truth"] = data["ground_truth"]

    if "ground_truth_verified" in q:
        q["ground_truth_verified"] = False
    if "status" in q:
        q["status"] = "pending"

    print(f"已替換：{old_image} -> {data['new_image']}")

# 檢查題數及圖片唯一性
images = [q["image"] for q in questions]
assert len(questions) == 20
assert len(set(images)) == 20, "仍有重複圖片！"

path.write_text(
    "\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n",
    encoding="utf-8"
)

print("\n完成！")
print("題目數：", len(questions))
print("不重複圖片數：", len(set(images)))
print("備份位置：", backup)
