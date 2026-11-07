
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "coco_candidates.json", encoding="utf-8") as f:
    images = json.load(f)

image_map = {
    item["image_id"]: item
    for item in images
}

# 20 manually designed benchmark questions
definitions = [
    # 1-4: Image Classification
    {
        "id": 1,
        "image_id": 435208,
        "category": "image_classification",
        "prompt": "請判斷這張圖片主要呈現哪一種室內場景，並簡短說明判斷依據。",
        "ground_truth": "客廳（Living room）"
    },
    {
        "id": 2,
        "image_id": 565776,
        "category": "image_classification",
        "prompt": "請判斷這張圖片主要呈現哪一種室內空間。",
        "ground_truth": "廚房（Kitchen）"
    },
    {
        "id": 3,
        "image_id": 357816,
        "category": "image_classification",
        "prompt": "請判斷圖片中的人物正在參與哪一種運動。",
        "ground_truth": "棒球（Baseball）"
    },
    {
        "id": 4,
        "image_id": 88432,
        "category": "image_classification",
        "prompt": "請判斷這張圖片主要呈現哪一種交通場景。",
        "ground_truth": "城市街道交叉路口（Street intersection）"
    },

    # 5-8: Object Recognition
    {
        "id": 5,
        "image_id": 416885,
        "category": "object_recognition",
        "prompt": "請列出圖片中可以辨識的主要物件，使用條列式回答。",
        "ground_truth": [
            "sandwich", "wine glass", "fork",
            "knife", "spoon", "dining table", "donut"
        ]
    },
    {
        "id": 6,
        "image_id": 231831,
        "category": "object_recognition",
        "prompt": "請列出圖片中可以辨識的主要物件，使用條列式回答。",
        "ground_truth": [
            "cat", "book", "remote",
            "cell phone", "scissors", "chair"
        ]
    },
    {
        "id": 7,
        "image_id": 193162,
        "category": "object_recognition",
        "prompt": "請列出圖片中出現的主要動物與人物，使用條列式回答。",
        "ground_truth": ["dog", "sheep", "cow", "person"]
    },
    {
        "id": 8,
        "image_id": 159977,
        "category": "object_recognition",
        "prompt": "請列出圖片中可以辨識的動物種類，使用條列式回答。",
        "ground_truth": ["zebra", "giraffe", "bird"]
    },

    # 9-12: Image Captioning
    {
        "id": 9,
        "image_id": 202339,
        "category": "image_captioning",
        "prompt": "請用繁體中文完整描述這張圖片的內容，包括人物、物件與場景。",
        "ground_truth": image_map[202339]["captions"]
    },
    {
        "id": 10,
        "image_id": 34257,
        "category": "image_captioning",
        "prompt": "請用繁體中文完整描述這張圖片的內容，包括主要物件與場景。",
        "ground_truth": image_map[34257]["captions"]
    },
    {
        "id": 11,
        "image_id": 558213,
        "category": "image_captioning",
        "prompt": "請用繁體中文描述圖片中的人物正在進行什麼活動，以及周遭環境。",
        "ground_truth": image_map[558213]["captions"]
    },
    {
        "id": 12,
        "image_id": 73946,
        "category": "image_captioning",
        "prompt": "請用繁體中文描述圖片中的人物、物件及其互動情形。",
        "ground_truth": image_map[73946]["captions"]
    },

    # 13-16: Visual Question Answering
    {
        "id": 13,
        "image_id": 496954,
        "category": "visual_question_answering",
        "prompt": "圖片中的主要食物是什麼？",
        "ground_truth": "蛋糕（Cake）"
    },
    {
        "id": 14,
        "image_id": 92091,
        "category": "visual_question_answering",
        "prompt": "圖片中的小女孩除了嘴裡的牙刷之外，手上還拿著什麼？",
        "ground_truth": "泰迪熊（Teddy bear）"
    },
    {
        "id": 15,
        "image_id": 369757,
        "category": "visual_question_answering",
        "prompt": "圖片中展示的主要食物是什麼？",
        "ground_truth": "披薩（Pizza）"
    },
    {
        "id": 16,
        "image_id": 33114,
        "category": "visual_question_answering",
        "prompt": "圖片中可以看到哪一種航空器？",
        "ground_truth": "飛機（Airplane）"
    },

    # 17-20: Attribute & Spatial Understanding
    {
        "id": 17,
        "image_id": 261888,
        "category": "attribute_spatial",
        "prompt": "圖片中總共有幾匹馬？請回答數字。",
        "ground_truth": "2"
    },
    {
        "id": 18,
        "image_id": 482719,
        "category": "attribute_spatial",
        "prompt": "圖片中的碗裡有幾根香蕉？請回答數字。",
        "ground_truth": "1"
    },
    {
        "id": 19,
        "image_id": 428454,
        "category": "attribute_spatial",
        "prompt": "圖片中的人物站在什麼運動器材上？",
        "ground_truth": "Snowboard（滑雪板）"
    },
    {
        "id": 20,
        "image_id": 257169,
        "category": "attribute_spatial",
        "prompt": "圖片中的人物手上拿著什麼物品？",
        "ground_truth": "Hair dryer（吹風機）"
    },
]

# Generate JSONL
output_path = BASE_DIR / "questions.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for item in definitions:
        image = image_map[item["image_id"]]

        record = {
            "id": item["id"],
            "image_id": item["image_id"],
            "image": image["image"],
            "category": item["category"],
            "prompt": item["prompt"],
            "ground_truth": item["ground_truth"],
        }

        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

print(f"Created {output_path}")
print(f"Total questions: {len(definitions)}")

from collections import Counter

counts = Counter(item["category"] for item in definitions)

for category, count in counts.items():
    print(f"{category}: {count}")
