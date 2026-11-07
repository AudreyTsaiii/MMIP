import json
from pathlib import Path

DATA = Path(__file__).parent / "questions.jsonl"

def save(questions):
    DATA.write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n",
        encoding="utf-8"
    )

def main():
    questions = [
        json.loads(line)
        for line in DATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(f"共 {len(questions)} 題，開始人工確認 Ground Truth。")
    print("y=正確  e=修改  n=不接受  s=跳過  q=離開\n")

    for i, q in enumerate(questions):
        if q.get("ground_truth_verified") is True:
            continue

        print("=" * 60)
        print(f"題目 {i+1}/{len(questions)}")
        print("圖片：", q.get("image"))
        print("問題：", q.get("question", q.get("prompt", "")))
        print("Ground Truth：", q.get("ground_truth"))
        print("=" * 60)

        while True:
            action = input("請輸入 y/e/n/s/q：").strip().lower()

            if action == "y":
                q["ground_truth_verified"] = True
                q["status"] = "verified"
                break

            elif action == "e":
                new_gt = input("請輸入正確 Ground Truth：").strip()
                if not new_gt:
                    print("答案不可空白。")
                    continue
                q["ground_truth"] = new_gt
                q["ground_truth_verified"] = False
                q["status"] = "revised"
                print("已修改，請再次確認新答案。")
                continue

            elif action == "n":
                q["ground_truth_verified"] = False
                q["status"] = "rejected"
                break

            elif action == "s":
                break

            elif action == "q":
                save(questions)
                print("已儲存，離開審核。")
                return

            else:
                print("請輸入 y/e/n/s/q。")

        save(questions)

    verified = sum(q.get("ground_truth_verified") is True for q in questions)
    print(f"\n審核完成：{verified}/{len(questions)} 題已確認。")

if __name__ == "__main__":
    main()
