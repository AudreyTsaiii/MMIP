"""Conservative, auditable dictionary extraction for COCO object-list answers."""
import re
import unicodedata

VERSION = "coco-alias-v1"
# Only explicit object names are mapped. Ambiguous table/stove/utensils stay unmapped.
ZH = {
    "person": "人 人物 男人 女人 男子 女子 男孩 女孩 小孩 孩子 嬰兒 婴儿",
    "bicycle": "自行車 自行车 腳踏車 脚踏车 單車 单车",
    "car": "汽車 汽车 轎車 轿车", "motorcycle": "摩托車 摩托车 機車 机车",
    "airplane": "飛機 飞机", "bus": "公車 公车 公交車 公交车 巴士 客車 客车",
    "train": "火車 火车 列車 列车", "truck": "卡車 卡车 貨車 货车 皮卡車 皮卡车",
    "boat": "船 船隻 船只", "traffic light": "紅綠燈 红绿灯 交通號誌 交通信號燈 交通信号灯",
    "fire hydrant": "消防栓 消火栓", "stop sign": "停車標誌 停车标志 停止標誌 停止标志",
    "parking meter": "停車計時器 停车计时器", "bench": "長椅 长椅 長凳 长凳",
    "bird": "鳥 鸟", "cat": "貓 猫", "dog": "狗 犬", "horse": "馬 马",
    "sheep": "綿羊 绵羊 羊", "cow": "牛 奶牛", "elephant": "大象 象",
    "bear": "熊", "zebra": "斑馬 斑马", "giraffe": "長頸鹿 长颈鹿",
    "backpack": "背包 後背包 后背包", "umbrella": "傘 伞 雨傘 雨伞 遮陽傘 遮阳伞",
    "handbag": "手提包 手袋", "tie": "領帶 领带", "suitcase": "行李箱 拉桿箱 拉杆箱",
    "frisbee": "飛盤 飞盘", "skis": "雙板滑雪板 双板滑雪板 滑雪雙板 滑雪双板",
    "snowboard": "單板滑雪板 单板滑雪板 滑雪單板 滑雪单板",
    "sports ball": "運動球 运动球 棒球 網球 网球 足球 籃球 篮球 排球",
    "kite": "風箏 风筝", "baseball bat": "棒球棒 球棒", "baseball glove": "棒球手套",
    "skateboard": "滑板", "surfboard": "衝浪板 冲浪板",
    "tennis racket": "網球拍 网球拍", "bottle": "瓶子 水瓶 酒瓶 飲料瓶 饮料瓶",
    "wine glass": "酒杯 葡萄酒杯 紅酒杯 红酒杯", "cup": "杯子 馬克杯 马克杯 茶杯 咖啡杯",
    "fork": "叉子 餐叉", "knife": "刀 餐刀 刀子", "spoon": "湯匙 汤匙 勺子 調羹 调羹",
    "bowl": "碗", "banana": "香蕉", "apple": "蘋果 苹果", "sandwich": "三明治",
    "orange": "橘子 橙子 柳橙", "broccoli": "綠花椰菜 绿花椰菜 西蘭花 西兰花 青花菜",
    "carrot": "胡蘿蔔 胡萝卜 紅蘿蔔 红萝卜", "hot dog": "熱狗 热狗",
    "pizza": "披薩 披萨 比薩 比萨", "donut": "甜甜圈", "cake": "蛋糕",
    "chair": "椅子 座椅", "couch": "沙發 沙发", "potted plant": "盆栽 盆栽植物",
    "bed": "床 床鋪 床铺", "dining table": "餐桌", "toilet": "馬桶 马桶 坐便器",
    "tv": "電視 电视 電視機 电视机", "laptop": "筆電 笔电 筆記型電腦 筆記本電腦 笔记本电脑",
    "mouse": "滑鼠 鼠標 鼠标", "remote": "遙控器 遥控器", "keyboard": "鍵盤 键盘",
    "cell phone": "手機 手机 行動電話 移動電話 移动电话",
    "microwave": "微波爐 微波炉", "oven": "烤箱 烤爐 烤炉", "toaster": "烤麵包機 烤面包机 吐司機 吐司机",
    "sink": "洗手台 水槽 洗碗槽", "refrigerator": "冰箱 電冰箱 电冰箱",
    "book": "書 书 書本 书本", "clock": "時鐘 时钟 鐘錶 钟表 掛鐘 挂钟 鬧鐘 闹钟",
    "vase": "花瓶", "scissors": "剪刀", "teddy bear": "泰迪熊 玩具熊",
    "hair drier": "吹風機 吹风机", "toothbrush": "牙刷",
}
EN = {
    "person": ["people", "man", "men", "woman", "women", "child", "children", "boy", "boys", "girl", "girls", "baby", "babies", "skateboarder", "skateboarders", "surfer", "surfers", "skier", "skiers"],
    "motorcycle": ["motorbike", "motorbikes"], "airplane": ["aeroplane", "aeroplanes"],
    "bus": ["buses"], "person_unused": [], "mouse": ["computer mouse"],
    "couch": ["sofa", "sofas"], "tv": ["television", "televisions"],
    "cell phone": ["cell phones", "mobile phone", "mobile phones", "smartphone", "smartphones"],
    "hair drier": ["hair dryer", "hair dryers", "hair driers"],
    "donut": ["doughnut", "doughnuts"], "knife": ["knives"],
    "sports ball": ["baseball", "baseballs", "tennis ball", "tennis balls", "soccer ball", "soccer balls", "basketball", "basketballs", "volleyball", "volleyballs"],
    "skis": ["ski"], "potted plant": ["potted plants"],
}
# These longer phrases must not be counted as their substring category.
BLOCKED = ["knife block", "knife blocks", "apple logo", "apple logos", "mouse pad", "mouse pads", "baseball cap", "baseball caps", "baseball field", "baseball fields", "toy car", "toy cars"]
NEGATION = re.compile(r"\b(no|not|without|absent)\b|沒有|没有|未見|未见|不存在|無法確認|无法确认|可能|也許|也许|perhaps|maybe", re.I)


def english_forms(category):
    if category in ("person", "sheep", "skis", "scissors", "mouse"):
        return [category]
    last = category.split()[-1]
    plural = last[:-1] + "ies" if last.endswith("y") else last + ("es" if last.endswith(("s", "x", "ch", "sh")) else "s")
    return [category, category[:-len(last)] + plural]


def extract_objects(text, categories):
    text = unicodedata.normalize("NFKC", text).lower()
    aliases = []
    for category in categories:
        for alias in set(english_forms(category) + EN.get(category, []) + ZH.get(category, "").split()):
            aliases.append((alias, category))
    matches, review = [], []
    # Split object lists and sentences; negated/uncertain clauses need human review.
    for segment in re.split(r"[,，、;；。\n]+", text):
        if not segment.strip():
            continue
        if NEGATION.search(segment):
            review.append(segment.strip())
            continue
        spans = []
        for phrase in BLOCKED:
            spans.extend((m.start(), m.end()) for m in re.finditer(r"(?<![a-z])"+re.escape(phrase)+r"(?![a-z])", segment))
        candidates = []
        for alias, category in aliases:
            pattern = re.escape(alias)
            if alias.isascii():
                pattern = r"(?<![a-z])" + pattern + r"(?![a-z])"
            for match in re.finditer(pattern, segment):
                candidates.append((match.start(), match.end(), alias, category))
        # Longest match wins, e.g. hot dog before dog and baseball bat before baseball.
        for start, end, alias, category in sorted(candidates, key=lambda m: (-(m[1]-m[0]), m[0], m[3])):
            if any(start < b and end > a for a, b in spans):
                continue
            spans.append((start, end))
            matches.append({"category": category, "matched_text": alias, "segment": segment.strip()})
    return {"predicted": sorted({m["category"] for m in matches}), "matches": matches,
            "review_segments": review}


def parse_predictions(text, categories):
    return extract_objects(text, categories)["predicted"]


def summarize(predictions):
    if not predictions:
        raise ValueError("Empty prediction split")
    tp, fp, fn = (sum(row[k] for row in predictions) for k in ("tp", "fp", "fn"))
    n = len(predictions)
    return {
        "images": n, "TP": tp, "FP": fp, "FN": fn,
        "precision": tp/(tp+fp) if tp+fp else None,
        "recall": tp/(tp+fn) if tp+fn else None,
        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        "hallucination_rate": fp/(tp+fp) if tp+fp else None,
        "images_with_unmatched_predicted_categories_rate": sum(r["fp"] > 0 for r in predictions)/n,
        "empty_prediction_rate": sum(not r["predicted"] for r in predictions)/n,
        "average_latency_sec": sum(r["latency"] for r in predictions)/n if all("latency" in r for r in predictions) else None,
    }
