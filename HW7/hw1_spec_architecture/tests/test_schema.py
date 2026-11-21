import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pytest  # noqa: E402
from triage.schema import Intent, constrained_schema, parse_strict  # noqa: E402
GOOD = '{"intent":"logistics","urgency":"low","slots":{"order_id":"TW20261105009","product":null,"issue_summary":null,"image_findings":null},"need_clarification":false,"confidence":0.9}'


def test_schema_roundtrip_and_derived_fields():
    r = parse_strict(GOOD)
    assert r.department == "物流客服組" and r.missing_slots == []
    r2 = parse_strict(GOOD.replace("TW20261105009", "bad-id").replace('"logistics"', '"invoice"'))
    assert r2.slots.order_id is None and r2.missing_slots == ["order_id"]   # 非法訂單號 → None,而非讓整份輸出失效


def test_strict_parse_rejects_fenced_json():
    with pytest.raises(Exception):
        parse_strict("```json\n" + GOOD + "\n```")


def test_constrained_schema_enumerates_intents():
    assert set(constrained_schema()["properties"]["intent"]["enum"]) == {i.value for i in Intent}
