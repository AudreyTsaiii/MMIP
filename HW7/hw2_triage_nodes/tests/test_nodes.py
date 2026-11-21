import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pytest  # noqa: E402
from triage.backends import RuleBackend  # noqa: E402
from triage.nodes import NodeA, NodeB, NodeC, NodeD, SafeNode  # noqa: E402
from triage.reward import triage_reward  # noqa: E402
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


def test_reward_orders_correct_above_wrong():
    gold = {"intent": "logistics", "order_id": "TW20261105009", "need_clarification": False}
    assert triage_reward(GOOD, gold) == 1.0 > triage_reward(GOOD.replace("logistics", "invoice"), gold) > triage_reward("oops", gold)


def test_nodes_valid_with_rule_backend():
    from triage.backends import Message
    h = [Message("user", "訂單 TW20261105009 還沒到")]
    for N in (NodeA, NodeB, NodeC, NodeD):
        o = N(RuleBackend()).triage(h)
        assert o.valid_schema and o.result.intent == Intent.LOGISTICS


def test_safe_node_falls_back():
    from triage.backends import Backend, Message

    class Broken(Backend):
        def generate(self, *a, **k):
            from triage.backends import GenResult
            return GenResult("not json")
        def classify(self, *a, **k):
            raise RuntimeError("oom")
    o = SafeNode(NodeA(Broken()), NodeA(RuleBackend())).triage([Message("user", "我想退貨 TW20261105009")])
    assert o.valid_schema and o.path.startswith("fallback")
