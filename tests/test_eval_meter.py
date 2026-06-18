"""eval/harness/meter.py：按行测 SUT token 消耗（RunMeter + attach_token_meter）。"""
from eval.harness.meter import RunMeter, attach_token_meter


class _FakeHandler:
    """假 TokenCountingHandler：记录 reset 调用，暴露三类计数属性。"""
    def __init__(self, prompt=0, completion=0, total=0):
        self.prompt_llm_token_count = prompt
        self.completion_llm_token_count = completion
        self.total_llm_token_count = total
        self.reset_called = 0

    def reset_counts(self):
        self.reset_called += 1


class _FakeLLM:
    pass


def test_read_returns_three_token_keys_from_handler():
    meter = RunMeter(_FakeHandler(prompt=120, completion=30, total=150))
    assert meter.read() == {
        "prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
    }


def test_read_without_handler_returns_zeros():
    assert RunMeter(None).read() == {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    }


def test_reset_delegates_to_handler():
    h = _FakeHandler()
    meter = RunMeter(h)
    meter.reset()
    assert h.reset_called == 1


def test_reset_without_handler_is_noop():
    RunMeter(None).reset()   # 不抛即可


def test_attach_token_meter_sets_callback_manager_and_reads_zero():
    llm = _FakeLLM()
    meter = attach_token_meter(llm)
    assert getattr(llm, "callback_manager", None) is not None   # 挂到了 SUT llm 实例
    assert meter.read()["total_tokens"] == 0                    # 真 handler 初值 0
