"""按行测被测系统（SUT）的 LLM token 消耗。

把 LlamaIndex 的 TokenCountingHandler 挂到 **SUT llm 实例**的 callback_manager 上——
只数这一个实例的调用，评测 judge（另一 llm 实例）天然不计入。客户端 tokenizer 从
响应文本计数，流式/非流式都算得到，绕开 DeepSeek 流式不返回 usage 的缺口（见
configs/usage_logging.py 注释）。只统计 LLM token，不含 embedding。
"""


class RunMeter:
    """按行测 token：reset 清零、read 取 {prompt,completion,total}_tokens。

    与 TokenCountingHandler 解耦（handler 可为 None，便于单测注入假 handler）。
    """

    def __init__(self, handler=None):
        self._handler = handler

    def reset(self) -> None:
        if self._handler is not None:
            self._handler.reset_counts()

    def read(self) -> dict:
        h = self._handler
        if h is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {
            "prompt_tokens": h.prompt_llm_token_count,
            "completion_tokens": h.completion_llm_token_count,
            "total_tokens": h.total_llm_token_count,
        }


def attach_token_meter(llm) -> RunMeter:
    """给 SUT llm 挂 TokenCountingHandler（只数这一实例 → SUT-only），返回 RunMeter。"""
    from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

    handler = TokenCountingHandler()
    llm.callback_manager = CallbackManager([handler])
    return RunMeter(handler)
