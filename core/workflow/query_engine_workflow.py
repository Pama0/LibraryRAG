import logging

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)

class QueryEngineWorkflow(Workflow):
    def __init__(self, query_engine:BaseQueryEngine):
        super().__init__()
        self.query_engine = query_engine

    @step
    async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
        logging.info("进入简单查询流程")
        # 从 StartEvent 获取参数（run() 传入的参数会变成 StartEvent 的属性）
        question = ev.get("query")
        result=self.query_engine.query(question)
        print(result)
        return StopEvent(result=result)
