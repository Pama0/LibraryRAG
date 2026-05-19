import logging

from llama_index.core import VectorStoreIndex, get_response_synthesizer
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

class NormalRagForEvalWorkflow(Workflow):
    def __init__(self, index:VectorStoreIndex,llm:LLM):
        super().__init__()
        self.index = index
        self.llm=llm


    @step
    async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
        logging.info("进入简单查询流程")
        # 从 StartEvent 获取参数（run() 传入的参数会变成 StartEvent 的属性）
        question = ev.get("query")
        nodes = self.index.as_retriever(similarity_top_k=2).retrieve(question)
        # filter_nodes=SimilarityPostprocessor(similarity_cutoff=0.7)._postprocess_nodes(nodes)
        response_synthesizer = get_response_synthesizer()
        result=response_synthesizer.synthesize(query=question,nodes=nodes)
        print(result)
        return StopEvent(result=result)
