import logging

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.llms import LLM
from llama_index.core.schema import QueryBundle
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)
from llama_index.core.indices.vector_store.retrievers.auto_retriever.auto_retriever import (
    VectorIndexAutoRetriever,
)
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from core.rag.vector_store_info import VECTOR_STORE_INFO
from core.rag.auto_retriever_prompt import LEGAL_AUTO_RETRIEVER_PROMPT


class SimpleRagWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, llm: LLM):
        super().__init__()
        self.index = index
        self.llm = llm
        self.auto_retriever = VectorIndexAutoRetriever(
            index=index,
            vector_store_info=VECTOR_STORE_INFO,
            llm=llm,
            prompt_template_str=LEGAL_AUTO_RETRIEVER_PROMPT,
            similarity_top_k=5,
            max_top_k=20,
        )

    @step
    async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
        logging.info("进入简单查询流程（AutoRetriever）")
        question = ev.get("query")
        query_bundle = QueryBundle(query_str=question)

        # 第一步：AutoRetriever 提取过滤条件并检索
        nodes = []
        try:
            spec = await self.auto_retriever.agenerate_retrieval_spec(
                query_bundle=query_bundle
            )
            filter_list = [(f.key, f.operator.value, f.value) for f in spec.filters]
            logging.info(f"AutoRetriever 提取: query='{spec.query}', filters={filter_list}")

            retriever, spec_query_bundle = self.auto_retriever._build_retriever_from_spec(spec)
            nodes = retriever.retrieve(spec_query_bundle)
        except Exception as e:
            logging.warning(f"AutoRetriever 失败: {e}")

        # 第二步：空结果回退（file_name 过滤导致）或兜底纯向量检索
        if not nodes:
            reason = "file_name 过滤导致空结果" if nodes is not None and any(
                f.key == "file_name" for f in getattr(spec, "filters", [])
            ) else "AutoRetriever 失败"
            logging.warning(f"{reason}，退回纯向量检索")
            nodes = self.index.as_retriever(
                similarity_top_k=5
            ).retrieve(query_bundle)

        response_synthesizer = get_response_synthesizer()
        result = response_synthesizer.synthesize(query=question, nodes=nodes)
        print(result)
        return StopEvent(result=result)



