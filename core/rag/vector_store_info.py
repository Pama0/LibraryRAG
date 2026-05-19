"""向量存储元数据描述，供 AutoRetriever 使用"""

from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo

VECTOR_STORE_INFO = VectorStoreInfo(
    content_info="中国法律条文，包含各类管理条例和法规的具体条文内容",
    metadata_info=[
        MetadataInfo(
            name="article_no",
            type="str",
            description="条文编号（中文数字），如'一'、'十一'、'二十三'。用于精确匹配某一条。",
        ),
        MetadataInfo(
            name="article_no_int",
            type="int",
            description="条文编号（阿拉伯数字），如1、11、23。用于范围查询，如'第三条到第十条'应转为 article_no_int>=3 AND article_no_int<=10。",
        ),
        MetadataInfo(
            name="chapter_no",
            type="str",
            description="章节编号（中文数字），如'一'、'二'、'三'。用于筛选某一章。",
        ),
        MetadataInfo(
            name="chapter_title",
            type="str",
            description="章节标题，如'总则'、'保安服务公司'、'监督检查'、'法律责任'。用于按主题筛选章节。",
        ),
        MetadataInfo(
            name="file_name",
            type="str",
            description="法规文件名，如'保安服务管理条例.docx'。仅当用户查询明确提及法规名称时才设置此字段，严禁根据查询内容猜测文件名。",
        ),
    ],
)
