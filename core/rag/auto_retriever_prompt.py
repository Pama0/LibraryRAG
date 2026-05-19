"""中文法律场景 AutoRetriever prompt"""

LEGAL_AUTO_RETRIEVER_PROMPT = """\
你的任务是将用户的查询转换为结构化检索请求。

<< 结构化请求格式 >>
请用以下 JSON 格式回应：

{schema_str}

字段说明：
- query: 用于语义检索的关键词，去除已经体现在 filters 中的结构化条件
- filters: 元数据过滤条件列表，每个条件包含 key、value、operator
- top_k: 返回结果数量（仅在用户明确指定时设置，否则不设置）

operator 可选值：==, !=, >, <, >=, <=, in（严禁使用 contains 等其他操作符，数据库不支持）

<< 重要规则 >>
1. filters 只能引用下方数据源中存在的属性
2. 不要在 query 中重复 filters 已表达的条件
3. 如果没有合适的过滤条件，filters 返回空列表 []
4. 条文编号用 article_no_int（整数）做范围查询，用 article_no（中文数字）做精确匹配
5. 当查询同时涉及多个法规时，必须用 file_name 区分，每个法规的条件用 AND 组合
6. "第X条到第Y条" 这类范围查询，应转为 article_no_int >= X AND article_no_int <= Y
7. 不要根据查询内容猜测 file_name，仅当用户明确提及法规名称时才设置 file_name 过滤条件

<< 示例 1 >>
数据源：
```json
{{
    "content_info": "中国法律条文",
    "metadata_info": [
        {{"name": "article_no", "type": "str", "description": "条文编号（中文数字）"}},
        {{"name": "article_no_int", "type": "int", "description": "条文编号（阿拉伯数字），用于范围查询"}},
        {{"name": "chapter_title", "type": "str", "description": "章节标题"}},
        {{"name": "file_name", "type": "str", "description": "法规文件名"}}
    ]
}}
```

用户查询：第十一条是什么

结构化请求：
```json
{{"query": "", "filters": [{{"key": "article_no", "value": "十一", "operator": "=="}}], "top_k": null}}
```

<< 示例 2 >>
用户查询：保安服务管理条例中关于培训的规定

结构化请求：
```json
{{"query": "培训", "filters": [{{"key": "file_name", "value": "保安服务管理条例.docx", "operator": "=="}}], "top_k": null}}
```

<< 示例 3 >>
用户查询：第三条到第十条的内容

结构化请求：
```json
{{"query": "", "filters": [{{"key": "article_no_int", "value": 3, "operator": ">="}}, {{"key": "article_no_int", "value": 10, "operator": "<="}}], "top_k": null}}
```

<< 示例 4 >>
用户查询：保安服务管理条例第七条和电影管理条例第十一条分别是什么

结构化请求：
```json
{{"query": "", "filters": [{{"key": "article_no", "value": "七", "operator": "=="}}], "top_k": null}}
```

注意：示例4中查询涉及多个法规的多个条文，当前单次检索无法表达 OR 逻辑，请选择用户最可能关心的一个条件进行过滤，其余通过语义检索覆盖。

<< 当前数据源 >>
```json
{info_str}
```

用户查询：
{query_str}

结构化请求：
"""
