from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import TextNode, BaseNode, Document, NodeRelationship, RelatedNodeInfo
from typing import List, Optional
import re


def cn_num_to_int(cn: str) -> Optional[int]:
    """中文数字转整数，如 '十一' → 11, '二十三' → 23"""
    digit_map = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9,
    }
    if not cn:
        return None
    # 处理纯个位数："一" → 1
    if cn in digit_map:
        return digit_map[cn]
    result = 0
    current = 0
    for ch in cn:
        if ch in digit_map:
            current = digit_map[ch]
        elif ch == '十':
            if current == 0:
                current = 1  # "十一" = 10+1, 十前无数字视为1
            result += current * 10
            current = 0
        elif ch == '百':
            if current == 0:
                current = 1
            result += current * 100
            current = 0
    result += current
    return result if result > 0 else None


class ArticleSplitter(NodeParser):
    """条文专用切分器"""

    def _build_context_prefix(self, node_metadata: dict, article: dict) -> str:
        """构建上下文前缀：法规名 + 章节 → 拼入文本以提升 embedding 质量"""
        parts = []

        # 法规名：从 file_name 去掉后缀
        file_name = node_metadata.get("file_name", "")
        law_name = re.sub(r'\.(docx?|pdf|txt)$', '', file_name)
        if law_name:
            parts.append(f"【{law_name}】")

        # 章节（chapter 已是完整行如"第三章　保护和管理"）
        chapter = article.get("chapter", "")
        if chapter:
            parts.append(chapter)

        if parts:
            return " ".join(parts) + "\n"
        return ""

    def _extract_articles(self, text: str) -> List[dict]:
        """提取条文及其章节上下文"""
        articles = []

        # 分割成行
        lines = text.split('\n')
        current_chapter = ""
        current_article_no = 0
        current_chapter_no = 0
        current_chapter_title = ""

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # 匹配章节标题
            chapter_match = re.match(r'第([一二三四五六七八九十]+)章\s*(.+)', line)
            if chapter_match:
                current_chapter = line
                current_chapter_no = chapter_match.group(1)
                current_chapter_title = chapter_match.group(2)
                i += 1
                continue

            # 匹配条文开头
            article_match = re.match(r'第([一二三四五六七八九十百]+)条[　\s]*(.+)', line)
            if article_match:
                article_no = article_match.group(1)
                article_text = line

                # 收集该条的所有内容（包括后续的子项）
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    # 遇到新条文或新章节，停止
                    if re.match(r'第[一二三四五六七八九十百]+条', next_line):
                        break
                    if re.match(r'第[一二三四五六七八九十]+章', next_line):
                        break
                    if next_line:
                        article_text += '\n' + next_line
                    i += 1

                articles.append({
                    "article_no": article_no,
                    "article_no_int": cn_num_to_int(article_no),
                    "chapter": current_chapter,
                    "chapter_no": current_chapter_no,
                    "chapter_title": current_chapter_title,
                    "text": article_text
                })
            else:
                i += 1

        return articles

    def _parse_nodes(
        self,
        nodes: List[BaseNode],
        show_progress: bool = False,
        **kwargs
    ) -> List[TextNode]:
        """实现抽象方法，将节点按条文结构切分"""
        result_nodes = []

        for node in nodes:
            articles = self._extract_articles(node.text)

            for article in articles:
                context_text = self._build_context_prefix(node.metadata, article) + article['text']

                text_node = TextNode(
                    text=context_text,
                    metadata={
                        **node.metadata,
                        "article_no": article["article_no"],
                        "article_no_int": article["article_no_int"],
                        "chapter": article["chapter"],
                        "chapter_no": article["chapter_no"],
                        "chapter_title": article["chapter_title"],
                        "chunk_type": "legal_article"
                    },
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(
                            node_id=node.node_id,
                            metadata=node.metadata,
                        )
                    }
                )
                result_nodes.append(text_node)

        return result_nodes
