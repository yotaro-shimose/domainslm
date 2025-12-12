from pathlib import Path
from typing import Dict, List, Optional

from agents import RunContextWrapper, function_tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from domainslm.openai_util.agent import AgentWrapper

# ---------------------------------------------------------
# 1. Pydanticモデルの定義
# ---------------------------------------------------------


class ParentTitleItem(BaseModel):
    """
    parent_titles リストの中身の定義。
    エラーログによると辞書形式 {'id_sp': '...', 'text': '...', 'level': '...'} になっています。
    """

    id_sp: str
    text: str
    level: str  # 例: "h2", "h3"


class Span(BaseModel):
    """
    個々のテキストスパン。
    """

    id_sp: str
    tag: str
    start_sp: int
    end_sp: int
    text_sp: str
    title: str

    # 【修正箇所】 単なる List[str] ではなく、オブジェクトのリストとして定義
    parent_titles: List[ParentTitleItem] = Field(default_factory=list)

    # セクション情報（nullの場合もあるため Optional に設定）
    id_sec: Optional[str] = None
    start_sec: Optional[int] = None
    end_sec: Optional[int] = None
    text_sec: Optional[str] = None


class Document(BaseModel):
    """
    1つのドキュメントファイル。
    """

    doc_id: str
    title: str
    domain: str
    doc_text: str
    spans: Dict[str, Span]

    @property
    def sorted_spans(self) -> List[Span]:
        """
        start_sp (開始位置) 順にソートしてリストで返すヘルパー
        """
        return sorted(self.spans.values(), key=lambda s: s.start_sp)


class MultiDocDataset(BaseModel):
    """
    データセット全体のルート構造。
    """

    doc_data: Dict[str, Dict[str, Document]]

    def _get_doc_id_mapping(self, category: str) -> Dict[int, str]:
        """
        Create a mapping from integer indices to string doc_ids for a category.
        """
        if category not in self.doc_data:
            return {}
        return {
            idx: doc_id for idx, doc_id in enumerate(self.doc_data[category].keys())
        }

    def _get_reverse_doc_id_mapping(self, category: str) -> Dict[str, int]:
        """
        Create a mapping from string doc_ids to integer indices for a category.
        """
        if category not in self.doc_data:
            return {}
        return {
            doc_id: idx for idx, doc_id in enumerate(self.doc_data[category].keys())
        }

    def list_categories(self) -> List[str]:
        """
        Tool 1: List all available categories (Domains).
        Think of these as the 'Root Folders'.
        """
        # doc_data keys are the domains (e.g., 'ssa', 'dmv', 'va')
        return list(self.doc_data.keys())

    def list_titles_in_category(self, category: str) -> List[Dict]:
        """
        Tool 2: List all document titles within a specific category.
        Returns a list of dictionaries containing 'title' and 'doc_id' (as integer index).
        """
        if category not in self.doc_data:
            return [{"error": f"Category '{category}' not found."}]

        documents = self.doc_data[category]
        results = []

        # We return both ID (as integer index) and Title.
        # The Agent reads the Title, but uses the integer ID to call the next tool.
        for idx, (doc_id, doc_obj) in enumerate(documents.items()):
            results.append({"title": doc_obj.title, "doc_id": idx})

        return results

    def read_document(self, category: str, doc_id: int) -> str:
        """
        Tool 3: Read the content of a specific document using integer index.
        Instead of returning a flat blob of text, this reconstructs the
        document with Markdown headers based on the 'spans' tags.
        """
        if category not in self.doc_data:
            return f"Error: Category '{category}' not found."

        # Map integer index to string doc_id
        id_mapping = self._get_doc_id_mapping(category)
        if doc_id not in id_mapping:
            return (
                f"Error: Document index '{doc_id}' not found in category '{category}'."
            )

        string_doc_id = id_mapping[doc_id]
        docs_in_domain = self.doc_data[category]
        document = docs_in_domain[string_doc_id]

        # Construct a readable Markdown-like string from sorted spans
        formatted_text = []
        formatted_text.append(f"# DOCUMENT: {document.title}\n")

        for span in document.sorted_spans:
            text = span.text_sp.strip()
            if not text:
                continue

            # Add visual hierarchy based on tags
            if span.tag.startswith("h"):
                # Add newline before headers for readability
                formatted_text.append(f"\n## {text}")
            elif span.tag == "li":
                formatted_text.append(f"- {text}")
            else:
                formatted_text.append(text)

        return "\n".join(formatted_text)


@function_tool
def list_categories(wrapper: RunContextWrapper[MultiDocDataset]) -> List[str]:
    """
    List all available categories (Domains).
    Think of these as the 'Root Folders'.
    """
    dataset = wrapper.context
    return dataset.list_categories()


@function_tool
def list_titles_in_category(
    wrapper: RunContextWrapper[MultiDocDataset], category: str
) -> List[Dict]:
    """
    List all document titles within a specific category.
    Returns a list of dictionaries containing 'title' and 'doc_id' (as integer).
    """
    dataset = wrapper.context
    return dataset.list_titles_in_category(category)


@function_tool
def read_document(
    wrapper: RunContextWrapper[MultiDocDataset], category: str, doc_id: int
) -> str:
    """
    Read the content of a specific document.
    Instead of returning a flat blob of text, this reconstructs the
    document with Markdown headers based on the 'spans' tags.
    """
    dataset = wrapper.context
    return dataset.read_document(category, doc_id)


# ---------------------------------------------------------
# 2. 実行・検証コード
# ---------------------------------------------------------

# あなたが提示したJSONデータ（サンプル）
data_root = Path("multidoc2dial/data/multidoc2dial/multidoc2dial_doc.json")
dataset = MultiDocDataset.model_validate_json(data_root.read_text())


agent = AgentWrapper[str].create(
    name="MultiDoc2Dial Document Reader",
    instructions="""You are an intelligent agent designed to assist users to talk about documents from the MultiDoc2Dial dataset.
Your task is to help users navigate through a collection of documents by listing categories, titles, and reading document contents based on their requests.
""",
    model="gpt-5-mini",
    tools=[list_categories, list_titles_in_category, read_document],
    output_type=str,
)


async def main():
    load_dotenv()
    context = dataset
    # user_input = "I need information about vehicle registration from the DMV documents."
    user_input = "What can I do if I forgot to update my address?"
    response = await agent.run(input=user_input, context=context)
    print("Agent Response:")
    print(response.final_output())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
