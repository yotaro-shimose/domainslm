from typing import Literal, TypedDict

type Role = Literal["system", "user", "assistant", "tool"]


class ChatMLTextItem(TypedDict):
    role: Role
    content: str


class ChatMLSample(TypedDict):
    messages: list[ChatMLTextItem]
