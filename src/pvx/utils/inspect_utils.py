from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageAssistant,
)

_ROLE = {
    "system": ChatMessageSystem,
    "user": ChatMessageUser,
    "assistant": ChatMessageAssistant,
}

def dicts_to_chatmessages(msgs: list[dict] | dict) -> list[ChatMessage]:
    '''
    Convert list of dictionary messages into Inspect's ChatMessages
    
    Args:
        msgs (list[dict]): 
    
    Returns:
        list[ChatMessage]: list of Chatmessages
    '''
    if not isinstance(msgs, list):
        msgs = [msgs]
    
    out: list[ChatMessage] = []
    for m in msgs:
        role = m["role"]
        content = m["content"]
        cls = _ROLE[role]

        out.append(cls(role=role, content=content))
        
    return out

def print_sample_input(input: str | list[ChatMessage]) -> str:
    '''
    Args:
    Returns:
    '''
    if isinstance(input, str):
        return input
    elif isinstance(input, list):
        res = ""
        for cm in input:
            res += f'[[{cm.role}]]\n{cm.content}\n\n'
            
        return res