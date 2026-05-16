import json
from typing import Type, Optional
from pydantic.v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage

from backend.core.config import settings
from backend.core.llm import get_llm

class SummarizeModuleInput(BaseModel):
    module_name: str = Field(description="Name of the module or class to summarize.")

class SummarizeModuleTool(BaseTool):
    name: str = "summarize_module"
    description: str = "Generate a plain-English summary of a module or class by name."
    args_schema: Type[BaseModel] = SummarizeModuleInput
    
    # Internal state injected at instantiation
    collection_name: str

    def _run(self, module_name: str) -> str:
        if not self.collection_name:
            return "Error: collection_name is required"
        
        # Import search_code logic directly to avoid circular dependency
        from backend.agent.tools.search_code import search_code
        
        # Call the search_code logic
        result = search_code(
            query=module_name, 
            collection_name=self.collection_name, 
            top_k=5
        )
        
        try:
            chunks = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return f"No information found for module or class '{module_name}'."

        if not chunks or not isinstance(chunks, list):
            return f"No information found for module or class '{module_name}'."

        context = "\n\n".join(
            f"File: {c.get('file_path', 'unknown')} (lines {c.get('start_line')}-{c.get('end_line')})\n{c.get('content', '')}"
            for c in chunks
        )

        llm = get_llm(temperature=0.2)
        prompt = (
            f"You are a senior software engineer. Summarize the following code module/class "
            f"named '{module_name}' in plain English. Explain its purpose, key functions, and "
            f"how it fits into the codebase.\n\n{context}"
        )
        response = llm.invoke([SystemMessage(content=prompt)])
        return str(response.content)
