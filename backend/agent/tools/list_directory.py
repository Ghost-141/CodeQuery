import os
import json
from typing import Type
from pydantic.v1 import BaseModel, Field
from langchain_core.tools import BaseTool

class ListDirectoryInput(BaseModel):
    path: str = Field(default=".", description="Relative path from repo root to list.")

class ListDirectoryTool(BaseTool):
    name: str = "list_directory"
    description: str = "List files and subdirectories at the given path to understand repo structure."
    args_schema: Type[BaseModel] = ListDirectoryInput
    
    # Internal state injected at instantiation
    repo_local_path: str

    def _run(self, path: str = ".") -> str:
        if not self.repo_local_path:
            return "Error: repo_local_path is required"
        
        full_path = os.path.join(self.repo_local_path, path)
        if not os.path.abspath(full_path).startswith(os.path.abspath(self.repo_local_path)):
            return "Error: Invalid path."
            
        try:
            entries = []
            for entry in sorted(os.listdir(full_path)):
                entry_path = os.path.join(full_path, entry)
                entries.append({
                    "name": entry,
                    "path": os.path.relpath(entry_path, self.repo_local_path),
                    "type": "directory" if os.path.isdir(entry_path) else "file",
                })
            return json.dumps(entries, indent=2)
        except Exception as exc:
            return f"Error: {exc}"
