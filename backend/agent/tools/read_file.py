import os
from typing import Type, Optional
from pydantic.v1 import BaseModel, Field
from langchain_core.tools import BaseTool

class ReadFileInput(BaseModel):
    path: str = Field(description="Relative path from repo root to the file.")
    start_line: Optional[int] = Field(default=None, description="Start line number (1-indexed).")
    end_line: Optional[int] = Field(default=None, description="End line number (1-indexed).")

class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = (
        "Read the raw content of a file by its path relative to the repo root. "
        "Use start_line and end_line for large files."
    )
    args_schema: Type[BaseModel] = ReadFileInput
    
    # Internal state injected at instantiation
    repo_local_path: str

    def _run(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        if not self.repo_local_path:
            return "Error: repo_local_path is required"
        
        full_path = os.path.join(self.repo_local_path, path)
        if not os.path.abspath(full_path).startswith(os.path.abspath(self.repo_local_path)):
            return "Error: Invalid file path."
            
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                if start_line is not None or end_line is not None:
                    lines = f.readlines()
                    start = max(0, (start_line or 1) - 1)
                    end = end_line or len(lines)
                    return "".join(lines[start:end])
                return f.read()
        except Exception as exc:
            return f"Error reading file: {exc}"
