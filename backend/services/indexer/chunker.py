import os
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_python as tspython


@dataclass
class CodeChunk:
    file_path: str
    start_line: int
    end_line: int
    node_type: str
    name: str
    content: str
    parent_name: str | None = None
    parent_type: str | None = None
    hierarchy_path: str = ""
    imports: list[str] | None = None


PYTHON_LANGUAGE = Language(tspython.language())


def get_node_text(source_bytes: bytes, node) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode(
        "utf-8", errors="replace"
    )


def _process_node(
    node,
    source_bytes: bytes,
    file_path: str,
    module_name: str,
    parent_name: str = None,
    parent_type: str = None,
    module_imports: list[str] = None,
) -> list[CodeChunk]:
    """Recursively process an AST node and extract chunks with hierarchy metadata."""
    chunks: list[CodeChunk] = []
    node_name = _extract_name(source_bytes, node)

    if node.type == "class_definition":
        # Build hierarchy: module.ClassName
        hierarchy = f"{module_name}.{node_name}" if module_name else node_name
        chunks.append(
            CodeChunk(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                node_type="class_definition",
                name=node_name,
                content=get_node_text(source_bytes, node),
                parent_name=parent_name,
                parent_type=parent_type,
                hierarchy_path=hierarchy,
                imports=module_imports,
            )
        )
        # Process children (methods) with this class as parent
        for child in node.children:
            if child.type == "block":
                for item in child.children:
                    if item.type in ("function_definition", "class_definition"):
                        chunks.extend(
                            _process_node(
                                item,
                                source_bytes,
                                file_path,
                                module_name,
                                parent_name=node_name,
                                parent_type="class_definition",
                                module_imports=module_imports,
                            )
                        )

    elif node.type == "function_definition":
        # Build hierarchy: module.ClassName.method_name or module.function_name
        if parent_name:
            hierarchy = (
                f"{module_name}.{parent_name}.{node_name}"
                if module_name
                else f"{parent_name}.{node_name}"
            )
        else:
            hierarchy = f"{module_name}.{node_name}" if module_name else node_name
        chunks.append(
            CodeChunk(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                node_type="function_definition",
                name=node_name,
                content=get_node_text(source_bytes, node),
                parent_name=parent_name,
                parent_type=parent_type,
                hierarchy_path=hierarchy,
                imports=module_imports,
            )
        )

    return chunks


def chunk_python_file(file_path: str, repo_root: str) -> list[CodeChunk]:

    with open(file_path, "rb") as f:
        source_bytes = f.read()

    parser = Parser(PYTHON_LANGUAGE)
    tree = parser.parse(source_bytes)
    root = tree.root_node

    relative_path = os.path.relpath(file_path, repo_root)
    module_name = Path(file_path).stem
    chunks: list[CodeChunk] = []

    module_docstring = None
    module_imports = []

    for child in root.children:
        if child.type == "expression_statement":
            inner = child.children[0] if child.children else None
            if inner and inner.type == "string" and not module_docstring:
                module_docstring = get_node_text(source_bytes, child)
                continue
        if child.type in (
            "import_statement",
            "import_from_statement",
            "future_import_statement",
        ):
            module_imports.append(get_node_text(source_bytes, child))
            continue

        if child.type in ("function_definition", "class_definition"):
            chunks.extend(
                _process_node(
                    child,
                    source_bytes,
                    relative_path,
                    module_name,
                    module_imports=module_imports,
                )
            )

    # If no top-level definitions, store whole file as one chunk
    if not chunks:
        chunks.append(
            CodeChunk(
                file_path=relative_path,
                start_line=1,
                end_line=source_bytes.decode("utf-8", errors="replace").count("\n") + 1,
                node_type="module",
                name=module_name,
                content=source_bytes.decode("utf-8", errors="replace"),
                hierarchy_path=module_name,
                imports=module_imports,
            )
        )
    else:
        # Prepend module context (docstring + imports) to first chunk if available
        header_parts = []
        if module_docstring:
            header_parts.append(module_docstring)
        if module_imports:
            header_parts.append("\n".join(module_imports))
        if header_parts:
            header = "\n\n".join(header_parts) + "\n\n"
            chunks[0].content = header + chunks[0].content
            chunks[0].start_line = 1

    return chunks


def chunk_text_file(
    file_path: str, repo_root: str, chunk_size: int = 1000
) -> list[CodeChunk]:
    """Generic chunker for non-Python text files (README, docs, etc.)."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    relative_path = os.path.relpath(file_path, repo_root)
    chunks: list[CodeChunk] = []

    # Simple fixed-size chunking for now
    lines = content.splitlines()
    for i in range(0, len(lines), 50):  # 50 lines per chunk approx
        chunk_lines = lines[i : i + 50]
        if not chunk_lines:
            continue

        chunks.append(
            CodeChunk(
                file_path=relative_path,
                start_line=i + 1,
                end_line=i + len(chunk_lines),
                node_type="text",
                name=Path(file_path).name,
                content="\n".join(chunk_lines),
            )
        )
    return chunks


def _extract_name(source_bytes: bytes, node) -> str:
    for child in node.children:
        if child.type == "identifier":
            return get_node_text(source_bytes, child)
    return ""
