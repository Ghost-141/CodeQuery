from concurrent.futures import ThreadPoolExecutor
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    SparseVectorParams,
    SparseVector,
)
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.core.database import SessionLocal, Repo
from backend.core.qdrant_client import qdrant_client
from backend.services.indexer.chunker import chunk_python_file, chunk_text_file
from backend.services.indexer.embedder import embed_texts, embed_sparse_texts
from backend.services.indexer.manager import clone_repo, list_source_files

from backend.core.logger import setup_logger

logger = setup_logger(__name__)

# Single global executor
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="indexer")


def run_indexing_task(repo_id: str, url: str, local_path: str, collection_name: str):
    logger.info(f"Submitting indexing task for repo_id={repo_id}")
    future = _executor.submit(_index_repo, repo_id, url, local_path, collection_name)

    def _on_done(f):
        try:
            f.result()
            logger.info(f"Indexing task completed for repo_id={repo_id}")
        except Exception as e:
            logger.error(f"Indexing task FAILED for repo_id={repo_id}: {e}")
            _update_repo_status(repo_id, "failed", str(e))

    future.add_done_callback(_on_done)
    logger.info(f"Task submitted for repo_id={repo_id}")


def _update_repo_status(repo_id: str, status: str, error_message: str = None):
    db = SessionLocal()
    try:
        repo = db.query(Repo).filter(Repo.id == repo_id).first()
        if repo:
            repo.status = status
            if error_message:
                repo.error_message = error_message
            db.commit()
            logger.debug(f"Repo {repo_id} status updated in DB to {status}")
    except Exception as e:
        logger.debug(f"Failed to update repo status: {e}")
    finally:
        db.close()


def _index_repo(repo_id: str, url: str, local_path: str, collection_name: str):
    logger.debug(f"Worker started for repo_id={repo_id}")

    try:
        _update_repo_status(repo_id, "cloning")
        logger.debug(f"Calling clone_repo for {url}")
        clone_repo(url, local_path)
        logger.debug(f"clone_repo returned for {repo_id}")

        _update_repo_status(repo_id, "parsing")
        source_files = list(list_source_files(local_path))
        logger.info(f"Found {len(source_files)} source files to parse")

        all_chunks = []
        for idx, sf in enumerate(source_files):
            try:
                logger.info(f"Parsing {idx+1}/{len(source_files)}: {sf}")
                if sf.endswith(".py"):
                    chunks = chunk_python_file(sf, local_path)
                else:
                    chunks = chunk_text_file(sf, local_path)
                all_chunks.extend(chunks)
                logger.info(f"  → {len(chunks)} chunks from {sf}")
            except Exception as e:
                logger.error(f"Failed to chunk {sf}: {e}")
                continue

        if not all_chunks:
            _update_repo_status(repo_id, "failed", "No source files found or parsed.")
            return

        _update_repo_status(repo_id, "embedding")
        logger.debug(f"Generating embeddings for {len(all_chunks)} chunks")

        # Prepare Qdrant collection
        try:
            qdrant_client.get_collection(collection_name)
        except UnexpectedResponse as e:
            if e.status_code == 404:
                logger.debug(
                    f"Collection {collection_name} not found, creating new collection"
                )
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(size=384, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={"sparse": SparseVectorParams()},
                )
            else:
                raise

        batch_size = 32  # Larger batch for faster embedding
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [c.content for c in batch]

            # Parallelize dense + sparse embedding generation
            with ThreadPoolExecutor(max_workers=2) as embed_pool:
                dense_future = embed_pool.submit(embed_texts, texts)
                sparse_future = embed_pool.submit(embed_sparse_texts, texts)
                dense_embeddings = dense_future.result()
                sparse_embeddings = sparse_future.result()

            points = []
            for j, chunk in enumerate(batch):
                points.append(
                    PointStruct(
                        id=i + j,
                        vector={
                            "dense": dense_embeddings[j],
                            "sparse": SparseVector(
                                indices=sparse_embeddings[j]["indices"],
                                values=sparse_embeddings[j]["values"],
                            ),
                        },
                        payload={
                            "file_path": chunk.file_path,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "node_type": chunk.node_type,
                            "name": chunk.name,
                            "content": chunk.content,
                            "repo_id": repo_id,
                        },
                    )
                )
            qdrant_client.upsert(collection_name=collection_name, points=points)
            logger.debug(f"Indexed {i + len(batch)}/{len(all_chunks)} chunks")

        _update_repo_status(repo_id, "ready")
        logger.debug(f"Indexing complete for repo_id={repo_id}")

    except Exception as exc:
        logger.error(f"CRITICAL ERROR in worker: {exc}")
        _update_repo_status(repo_id, "failed", str(exc))
