"""ChromaDB service with clean abstraction."""

from typing import Iterator
from pathlib import Path
import chromadb
from chromadb.api.models.Collection import Collection
from loguru import logger

from core.config import get_settings
from core.exceptions import DatabaseError


class ChromaService:
    """Service for ChromaDB operations."""

    def __init__(
        self, db_path: Path | str | None = None, collection_name: str | None = None
    ):
        """Initialize ChromaDB service.

        Args:
            db_path: Path to ChromaDB persistence directory
            collection_name: Name of the collection to use
        """
        settings = get_settings()

        self.db_path = Path(db_path) if db_path else settings.database.path
        self.collection_name = collection_name or settings.database.collection_name
        self._client: chromadb.Client | None = None
        self._collection: Collection | None = None

        logger.debug(
            f"ChromaService initialized: path={self.db_path}, collection={self.collection_name}"
        )

    def _get_client(self) -> chromadb.Client:
        """Get or create ChromaDB client."""
        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(path=str(self.db_path))
                logger.debug(f"Created ChromaDB client at {self.db_path}")
            except Exception as e:
                raise DatabaseError(f"Failed to create ChromaDB client: {e}")
        return self._client

    def _get_collection(self) -> Collection:
        """Get or create collection."""
        if self._collection is None:
            try:
                client = self._get_client()

                self._collection = client.get_or_create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                )
                logger.debug(f"Got collection: {self.collection_name}")
            except Exception as e:
                raise DatabaseError(f"Failed to get/create collection: {e}")
        return self._collection

    def add_song(
        self, song_id: str, embedding: list[float], metadata: dict | None = None
    ) -> None:
        """Add a song embedding to the database.

        Args:
            song_id: Unique identifier for the song
            embedding: Vector embedding of the song
            metadata: Optional metadata dict

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            collection = self._get_collection()
            collection.add(
                ids=[song_id], embeddings=[embedding], metadatas=[metadata or {}]
            )
            logger.debug(f"Added song: {song_id}")
        except Exception as e:
            raise DatabaseError(f"Failed to add song {song_id}: {e}")

    def search_similar(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """Search for similar songs.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of result dictionaries with id, distance, and metadata

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            collection = self._get_collection()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_dict,
            )

            # Format results
            formatted = []
            for i in range(len(results["ids"][0])):
                formatted.append(
                    {
                        "id": results["ids"][0][i],
                        "distance": results["distances"][0][i],
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                    }
                )

            return formatted

        except Exception as e:
            raise DatabaseError(f"Failed to search: {e}")

    def get_song(self, song_id: str) -> dict | None:
        """Get a specific song by ID.

        Args:
            song_id: Song identifier

        Returns:
            Song data or None if not found
        """
        try:
            collection = self._get_collection()
            result = collection.get(ids=[song_id])

            if not result["ids"]:
                return None

            return {
                "id": result["ids"][0],
                "embedding": result["embeddings"][0] if result["embeddings"] else None,
                "metadata": result["metadatas"][0] if result["metadatas"] else {},
            }

        except Exception as e:
            raise DatabaseError(f"Failed to get song {song_id}: {e}")

    def song_exists(self, song_id: str) -> bool:
        """Check if a song exists in the database.

        Args:
            song_id: Song identifier

        Returns:
            True if exists, False otherwise
        """
        return self.get_song(song_id) is not None

    def update_metadata(self, song_id: str, metadata: dict) -> None:
        """Update metadata for an existing song.

        Args:
            song_id: Song identifier
            metadata: New metadata dict

        Raises:
            DatabaseError: If the operation fails
        """
        try:
            collection = self._get_collection()
            collection.update(ids=[song_id], metadatas=[metadata])
            logger.debug(f"Updated metadata for: {song_id}")
        except Exception as e:
            raise DatabaseError(f"Failed to update metadata for {song_id}: {e}")

    def list_songs(self, limit: int = 100, offset: int = 0) -> Iterator[dict]:
        """List all songs in the database.

        Args:
            limit: Maximum number to return
            offset: Offset for pagination

        Yields:
            Song dictionaries
        """
        try:
            collection = self._get_collection()
            results = collection.get(limit=limit, offset=offset)

            for i in range(len(results["ids"])):
                yield {
                    "id": results["ids"][i],
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                }

        except Exception as e:
            raise DatabaseError(f"Failed to list songs: {e}")

    def delete_song(self, song_id: str) -> None:
        """Delete a song from the database.

        Args:
            song_id: Song identifier to delete
        """
        try:
            collection = self._get_collection()
            collection.delete(ids=[song_id])
            logger.debug(f"Deleted song: {song_id}")
        except Exception as e:
            raise DatabaseError(f"Failed to delete song {song_id}: {e}")

    def count(self) -> int:
        """Get total number of songs in the collection.

        Returns:
            Count of songs
        """
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception as e:
            raise DatabaseError(f"Failed to count songs: {e}")
