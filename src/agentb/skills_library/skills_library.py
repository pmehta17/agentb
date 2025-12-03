"""Skills Library - Long-term semantic cache of successful workflows."""

import uuid

import chromadb
import structlog
from sentence_transformers import SentenceTransformer

from agentb.core.config import Config
from agentb.core.types import Plan


logger = structlog.get_logger()


class SkillsLibrary:
    """Vector database for semantic caching of successful workflows."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the skills library.

        Args:
            config: Application configuration
        """
        self.config = config or Config()

        # Initialize embedding model
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize ChromaDB
        persist_dir = str(self.config.chroma_persist_dir)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="skills",
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "skills_library_initialized",
            persist_dir=persist_dir,
            skill_count=self._collection.count(),
        )

    def add_skill(self, task: str, plan: Plan) -> str:
        """Add a new skill to the library.

        Args:
            task: Task description
            plan: Successful execution plan

        Returns:
            ID of the added skill
        """
        # Generate embedding
        embedding = self._encoder.encode(task).tolist()

        # Generate unique ID
        skill_id = str(uuid.uuid4())

        # Serialize plan to JSON
        plan_json = plan.model_dump_json()

        # Add to collection
        self._collection.add(
            ids=[skill_id],
            embeddings=[embedding],
            documents=[plan_json],
            metadatas=[{"task": task}],
        )

        logger.info(
            "skill_added",
            skill_id=skill_id,
            task=task,
            steps=len(plan.steps),
        )

        return skill_id

    def find_skill(self, task: str) -> Plan | None:
        """Find a matching skill for a task.

        Args:
            task: Task description to match

        Returns:
            Matching plan if similarity > threshold, None otherwise
        """
        result = self.find_skill_with_id(task)
        return result[1] if result else None

    def find_skill_with_id(self, task: str) -> tuple[str, Plan] | None:
        """Find a matching skill for a task with its ID.

        Args:
            task: Task description to match

        Returns:
            Tuple of (skill_id, plan) if similarity > threshold, None otherwise
        """
        if self._collection.count() == 0:
            logger.debug("skills_library_empty")
            return None

        # Generate embedding for task
        embedding = self._encoder.encode(task).tolist()

        # Query for nearest neighbor
        # Note: ChromaDB returns IDs by default
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "distances", "metadatas"],
        )

        if not results["documents"] or not results["documents"][0]:
            logger.debug("no_matching_skill", task=task)
            return None

        # ChromaDB returns distances, convert to similarity
        # For cosine distance: similarity = 1 - distance
        distance = results["distances"][0][0]
        similarity = 1 - distance

        if similarity >= self.config.skill_similarity_threshold:
            skill_id = results["ids"][0][0]
            plan_json = results["documents"][0][0]
            plan = Plan.model_validate_json(plan_json)
            original_task = results["metadatas"][0][0]["task"]

            logger.info(
                "skill_found",
                original_task=original_task,
                similarity=similarity,
                steps=len(plan.steps),
            )

            return (skill_id, plan)
        else:
            logger.debug(
                "skill_below_threshold",
                task=task,
                similarity=similarity,
                threshold=self.config.skill_similarity_threshold,
            )
            return None

    def get_embedding(self, task: str) -> list[float]:
        """Generate embedding for a task.

        Args:
            task: Task description

        Returns:
            Embedding vector
        """
        return self._encoder.encode(task).tolist()

    def list_skills(self) -> list[dict]:
        """List all skills in the library.

        Returns:
            List of skill metadata
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.get(include=["metadatas"])
        return [
            {"id": skill_id, "task": meta["task"]}
            for skill_id, meta in zip(results["ids"], results["metadatas"])
        ]

    def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill from the library.

        Args:
            skill_id: ID of the skill to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            self._collection.delete(ids=[skill_id])
            logger.info("skill_deleted", skill_id=skill_id)
            return True
        except Exception as e:
            logger.error("skill_delete_failed", skill_id=skill_id, error=str(e))
            return False

    def clear(self) -> None:
        """Clear all skills from the library."""
        # Delete and recreate collection
        self._client.delete_collection("skills")
        self._collection = self._client.get_or_create_collection(
            name="skills",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("skills_library_cleared")

    @property
    def count(self) -> int:
        """Get the number of skills in the library."""
        return self._collection.count()
