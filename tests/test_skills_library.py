"""Unit tests for Skills Library module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agentb.core.config import Config
from agentb.core.types import ActionType, Plan, PlanStep, SemanticRole
from agentb.skills_library.skills_library import SkillsLibrary


@pytest.fixture
def sample_plan() -> Plan:
    """Create a sample Plan for testing.

    Returns:
        Plan with two steps
    """
    return Plan(
        task="Search for Python tutorials",
        steps=[
            PlanStep(
                step=1,
                action=ActionType.NAVIGATE,
                target_description="Google homepage",
                value="https://google.com",
                semantic_role=SemanticRole.NAVIGATION,
                required_state="Browser open",
            ),
            PlanStep(
                step=2,
                action=ActionType.TYPE,
                target_description="Search input field",
                value="Python tutorials",
                semantic_role=SemanticRole.FORM_FIELD,
                required_state="Google homepage loaded",
            ),
        ],
    )


@pytest.fixture
def another_plan() -> Plan:
    """Create a different sample Plan for testing.

    Returns:
        Plan with one step
    """
    return Plan(
        task="Click submit button",
        steps=[
            PlanStep(
                step=1,
                action=ActionType.CLICK,
                target_description="Submit button",
                value=None,
                semantic_role=SemanticRole.PRIMARY_ACTION,
                required_state="Form filled",
            ),
        ],
    )


@pytest.fixture
def mock_encoder():
    """Create a mock SentenceTransformer encoder.

    Returns:
        MagicMock configured to simulate SentenceTransformer
    """
    encoder = MagicMock()

    # Mock encode to return different embeddings for different tasks
    def encode_side_effect(text):
        # Simple mock: return different embeddings based on text
        if "Python" in text or "tutorials" in text:
            return MagicMock(tolist=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
        elif "Java" in text:
            return MagicMock(tolist=lambda: [0.9, 0.8, 0.7, 0.6, 0.5])
        elif "Click" in text or "submit" in text:
            return MagicMock(tolist=lambda: [0.5, 0.5, 0.5, 0.5, 0.5])
        else:
            return MagicMock(tolist=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    encoder.encode = MagicMock(side_effect=encode_side_effect)
    return encoder


@pytest.fixture
def mock_collection():
    """Create a mock ChromaDB collection.

    Returns:
        MagicMock configured to simulate ChromaDB collection
    """
    collection = MagicMock()

    # Use simple return_value for count instead of side_effect
    collection.count.return_value = 0
    collection.add = MagicMock()
    collection.get = MagicMock(return_value={"ids": [], "metadatas": []})
    collection.delete = MagicMock()
    collection.query = MagicMock()

    return collection


class TestSkillsLibrary:
    """Test suite for SkillsLibrary class."""

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_init_creates_collection(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
    ) -> None:
        """Test that __init__ creates ChromaDB collection."""
        mock_sentence_transformer.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        # Initialize library
        library = SkillsLibrary(test_config)

        # Verify SentenceTransformer initialized
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")

        # Verify ChromaDB client created
        mock_chroma_client.assert_called_once_with(
            path=str(test_config.chroma_persist_dir)
        )

        # Verify collection created
        mock_client_instance.get_or_create_collection.assert_called_once_with(
            name="skills",
            metadata={"hnsw:space": "cosine"},
        )

        assert library.config == test_config
        assert library._encoder == mock_encoder
        assert library._collection == mock_collection

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_add_skill_stores_plan(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
        sample_plan: Plan,
    ) -> None:
        """Test that add_skill() stores plan with embedding."""
        mock_sentence_transformer.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        library = SkillsLibrary(test_config)

        # Add skill
        task = "Search for Python tutorials"
        skill_id = library.add_skill(task, sample_plan)

        # Verify skill ID returned
        assert skill_id is not None
        assert isinstance(skill_id, str)

        # Verify encoder was called
        mock_encoder.encode.assert_called_with(task)

        # Verify collection.add was called
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]

        # Verify arguments
        assert len(call_args["ids"]) == 1
        assert call_args["ids"][0] == skill_id
        assert call_args["embeddings"][0] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert call_args["metadatas"][0]["task"] == task

        # Verify plan was serialized
        stored_plan_json = call_args["documents"][0]
        stored_plan = Plan.model_validate_json(stored_plan_json)
        assert stored_plan.task == sample_plan.task
        assert len(stored_plan.steps) == len(sample_plan.steps)

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_find_skill_returns_matching_plan(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
        sample_plan: Plan,
    ) -> None:
        """Test that find_skill() returns matching plan above threshold."""
        mock_sentence_transformer.return_value = mock_encoder

        # Set collection count to non-zero BEFORE creating library
        mock_collection.count.return_value = 1

        # Mock query result with high similarity (low distance)
        plan_json = sample_plan.model_dump_json()
        mock_collection.query.return_value = {
            "documents": [[plan_json]],
            "distances": [[0.03]],  # 0.97 similarity (above 0.95 threshold)
            "metadatas": [[{"task": "Search for Python tutorials"}]],
        }

        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        library = SkillsLibrary(test_config)

        # Find skill
        task = "Search for Python tutorials"
        found_plan = library.find_skill(task)

        # Verify plan found
        assert found_plan is not None
        assert found_plan.task == sample_plan.task
        assert len(found_plan.steps) == len(sample_plan.steps)

        # Verify encoder was called
        mock_encoder.encode.assert_called_with(task)

        # Verify query was called
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],
            n_results=1,
            include=["documents", "distances", "metadatas"],
        )

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_find_skill_returns_none_below_threshold(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
        sample_plan: Plan,
    ) -> None:
        """Test that find_skill() returns None when similarity below threshold."""
        mock_sentence_transformer.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_collection.count.return_value = 1

        # Mock query result with low similarity (high distance)
        plan_json = sample_plan.model_dump_json()
        mock_collection.query.return_value = {
            "documents": [[plan_json]],
            "distances": [[0.50]],  # 0.50 similarity (below 0.95 threshold)
            "metadatas": [[{"task": "Search for Python tutorials"}]],
        }

        library = SkillsLibrary(test_config)

        # Find skill
        task = "Search for Java tutorials"
        found_plan = library.find_skill(task)

        # Verify None returned
        assert found_plan is None

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_find_skill_returns_none_when_empty(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
    ) -> None:
        """Test that find_skill() returns None when library is empty."""
        mock_sentence_transformer.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        # Empty collection
        mock_collection.count.return_value = 0

        library = SkillsLibrary(test_config)

        # Find skill
        found_plan = library.find_skill("Any task")

        # Verify None returned
        assert found_plan is None

        # Verify query not called (early return)
        mock_collection.query.assert_not_called()

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_find_skill_handles_empty_results(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
    ) -> None:
        """Test that find_skill() handles empty query results."""
        mock_sentence_transformer.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_collection.count.return_value = 1

        # Mock empty query result
        mock_collection.query.return_value = {
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        library = SkillsLibrary(test_config)

        # Find skill
        found_plan = library.find_skill("Task")

        # Verify None returned
        assert found_plan is None

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_list_skills_returns_all_skills(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
    ) -> None:
        """Test that list_skills() returns all skills."""
        mock_sentence_transformer.return_value = mock_encoder

        # Mock collection with two skills BEFORE creating library
        mock_collection.count.return_value = 2
        mock_collection.get.return_value = {
            "ids": ["skill-1", "skill-2"],
            "metadatas": [
                {"task": "Search for Python"},
                {"task": "Click submit button"},
            ],
        }

        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        library = SkillsLibrary(test_config)

        # List skills
        skills = library.list_skills()

        # Verify skills returned
        assert len(skills) == 2
        assert skills[0] == {"id": "skill-1", "task": "Search for Python"}
        assert skills[1] == {"id": "skill-2", "task": "Click submit button"}

        # Verify get was called
        mock_collection.get.assert_called_once_with(include=["metadatas"])

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_list_skills_returns_empty_when_no_skills(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
    ) -> None:
        """Test that list_skills() returns empty list when no skills."""
        mock_sentence_transformer.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        mock_collection.count.return_value = 0

        library = SkillsLibrary(test_config)

        # List skills
        skills = library.list_skills()

        # Verify empty list
        assert skills == []

        # Verify get not called (early return)
        mock_collection.get.assert_not_called()

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_delete_skill_removes_skill(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
    ) -> None:
        """Test that delete_skill() removes skill from library."""
        mock_sentence_transformer.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        library = SkillsLibrary(test_config)

        # Delete skill
        result = library.delete_skill("skill-123")

        # Verify success
        assert result is True

        # Verify delete was called
        mock_collection.delete.assert_called_once_with(ids=["skill-123"])

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_delete_skill_handles_exception(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
    ) -> None:
        """Test that delete_skill() handles exceptions gracefully."""
        mock_sentence_transformer.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        # Mock delete to raise exception
        mock_collection.delete.side_effect = Exception("Delete failed")

        library = SkillsLibrary(test_config)

        # Delete skill
        result = library.delete_skill("skill-123")

        # Verify failure returned
        assert result is False

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_clear_removes_all_skills(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
    ) -> None:
        """Test that clear() removes all skills."""
        mock_sentence_transformer.return_value = mock_encoder
        mock_client_instance = MagicMock()

        # Create new collection mock for after clear
        new_collection = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = new_collection
        mock_chroma_client.return_value = mock_client_instance

        library = SkillsLibrary(test_config)

        # Clear library
        library.clear()

        # Verify collection deleted
        mock_client_instance.delete_collection.assert_called_once_with("skills")

        # Verify new collection created
        assert mock_client_instance.get_or_create_collection.call_count == 2
        assert library._collection == new_collection

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_count_property_returns_skill_count(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
    ) -> None:
        """Test that count property returns skill count."""
        mock_sentence_transformer.return_value = mock_encoder

        # Set count BEFORE creating library
        mock_collection.count.return_value = 5

        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        library = SkillsLibrary(test_config)

        # Get count
        count = library.count

        # Verify count
        assert count == 5
        mock_collection.count.assert_called()

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_get_embedding_returns_vector(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
    ) -> None:
        """Test that get_embedding() returns embedding vector."""
        mock_sentence_transformer.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        library = SkillsLibrary(test_config)

        # Get embedding
        task = "Search for Python tutorials"
        embedding = library.get_embedding(task)

        # Verify embedding returned
        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_encoder.encode.assert_called_with(task)

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_plan_serialization_roundtrip(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
        sample_plan: Plan,
    ) -> None:
        """Test that Plan serialization/deserialization works correctly."""
        mock_sentence_transformer.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        library = SkillsLibrary(test_config)

        # Serialize plan
        plan_json = sample_plan.model_dump_json()

        # Deserialize plan
        restored_plan = Plan.model_validate_json(plan_json)

        # Verify plan matches
        assert restored_plan.task == sample_plan.task
        assert len(restored_plan.steps) == len(sample_plan.steps)

        for i, step in enumerate(restored_plan.steps):
            original_step = sample_plan.steps[i]
            assert step.step == original_step.step
            assert step.action == original_step.action
            assert step.target_description == original_step.target_description
            assert step.value == original_step.value
            assert step.semantic_role == original_step.semantic_role
            assert step.required_state == original_step.required_state

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_init_with_default_config(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        mock_encoder,
        mock_collection,
    ) -> None:
        """Test that SkillsLibrary initializes with default config."""
        mock_sentence_transformer.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        # Initialize without config
        library = SkillsLibrary()

        # Verify default config created
        assert library.config is not None
        assert isinstance(library.config, Config)

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_add_multiple_skills(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
        sample_plan: Plan,
        another_plan: Plan,
    ) -> None:
        """Test adding multiple skills to the library."""
        mock_sentence_transformer.return_value = mock_encoder
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        library = SkillsLibrary(test_config)

        # Add multiple skills
        id1 = library.add_skill("Search for Python", sample_plan)
        id2 = library.add_skill("Click submit", another_plan)

        # Verify different IDs
        assert id1 != id2

        # Verify add called twice
        assert mock_collection.add.call_count == 2

    @patch("agentb.skills_library.skills_library.chromadb.PersistentClient")
    @patch("agentb.skills_library.skills_library.SentenceTransformer")
    def test_find_skill_with_exact_threshold_match(
        self,
        mock_sentence_transformer,
        mock_chroma_client,
        test_config: Config,
        mock_encoder,
        mock_collection,
        sample_plan: Plan,
    ) -> None:
        """Test that find_skill() accepts exact threshold match."""
        mock_sentence_transformer.return_value = mock_encoder

        # Set count and query BEFORE creating library
        mock_collection.count.return_value = 1

        # Mock query result with exact threshold (0.95)
        plan_json = sample_plan.model_dump_json()
        mock_collection.query.return_value = {
            "documents": [[plan_json]],
            "distances": [[0.05]],  # Exactly 0.95 similarity
            "metadatas": [[{"task": "Test task"}]],
        }

        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        library = SkillsLibrary(test_config)

        # Find skill
        found_plan = library.find_skill("Test task")

        # Verify plan found (>= threshold)
        assert found_plan is not None
