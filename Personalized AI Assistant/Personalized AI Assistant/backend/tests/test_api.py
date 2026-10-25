import io
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "documentation" in data


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Personalized AI Assistant"


def test_model_status_endpoint():
    response = client.get("/api/model/status")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "loaded" in data
    assert "device" in data


def test_documents_list_and_stats():
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert "documents" in response.json()

    stats_response = client.get("/api/documents/stats")
    assert stats_response.status_code == 200
    stats = stats_response.json()
    assert "total_documents" in stats
    assert "vector_stats" in stats


def test_upload_invalid_file_extension():
    fake_file = io.BytesIO(b"binary executable code")
    response = client.post(
        "/api/documents/upload",
        files={"file": ("bad_program.exe", fake_file, "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_valid_text_and_delete():
    text_content = b"This is a sample research document about personalized AI assistants."
    fake_file = io.BytesIO(text_content)
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test_upload.txt", fake_file, "text/plain")}
    )
    assert response.status_code == 201
    doc_data = response.json()
    doc_id = doc_data["document_id"]
    assert doc_data["filename"] == "test_upload.txt"
    assert "preview" in doc_data["extracted_text_preview"] or len(doc_data["extracted_text_preview"]) > 0

    # Index document
    index_response = client.post("/api/documents/index")
    assert index_response.status_code == 200
    idx_data = index_response.json()
    assert idx_data["chunk_count"] >= 1

    # Delete document
    del_response = client.delete(f"/api/documents/{doc_id}")
    assert del_response.status_code == 200
    assert del_response.json()["status"] == "deleted"


def test_chat_validation():
    # Empty question should fail validation
    response = client.post("/api/chat", json={"question": "", "mode": "rag"})
    assert response.status_code == 422

    # Invalid mode should fail validation
    response = client.post("/api/chat", json={"question": "Hello", "mode": "unknown_mode"})
    assert response.status_code == 422


def test_evaluation_results_and_training_status():
    eval_resp = client.get("/api/evaluation/results")
    assert eval_resp.status_code == 200
    assert "status" in eval_resp.json()

    train_resp = client.get("/api/training/status")
    assert train_resp.status_code == 200
    assert "status" in train_resp.json()
