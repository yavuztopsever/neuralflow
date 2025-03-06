"""
Unit tests for storage functionality.
"""
import pytest
import os
from src.storage import Storage

@pytest.fixture
def storage():
    """Create a test storage instance."""
    return Storage("test_data")

@pytest.fixture
def cleanup():
    """Cleanup after tests."""
    yield
    if os.path.exists("test_data"):
        os.remove("test_data")

def test_storage_initialization(storage):
    """Test storage initialization."""
    assert storage.data_dir == "test_data"
    assert os.path.exists(storage.data_dir)

def test_storage_save_and_load(storage, cleanup):
    """Test saving and loading data."""
    test_data = {"key": "value"}
    storage.save("test.json", test_data)
    loaded_data = storage.load("test.json")
    assert loaded_data == test_data

def test_storage_file_not_found(storage):
    """Test handling of non-existent files."""
    with pytest.raises(FileNotFoundError):
        storage.load("nonexistent.json")

def test_storage_invalid_json(storage, cleanup):
    """Test handling of invalid JSON data."""
    with open(os.path.join(storage.data_dir, "invalid.json"), "w") as f:
        f.write("invalid json")
    
    with pytest.raises(ValueError):
        storage.load("invalid.json")

def test_storage_list_files(storage, cleanup):
    """Test listing files in storage."""
    test_files = ["test1.json", "test2.json"]
    for file in test_files:
        storage.save(file, {"data": "test"})
    
    files = storage.list_files()
    assert all(file in files for file in test_files)

def test_storage_delete_file(storage, cleanup):
    """Test deleting files from storage."""
    storage.save("test.json", {"data": "test"})
    storage.delete_file("test.json")
    assert not os.path.exists(os.path.join(storage.data_dir, "test.json")) 