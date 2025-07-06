import torch
import pytest
from utils.retrieval_metrics import (
    compute_mrr,
    compute_map,
    compute_ndcg_at_k,
    compute_median_rank,
    compute_recall_at_k,
)

def test_perfect_retrieval():
    """Test metrics when retrieval is perfect (diagonal similarity matrix)"""
    # Create a perfect similarity matrix (diagonal = 1, rest = 0)
    n_samples = 5
    sim_matrix = torch.zeros(n_samples, n_samples)
    torch.diagonal(sim_matrix)[:] = 1.0
    
    # Ground truth indices are just [0,1,2,3,4]
    gt_indices = torch.arange(n_samples)
    
    # Test recall@k
    recall_metrics = compute_recall_at_k(sim_matrix, gt_indices, k_values=[1, 3, 5])
    assert recall_metrics["Recall@1"] == 1.0, "Perfect retrieval should have Recall@1 = 1"
    assert recall_metrics["Recall@3"] == 1.0, "Perfect retrieval should have Recall@3 = 1"
    assert recall_metrics["Recall@5"] == 1.0, "Perfect retrieval should have Recall@5 = 1"
    
    # Test MRR
    mrr = compute_mrr(sim_matrix, gt_indices)
    assert mrr["MRR_V2T"] == 1.0, "Perfect retrieval should have MRR = 1"
    
    # Test MAP
    map_score = compute_map(sim_matrix, gt_indices)
    assert abs(map_score - 1.0) < 1e-6, "Perfect retrieval should have MAP = 1"
    
    # Test Median Rank
    median_rank = compute_median_rank(sim_matrix, gt_indices)
    assert median_rank == 1, "Perfect retrieval should have median rank = 1"
    
    # Test NDCG
    ndcg = compute_ndcg_at_k(sim_matrix, gt_indices, k_values=[1, 3, 5])
    assert abs(ndcg["NDCG@1_V2T"] - 1.0) < 1e-6, "Perfect retrieval should have NDCG@1 = 1"
    assert abs(ndcg["NDCG@3_V2T"] - 1.0) < 1e-6, "Perfect retrieval should have NDCG@3 = 1"
    assert abs(ndcg["NDCG@5_V2T"] - 1.0) < 1e-6, "Perfect retrieval should have NDCG@5 = 1"

def test_random_retrieval():
    """Test metrics with random similarity scores"""
    n_samples = 100
    torch.manual_seed(42)  # For reproducibility
    
    # Create random similarity matrix
    sim_matrix = torch.randn(n_samples, n_samples)
    gt_indices = torch.arange(n_samples)
    
    # Test recall@k
    recall_metrics = compute_recall_at_k(sim_matrix, gt_indices, k_values=[1, 5, 10])
    assert 0 <= recall_metrics["Recall@1"] <= 1, "Recall should be between 0 and 1"
    assert 0 <= recall_metrics["Recall@5"] <= 1, "Recall should be between 0 and 1"
    assert 0 <= recall_metrics["Recall@10"] <= 1, "Recall should be between 0 and 1"
    
    # Test that recall increases with k
    assert recall_metrics["Recall@1"] <= recall_metrics["Recall@5"] <= recall_metrics["Recall@10"], \
        "Recall should increase with k"
    
    # Test MRR
    mrr = compute_mrr(sim_matrix, gt_indices)
    assert 0 <= mrr["MRR_V2T"] <= 1, "MRR should be between 0 and 1"
    
    # Test MAP
    map_score = compute_map(sim_matrix, gt_indices)
    assert 0 <= map_score <= 1, "MAP should be between 0 and 1"
    
    # Test Median Rank
    median_rank = compute_median_rank(sim_matrix, gt_indices)
    assert 1 <= median_rank <= n_samples, f"Median rank should be between 1 and {n_samples}"
    
    # Test NDCG
    ndcg = compute_ndcg_at_k(sim_matrix, gt_indices, k_values=[1, 5, 10])
    assert 0 <= ndcg["NDCG@1_V2T"] <= 1, "NDCG should be between 0 and 1"
    assert 0 <= ndcg["NDCG@5_V2T"] <= 1, "NDCG should be between 0 and 1"
    assert 0 <= ndcg["NDCG@10_V2T"] <= 1, "NDCG should be between 0 and 1"

def test_worst_retrieval():
    """Test metrics when retrieval is worst possible (reversed diagonal)"""
    n_samples = 5
    sim_matrix = torch.zeros(n_samples, n_samples)
    # Put 1s in the reverse diagonal
    for i in range(n_samples):
        sim_matrix[i, n_samples-1-i] = 1.0
    
    gt_indices = torch.arange(n_samples)
    
    # Test recall@1
    recall_metrics = compute_recall_at_k(sim_matrix, gt_indices, k_values=[1])
    assert recall_metrics["Recall@1"] == pytest.approx(0.2), "Worst retrieval should have Recall@1 = 0.2 (1/5) since each query has a match"
    
    # Test MRR
    mrr = compute_mrr(sim_matrix, gt_indices)
    assert mrr["MRR_V2T"] < 0.5, "Worst retrieval should have low MRR"
    
    # Test Median Rank
    median_rank = compute_median_rank(sim_matrix, gt_indices)
    assert median_rank > n_samples/2, "Worst retrieval should have high median rank"

def test_non_square_similarity():
    """Test metrics with non-square similarity matrix (NxM case)"""
    n_videos = 10
    n_texts = 5
    sim_matrix = torch.zeros(n_videos, n_texts)
    # Make some videos match with their corresponding texts
    for i in range(min(n_videos, n_texts)):
        sim_matrix[i, i] = 1.0
    
    gt_indices = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])  # Some repeated indices
    
    # Test all metrics work with non-square matrix
    recall_metrics = compute_recall_at_k(sim_matrix, gt_indices, k_values=[1, 3])
    mrr = compute_mrr(sim_matrix, gt_indices)
    map_score = compute_map(sim_matrix, gt_indices)
    median_rank = compute_median_rank(sim_matrix, gt_indices)
    ndcg = compute_ndcg_at_k(sim_matrix, gt_indices, k_values=[1, 3])
    
    # All metrics should return valid values
    assert isinstance(recall_metrics["Recall@1"], float)
    assert isinstance(mrr["MRR_V2T"], float)
    assert isinstance(map_score, float)
    assert isinstance(median_rank, int)
    assert isinstance(ndcg["NDCG@1_V2T"], float)

def test_edge_cases():
    """Test edge cases and potential error conditions"""
    # Test single sample
    sim_matrix = torch.tensor([[1.0]])
    gt_indices = torch.tensor([0])
    
    recall_metrics = compute_recall_at_k(sim_matrix, gt_indices, k_values=[1])
    assert recall_metrics["Recall@1"] == 1.0
    
    # Test with k larger than number of samples
    n_samples = 3
    sim_matrix = torch.eye(n_samples)
    gt_indices = torch.arange(n_samples)
    
    recall_metrics = compute_recall_at_k(sim_matrix, gt_indices, k_values=[5])
    assert recall_metrics["Recall@5"] == 1.0
    
    ndcg = compute_ndcg_at_k(sim_matrix, gt_indices, k_values=[5])
    assert 0 <= ndcg["NDCG@5_V2T"] <= 1

if __name__ == "__main__":
    # Run all tests
    test_perfect_retrieval()
    test_random_retrieval()
    test_worst_retrieval()
    test_non_square_similarity()
    test_edge_cases()
    print("All tests passed!") 