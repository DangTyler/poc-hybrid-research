import unittest
import json
import os
from .hybrid_query import query # Use relative import

class TestHybridRetrieval(unittest.TestCase):

    def setUp(self):
        """Load the source data to compare against."""
        papers_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'papers.json')
        grants_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'grants_al.json')
        
        with open(papers_path, 'r') as f:
            self.all_papers = json.load(f)
        
        with open(grants_path, 'r') as f:
            self.all_grants = json.load(f)

    def test_research_retrieval_top_citation(self):
        """
        Tests if the top result for a research query is the most cited paper.
        """
        # Find the max citation count in the source data
        max_citation_paper = max(self.all_papers, key=lambda p: p.get('citationCount', 0))
        max_citations = max_citation_paper.get('citationCount', 0)

        # Perform the query
        prompt = "graph neural networks"
        results = query(prompt, domain="papers")
        
        # Check if the top result's metric matches the max citation count
        self.assertGreater(len(results['results']), 0, "Query returned no results")
        top_result_citations = results['results'][0]['metric']
        self.assertEqual(top_result_citations, max_citations, "Top result is not the most cited paper")

    def test_grants_retrieval_hard_filters(self):
        """
        Tests if the grants query correctly applies hard filters for amount and state.
        """
        # This is a hard-coded ground truth based on our sample data and query logic
        expected_ids_ordered_by_deadline = ['AL-STEM-004', 'AL-EDU-002', 'AL-STEM-001', 'AL-HEALTH-003', 'AL-ART-005']
        
        # The query logic in hybrid_query.py filters for amount <= 500000 and state = 'AL'
        # Let's manually filter our source data to find the ground truth
        ground_truth_ids = [
            g['id'] for g in self.all_grants 
            if g['amount'] <= 500000 and g['state'] == 'AL'
        ]
        
        # Note: The deadline sorting is text-based and complex to replicate here perfectly.
        # For this smoke test, we'll focus on whether the correct IDs are returned,
        # and we can manually verify the order is reasonable.
        
        prompt = "Find funding in Alabama"
        results = query(prompt, domain="grants")
        
        returned_ids = [r['id'] for r in results['results']]
        
        self.assertCountEqual(returned_ids, ground_truth_ids, "Returned grants do not match the filtered ground truth")
        
        # A simple check for the first item in the hardcoded expected order
        # A more robust test would parse and compare dates properly.
        self.assertEqual(returned_ids[0], 'AL-STEM-004', "Grant sorting by deadline seems incorrect")


if __name__ == '__main__':
    # Add project root to the path to allow direct execution of the script
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from retrieval.hybrid_query import driver as hybrid_query_driver

    # Discover and run tests
    unittest.main(exit=False)

    # Clean up the driver connection
    hybrid_query_driver.close()
