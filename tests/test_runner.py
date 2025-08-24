"""
Test Runner for Embedding System

This script provides a convenient way to run all tests for the embedding system.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Add the tests directory to the Python path
tests_path = Path(__file__).parent
sys.path.insert(0, str(tests_path))

# Import all test modules
from test_embedding_system import (
    TestEmbeddingManager,
    TestKnowledgeBaseIntegration,
    TestBatchProcessing,
    TestErrorHandling,
)


def create_test_suite():
    """Create a test suite with all test cases."""
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTest(unittest.makeSuite(TestEmbeddingManager))
    suite.addTest(unittest.makeSuite(TestKnowledgeBaseIntegration))
    suite.addTest(unittest.makeSuite(TestBatchProcessing))
    suite.addTest(unittest.makeSuite(TestErrorHandling))

    return suite


def run_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running Embedding System Tests")
    print("=" * 60)

    # Create test suite
    suite = create_test_suite()

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
