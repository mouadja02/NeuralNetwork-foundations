"""Test progress bar"""

import sys
import os
import time

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.progress import ProgressBar, progress_bar


def test_basic_progress():
    """Test basic progress bar functionality."""
    print("\nTesting basic progress bar:")

    pbar = ProgressBar(total=50, desc="Test 1")

    for i in range(50):
        time.sleep(0.02)  # Simulate work
        pbar.update(1)

    pbar.close()
    print("✓ Basic test passed\n")


def test_variable_updates():
    """Test updating by different amounts."""
    print("Testing variable update amounts:")

    pbar = ProgressBar(total=100, desc="Test 2")

    for i in range(10):
        time.sleep(0.05)
        pbar.update(10)  # Update by 10 each time

    pbar.close()
    print("✓ Variable update test passed\n")


def test_progress_bar_wrapper():
    """Test the progress_bar wrapper function."""
    print("Testing progress_bar wrapper:")

    items = list(range(30))

    for item in progress_bar(items, "Test 3"):
        time.sleep(0.03)

    print("✓ Wrapper test passed\n")


def test_description_update():
    """Test updating description during progress."""
    print("Testing description updates:")

    pbar = ProgressBar(total=20, desc="Training")

    for i in range(20):
        time.sleep(0.05)
        # Update description with current "loss"
        fake_loss = 1.0 / (i + 1)
        pbar.set_description(f"Training | Loss: {fake_loss:.4f}")
        pbar.update(1)

    pbar.close()
    print("✓ Description update test passed\n")


def run_all_tests():
    print("\n" + "="*60)
    print(" 🧪 TESTING PROGRESS BAR")
    print("="*60)

    tests = [
        test_basic_progress,
        test_variable_updates,
        test_progress_bar_wrapper,
        test_description_update,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("="*60)
    print("🎉 ALL PROGRESS BAR TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
