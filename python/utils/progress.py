"""
Progress bar for tracking training iterations.

Usage:
    progress = ProgressBar(total=100, desc="Training")
    for i in range(100):
        # ... do work ...
        progress.update(1)
    progress.close()
"""

import time
import sys


class ProgressBar:
    """
    A simple progress bar for tracking iterations.

    Displays:
    Training: [======>           ] 45% | 45/100 | 1.2s/it | ETA: 1.1s

    Features:
    - Visual bar (20 characters)
    - Percentage complete
    - Current/total counter
    - Time per iteration
    - Estimated time remaining
    """

    def __init__(self, total, desc="Progress", bar_length=20):
        """
        Initialize progress bar.

        Args:
            total: Total number of iterations
            desc: Description text to display
            bar_length: Width of the visual bar in characters
        """
        self.total = total
        self.desc = desc
        self.bar_length = bar_length
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.update_interval = 0.1  # Only update every 0.1 seconds

    def update(self, n=1):
        """
        Update progress by n steps.

        Args:
            n: Number of steps to increment (default 1)
        """
        self.current += n
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self._render()
        pass

    def _render(self):
        """
        Render and print the progress bar.

        Format:
        desc: [=====>     ] 45% | 45/100 | 1.2s/it | ETA: 1.1s
        """
        # 1. Calculate percentage: self.current / self.total
        percentage = self.current / self.total
        # 2. Build the visual bar:
        filled_length = int(self.bar_length * percentage)
        bar = "=" * (filled_length - 1) + ">" + " " * (self.bar_length - filled_length)
        # 3. Calculate timing:
        elapsed_time = time.time()- self.start_time
        time_per_iter = elapsed_time / self.current
        eta = time_per_iter * (self.total - self.current)
        # 4. Format the string:
        progress_bar = f"\r {self.desc} : [{bar}] {percentage:.0%}| {self.current}/{self.total} | {time_per_iter:.2f}s/it | ETA: {eta:.1f}s"
        sys.stdout.write(progress_bar)
        sys.stdout.flush()

    def close(self):
        """
        Finish the progress bar.

        Print final state and move to new line.
        """
        # 1. Set self.current = self.total
        self.current = self.total
        # 2. Call self._render()
        self._render()
        # 3. Print a newline: print()
        print()

    def set_description(self, desc):
        """
        Update the description text.

        Useful for displaying current loss/accuracy.

        Args:
            desc: New description string
        """
        self.desc = desc


class ProgressBarContext:
    """
    Context manager version for use with 'with' statement.

    Usage:
        with ProgressBarContext(100, "Training") as pbar:
            for i in range(100):
                # ... do work ...
                pbar.update(1)
    """

    def __init__(self, total, desc="Progress"):
        self.pbar = ProgressBar(total, desc)

    def __enter__(self):
        return self.pbar

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.close()
        return False


# Helper function for simple loops
def progress_bar(iterable, desc="Progress"):
    """
    Wrap an iterable with a progress bar.

    Usage:
        for item in progress_bar(my_list, "Processing"):
            # ... do work with item ...

    Args:
        iterable: Any iterable (list, range, etc.)
        desc: Description text

    Yields:
        Items from the iterable
    """
    # 1. Get total length: len(iterable)
    length = len(iterable)
    # 2. Create ProgressBar
    pb = ProgressBar(length, desc)
    # 3. Loop through iterable:
    for item in iterable:
        yield item
        pb.update(1)
    pb.close()
