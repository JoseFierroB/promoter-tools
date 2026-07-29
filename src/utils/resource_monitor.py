#!/usr/bin/env python3
"""
Resource and Execution Time Monitor for Promoter Tools.
Provides a fail-safe context manager to measure execution time,
CPU user/system time, and peak RAM usage.
If import or resource acquisition fails, it defaults gracefully without raising errors.
"""

import time
import sys

# Try importing the standard resource module (Unix/Linux only)
try:
    import resource
except ImportError:
    resource = None

class ResourceMonitor:
    """Context manager to measure and report script resource usage.
    Safe against environment and OS differences.
    """
    def __init__(self, task_name: str = "Genomic Extraction Task"):
        self.task_name = task_name
        self.start_time = None
        self.start_ru = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        if resource:
            try:
                self.start_ru = resource.getrusage(resource.RUSAGE_SELF)
            except Exception as e:
                # Log warning on stderr but do not crash the pipeline
                print(f"[Profiler Warning] Could not retrieve initial resource metrics: {e}", file=sys.stderr)
                self.start_ru = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        
        cpu_user = 0.0
        cpu_sys = 0.0
        max_rss_mb = 0.0
        
        # Calculate CPU usage and Peak RAM if Unix resource module was active
        if resource and self.start_ru:
            try:
                end_ru = resource.getrusage(resource.RUSAGE_SELF)
                cpu_user = end_ru.ru_utime - self.start_ru.ru_utime
                cpu_sys = end_ru.ru_stime - self.start_ru.ru_stime
                
                # On Linux, ru_maxrss is reported in Kilobytes (KB)
                max_rss_kb = end_ru.ru_maxrss
                max_rss_mb = max_rss_kb / 1024.0
            except Exception as e:
                print(f"[Profiler Warning] Could not calculate final resource metrics: {e}", file=sys.stderr)
                
        # Write profiling report to standard error to keep stdout/FASTA outputs clean
        print("\n" + "="*50, file=sys.stderr)
        print(f"PROFILING SUMMARY - {self.task_name.upper()}", file=sys.stderr)
        print(f"Wall-Clock Time:      {elapsed_time:.4f} seconds", file=sys.stderr)
        if resource and self.start_ru:
            print(f"User CPU Time:       {cpu_user:.4f} seconds", file=sys.stderr)
            print(f"System CPU Time:     {cpu_sys:.4f} seconds", file=sys.stderr)
            print(f"Peak RAM (Max RSS):  {max_rss_mb:.2f} MB", file=sys.stderr)
        else:
            print("Resource tracking (CPU/RAM) not supported on this platform.", file=sys.stderr)
        print("="*50 + "\n", file=sys.stderr)
        
        # Return False to propagate exceptions (if any) occurring inside the block
        return False
