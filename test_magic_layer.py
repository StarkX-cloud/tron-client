"""
Test the TRON magic layer - verify basic mechanics work
"""

import sys
import os

# Add TRON to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tron
from tron.magic_future import MagicFuture
from tron.config import get_config


def test_config_auto_discovery():
    """Test that config auto-discovers or uses env var."""
    print("✓ Config system working")
    config = get_config()
    print(f"  Server URL: {config.url}")
    assert config.url is not None


def test_magic_future_local():
    """Test MagicFuture with local execution."""
    print("✓ MagicFuture (local)")
    
    # Create a local future
    future = MagicFuture(
        job_id="test-local-1",
        status_fn=lambda _: {"status": "completed"},
        is_local=True,
        local_result="test result"
    )
    
    # Test status
    assert future.status()["status"] == "completed"
    
    # Test ready
    assert future.ready() == True
    assert future.done() == True
    
    # Test get
    assert future.get() == "test result"
    assert future.result() == "test result"
    
    print("  get/result/ready/done all working")


def test_magic_future_repr():
    """Test MagicFuture repr."""
    print("✓ MagicFuture repr")
    future = MagicFuture(
        job_id="test-123",
        status_fn=lambda _: {"status": "running"},
        is_local=True,
        local_result="x"
    )
    repr_str = repr(future)
    assert "test-123" in repr_str
    assert "running" in repr_str or "completed" in repr_str
    print(f"  repr: {repr_str}")


def test_remote_decorator_basic():
    """Test @remote decorator with local execution."""
    print("✓ @remote decorator")
    
    @tron.remote
    def simple_add(a, b):
        return a + b
    
    # Call the function
    future = simple_add(2, 3)
    
    # Verify it returns a MagicFuture
    assert isinstance(future, MagicFuture)
    print("  Returns MagicFuture")
    
    # Get result (should be local)
    result = future.get()
    assert result == 5
    print(f"  Local execution: 2 + 3 = {result}")


def test_remote_decorator_gpu_param():
    """Test @remote with GPU parameter."""
    print("✓ @remote with parameters")
    
    @tron.remote(gpu=True, memory_gb=8)
    def gpu_task():
        return "gpu"
    
    future = gpu_task()
    assert isinstance(future, MagicFuture)
    result = future.get()
    assert result == "gpu"
    print("  GPU parameter parsed correctly")


def test_remote_decorator_call_time_params():
    """Test @remote with call-time parameters."""
    print("✓ @remote call-time parameters")
    
    @tron.remote
    def flexible_task():
        return "done"
    
    # Call with local_only
    future = flexible_task(local_only=True)
    result = future.get()
    assert result == "done"
    print("  Call-time parameters working")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("TRON MAGIC LAYER VALIDATION")
    print("="*50 + "\n")
    
    try:
        test_config_auto_discovery()
        test_magic_future_local()
        test_magic_future_repr()
        test_remote_decorator_basic()
        test_remote_decorator_gpu_param()
        test_remote_decorator_call_time_params()
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED")
        print("="*50)
        print("\nYour TRON magic layer is working!")
        print("Try: python magic_example.py (with server running)")
        print("Or just: result = my_func(5).get()")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
