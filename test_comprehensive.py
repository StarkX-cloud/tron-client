"""
TRON Testing Guide - Verify Everything Works
"""

import sys
import os

# Add TRON to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_backward_compatibility():
    """Test that old APIs still work."""
    print("\n" + "="*50)
    print("TEST: Backward Compatibility (Old APIs)")
    print("="*50)
    
    try:
        # Old import style should still work
        from tron import task, Tron, serialize, deserialize, submit, status
        print("✓ Old imports work")
        
        # Can still use @task
        @task
        def old_style_task():
            return "old"
        
        print("✓ @task decorator works")
        
        # Serialization still works
        payload = serialize((lambda x: x, (1,), {}))
        print("✓ serialize works")
        
        # Config function exists
        from tron import config
        print("✓ config() function exists")
        
        return True
    except Exception as e:
        print(f"✗ Backward compatibility broken: {e}")
        return False


def test_new_magic_api():
    """Test the new magic layer."""
    print("\n" + "="*50)
    print("TEST: New Magic API")
    print("="*50)
    
    try:
        import tron
        
        # Basic remote decorator
        @tron.remote
        def simple_task(x):
            return x * 2
        
        result = simple_task(5)
        print(f"✓ @remote decorator: {type(result).__name__}")
        
        # With GPU
        @tron.remote(gpu=True, memory_gb=8)
        def gpu_task():
            return "gpu"
        
        print("✓ @remote with parameters")
        
        # Get result locally
        output = simple_task(10).get()
        assert output == 20, f"Expected 20, got {output}"
        print(f"✓ .get() blocking works: 10 * 2 = {output}")
        
        # Status check
        future = simple_task(100)
        assert future.ready() or future.done()
        print("✓ .ready()/.done() methods work")
        
        return True
    except Exception as e:
        print(f"✗ Magic API failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_system():
    """Test auto-discovery and configuration."""
    print("\n" + "="*50)
    print("TEST: Configuration & Auto-Discovery")
    print("="*50)
    
    try:
        from tron.config import get_config
        
        config = get_config()
        url = config.url
        print(f"✓ Server URL: {url}")
        
        # Environment variable respected
        os.environ["TRON_URL"] = "http://custom:9000"
        from tron.config import _config as fresh_config
        fresh_config._discovered = False  # Reset discovery
        fresh_config.server_url = None
        
        new_url = fresh_config.url
        assert "custom" in new_url, f"Env var not respected: {new_url}"
        print(f"✓ Environment variable respected: {new_url}")
        
        # TRON_SERVER as alternate name
        if "TRON_URL" in os.environ:
            del os.environ["TRON_URL"]
        os.environ["TRON_SERVER"] = "http://alias:9000"
        fresh_config._discovered = False
        fresh_config.server_url = None
        alias_url = fresh_config.url
        assert "alias" in alias_url, f"TRON_SERVER env var not respected: {alias_url}"
        print(f"✓ Alternate env var TRON_SERVER respected: {alias_url}")
        
        if "TRON_SERVER" in os.environ:
            del os.environ["TRON_SERVER"]
        
        return True
    except Exception as e:
        print(f"✗ Config system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_server_registry():
    """Test named server registry and selection."""
    print("\n" + "="*50)
    print("TEST: Server Registry")
    print("="*50)
    
    try:
        import tron
        
        tron.add_server("dev", "http://dev-server:9000")
        tron.add_server("prod", "https://prod-server")
        servers = tron.list_servers()
        assert "dev" in servers and "prod" in servers
        print(f"✓ Named servers registered: {list(servers.keys())}")
        
        selected = tron.use_server("prod")
        assert selected == "https://prod-server"
        assert tron.get_config().url == selected
        print(f"✓ Selected named server prod -> {selected}")
        
        return True
    except Exception as e:
        print(f"✗ Server registry failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensure_server():
    """Test ensure_server auto-starts a local TRON server if needed."""
    print("\n" + "="*50)
    print("TEST: Ensure Server")
    print("="*50)
    
    try:
        import tron

        # Remove any explicit config so ensure_server can choose
        from tron.config import _config as fresh_config
        fresh_config.server_url = None
        fresh_config._discovered = False

        url = tron.ensure_server(wait=True, timeout=5)
        print(f"✓ ensure_server connected to: {url}")

        assert url.startswith("http://"), f"Unexpected URL: {url}"
        assert tron.get_config().url == url

        stopped = tron.stop_local_server()
        assert stopped is True, "Expected stop_local_server() to stop the server"
        print("✓ Local server started and stopped successfully")

        return True
    except Exception as e:
        print(f"✗ Ensure server failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_local_execution():
    """Test that local execution works."""
    print("\n" + "="*50)
    print("TEST: Local Execution (No Server Needed)")
    print("="*50)
    
    try:
        import tron
        
        @tron.remote
        def expensive_computation(n):
            return sum(range(n))
        
        # Should execute locally
        result = expensive_computation(100).get()
        expected = sum(range(100))
        assert result == expected, f"Expected {expected}, got {result}"
        print(f"✓ Local execution: sum(0..99) = {result}")
        
        # With multiple calls
        r1 = expensive_computation(10).get()
        r2 = expensive_computation(20).get()
        r3 = expensive_computation(30).get()
        
        assert r1 == sum(range(10))
        assert r2 == sum(range(20))
        assert r3 == sum(range(30))
        print(f"✓ Multiple local calls: {r1}, {r2}, {r3}")
        
        return True
    except Exception as e:
        print(f"✗ Local execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_execution():
    """Test parallel execution."""
    print("\n" + "="*50)
    print("TEST: Parallel Execution")
    print("="*50)
    
    try:
        import tron
        
        @tron.remote
        def square(x):
            return x ** 2
        
        # Fire off multiple tasks
        futures = [square(x) for x in range(1, 6)]
        print(f"✓ Submitted {len(futures)} tasks")
        
        # Collect results
        results = [f.get() for f in futures]
        expected = [1, 4, 9, 16, 25]
        assert results == expected, f"Expected {expected}, got {results}"
        print(f"✓ Results: {results}")
        
        return True
    except Exception as e:
        print(f"✗ Parallel execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling."""
    print("\n" + "="*50)
    print("TEST: Error Handling")
    print("="*50)
    
    try:
        import tron
        
        @tron.remote
        def failing_task():
            raise ValueError("Intentional error")
        
        # Force local-only so it doesn't try to submit to missing server
        try:
            result = failing_task(local_only=True).get()
            print("✗ Error was not raised")
            return False
        except (ValueError, RuntimeError) as e:
            # Should see the ValueError from local execution
            error_msg = str(e)
            if "Intentional error" in error_msg or "local execution failed" in error_msg:
                print(f"✓ Error handling works: {type(e).__name__}")
                return True
            else:
                print(f"✗ Wrong error: {e}")
                return False
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_server_gracefully():
    """Test that it works without a server."""
    print("\n" + "="*50)
    print("TEST: Works Without Server")
    print("="*50)
    
    try:
        import tron
        
        @tron.remote
        def task():
            return "works anyway"
        
        # Should work because it falls back to local
        result = task().get()
        assert result == "works anyway"
        print(f"✓ Functions work even without server: {result}")
        
        return True
    except Exception as e:
        print(f"✗ No-server test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("╔" + "="*48 + "╗")
    print("║" + " TRON COMPREHENSIVE TEST SUITE ".center(48) + "║")
    print("╚" + "="*48 + "╝")
    
    tests = [
        ("Backward Compatibility", test_backward_compatibility),
        ("New Magic API", test_new_magic_api),
        ("Configuration System", test_config_system),
        ("Local Execution", test_local_execution),
        ("Parallel Execution", test_parallel_execution),
        ("Error Handling", test_error_handling),
        ("Works Without Server", test_no_server_gracefully),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "="*50)
    if passed == total:
        print(f"🎉 ALL {total} TESTS PASSED!")
        print("="*50)
        return 0
    else:
        print(f"❌ {total - passed}/{total} tests failed")
        print("="*50)
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    
    print("\n📋 WHAT TO TEST NEXT:\n")
    print("1. With Server Running:")
    print("   - Start your TRON server (python queue_server.py)")
    print("   - Run: python test_comprehensive.py")
    print("   - Tasks should auto-route to remote workers\n")
    
    print("2. With Docker:")
    print("   - docker-compose up")
    print("   - python test_comprehensive.py\n")
    
    print("3. Integration Test:")
    print("   - python magic_example.py (with server)\n")
    
    sys.exit(exit_code)
