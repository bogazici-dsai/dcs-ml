#!/usr/bin/env python3
# test_enhanced_system.py - Quick validation test for enhanced RL-LLM system

import numpy as np
import sys
import os

def test_imports():
    """Test that all enhanced components can be imported"""
    print("Testing component imports...")
    
    try:
        from HarfangAssistant_Enhanced import HarfangTacticalAssistant
        print("HarfangTacticalAssistant imported successfully")
    except Exception as e:
        print(f"FAILED to import HarfangTacticalAssistant: {e}")
        return False
    
    try:
        from utils.enhanced_dataset_logger import TacticalDataLogger
        print("TacticalDataLogger imported successfully")
    except Exception as e:
        print(f"FAILED to import TacticalDataLogger: {e}")
        return False
    
    try:
        from action_space_optimizer import ActionSpaceOptimizer
        print("ActionSpaceOptimizer imported successfully")
    except Exception as e:
        print(f"FAILED to import ActionSpaceOptimizer: {e}")
        return False
    
    # Test environment imports (may fail if Harfang not installed)
    try:
        from env.hirl.environments.HarfangEnv_GYM_Enhanced import HarfangEnhancedEnv
        print("HarfangEnhancedEnv imported successfully")
        enhanced_available = True
    except Exception as e:
        print(f"WARNING HarfangEnhancedEnv not available (expected if Harfang not installed): {e}")
        enhanced_available = False
    
    try:
        from env.hirl.environments.HarfangEnv_GYM_ppo_v2 import HarfangEnv
        print("HarfangEnv V2 imported successfully")
        v2_available = True
    except Exception as e:
        print(f"WARNING HarfangEnv V2 not available: {e}")
        v2_available = False
    
    return True, enhanced_available, v2_available

def test_tactical_assistant():
    """Test tactical assistant functionality without LLM"""
    print("\n[AI] Testing TacticalAssistant features...")
    
    try:
        # Create mock assistant (without actual LLM)
        assistant = HarfangTacticalAssistant(llm=None, verbose=False)
        
        # Test feature extraction with mock data
        mock_state = np.random.randn(25)  # Enhanced state size
        mock_action = np.random.randn(4)
        mock_info = {'distance': 2000, 'threat_level': 0.3}
        
        features = assistant.extract_features(
            state=mock_state,
            prev_state=None,
            action=mock_action,
            info=mock_info,
            lock_duration=5
        )
        
        print(f"Feature extraction successful: {len(features)} features")
        print(f"   Key features: {list(features.keys())[:5]}...")
        
        # Test tactical assessments
        assert 'tactical_situation' in features
        assert 'engagement_phase' in features
        assert 'pursuit_geometry' in features
        print("Tactical assessment methods working")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] TacticalAssistant test failed: {e}")
        return False

def test_data_logger():
    """Test comprehensive data logging"""
    print("\n[DATA] Testing TacticalDataLogger...")
    
    try:
        import tempfile
        import os
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TacticalDataLogger(
                out_dir=temp_dir,
                filename_prefix="test",
                create_separate_files=True
            )
            
            # Test logging a step
            mock_state = np.random.randn(25)
            mock_action = np.random.randn(4)
            mock_features = {
                'distance': 2000,
                'tactical_situation': 'OFFENSIVE',
                'threat_level': 0.3,
                'engagement_phase': 'WVR'
            }
            mock_llm_response = {
                'shaping_delta': 0.1,
                'critique': 'Good positioning'
            }
            
            logger.log_step(
                episode=0,
                step=0,
                state=mock_state,
                action=mock_action,
                base_reward=1.0,
                shaping_delta=0.1,
                done=False,
                info={'success': False},
                features=mock_features,
                llm_response=mock_llm_response
            )
            
            # Test episode metrics logging
            episode_data = {
                'length': 100,
                'total_reward': 50.0,
                'victory': False,
                'shot_accuracy': 0.5
            }
            logger.log_episode_metrics(0, episode_data)
            
            # Check files were created
            files = os.listdir(temp_dir)
            expected_files = ['test_', '_llm_', '_metrics_', '_events_']
            found_files = [f for f in files if any(expected in f for expected in expected_files)]
            
            print(f"Data logging successful: {len(found_files)} files created")
            logger.close()
        
        return True
        
    except Exception as e:
        print(f"[FAIL] TacticalDataLogger test failed: {e}")
        return False

def test_action_optimizer():
    """Test action space optimizer"""
    print("\nTesting ActionSpaceOptimizer...")
    
    try:
        initial_bounds = {
            'pitch': (-1.0, 1.0),
            'roll': (-1.0, 1.0),
            'yaw': (-1.0, 1.0),
            'fire': (0.0, 1.0)
        }
        
        optimizer = ActionSpaceOptimizer(
            initial_action_bounds=initial_bounds,
            optimization_window=100,
            min_samples_for_optimization=10
        )
        
        # Simulate some data collection
        for i in range(15):
            mock_action = np.random.randn(4)
            mock_reward = np.random.randn()
            mock_llm_response = {
                'shaping_delta': np.random.randn() * 0.1,
                'critique': 'test critique',
                'action_space_ops': {'add': ['defensive_spiral']} if i % 5 == 0 else {}
            }
            mock_tactical_metrics = {
                'tactical_situation': 'NEUTRAL',
                'threat_level': np.random.rand(),
                'distance': 2000 + np.random.randn() * 500
            }
            
            optimizer.record_step(
                action=mock_action,
                reward=mock_reward,
                llm_response=mock_llm_response,
                tactical_metrics=mock_tactical_metrics,
                step_info={}
            )
        
        # Test analysis
        analysis = optimizer.analyze_action_effectiveness()
        print(f"Action optimization analysis: {analysis['status']}")
        
        # Test report generation
        report = optimizer.get_optimization_report()
        print(f"Optimization report generated: {len(report)} sections")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] ActionSpaceOptimizer test failed: {e}")
        return False

def test_main_integration():
    """Test that main harfang_rl_llm.py can be imported"""
    print("\nTesting main integration script...")
    
    try:
        # This might fail due to missing dependencies, but should at least parse
        import harfang_rl_llm
        print("Main integration script imported successfully")
        
        # Test that we can access the main function
        assert hasattr(harfang_rl_llm, 'main')
        print("Main function accessible")
        
        return True
        
    except ImportError as e:
        if 'langchain' in str(e) or 'ChatOllama' in str(e):
            print("WARNING Main script import failed due to missing LLM dependencies (expected)")
            print("   Install langchain-ollama to use LLM features")
            return True
        else:
            print(f"[FAIL] Main integration test failed: {e}")
            return False
    except Exception as e:
        print(f"[FAIL] Main integration test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("[INFO] ENHANCED HARFANG RL-LLM SYSTEM VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Component Imports", test_imports),
        ("Tactical Assistant", test_tactical_assistant),
        ("Data Logger", test_data_logger),
        ("Action Optimizer", test_action_optimizer),
        ("Main Integration", test_main_integration)
    ]
    
    results = []
    enhanced_env_available = False
    
    for test_name, test_func in tests:
        print(f"\n[TEST] Running {test_name} test...")
        try:
            if test_name == "Component Imports":
                success, enhanced_available, v2_available = test_func()
                enhanced_env_available = enhanced_available
                results.append((test_name, success))
            else:
                success = test_func()
                results.append((test_name, success))
        except Exception as e:
            print(f"[FAIL] {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("[LIST] TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if enhanced_env_available:
        print("Enhanced environment available - full functionality ready")
    else:
        print("WARNING Enhanced environment not available - install Harfang3D for full functionality")
    
    print("\nNext steps:")
    if passed == total:
        print("   All tests passed! System ready for use.")
        print("   Run: python harfang_rl_llm.py --help")
    else:
        print("   Some tests failed. Check error messages above.")
        print("   [TOOL] Install missing dependencies and retry.")
    
    print("\n[DOCS] Documentation: README_ENHANCED_SYSTEM.md")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
