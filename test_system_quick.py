#!/usr/bin/env python3
# Quick system test without WandB
import numpy as np
from llm.multi_llm_manager import MultiLLMManager
from llm.multi_stage_tactical_assistant import EnhancedTacticalAssistant
from env.mock_harfang_env import MockHarfangEnhancedEnv


def test_llm_integration():
    """Test LLM integration with Gemma 3 4B"""
    print("🧪 Testing LLM Integration...")
    
    # Initialize LLM manager
    manager = MultiLLMManager(verbose=True)
    
    # Initialize Gemma 3 4B
    llm = manager.initialize_model('gemma3:4b', temperature=0.0)
    if llm is None:
        print("❌ Failed to initialize Gemma 3 4B")
        return False
    
    # Test basic response
    response = llm.invoke("You are a fighter pilot. Respond with 'READY FOR COMBAT'")
    print(f"LLM Response: {response.content}")
    
    if "READY FOR COMBAT" in response.content:
        print("✅ Gemma 3 4B working correctly")
        return True
    else:
        print("❌ Unexpected LLM response")
        return False


def test_tactical_assistant():
    """Test enhanced tactical assistant"""
    print("\n🎯 Testing Tactical Assistant...")
    
    # Initialize LLM
    manager = MultiLLMManager(verbose=False)
    llm = manager.initialize_model('gemma3:4b', temperature=0.0)
    
    if llm is None:
        print("❌ LLM not available")
        return False
    
    # Initialize tactical assistant
    assistant = EnhancedTacticalAssistant(
        llm=llm,
        verbose=True,
        max_rate_hz=10.0,
        use_multi_stage=True
    )
    
    # Test feature extraction
    mock_features = {
        'distance': 8000,
        'aspect_angle': 15.0,
        'closure_rate': 250.0,
        'locked': 1,
        'threat_level': 0.3,
        'engagement_phase': 'BVR',
        'energy_state': 'HIGH',
        'action': [0.1, 0.0, 0.0, 0.0]
    }
    
    # Test tactical guidance
    try:
        shaping_delta, response = assistant.request_shaping(mock_features, step=1)
        print(f"✅ Tactical guidance working: shaping={shaping_delta:.3f}")
        print(f"   Response keys: {list(response.keys())}")
        return True
    except Exception as e:
        print(f"❌ Tactical assistant failed: {e}")
        return False


def test_mock_environment():
    """Test mock Harfang environment"""
    print("\n🎮 Testing Mock Environment...")
    
    # Initialize environment
    env = MockHarfangEnhancedEnv(max_episode_steps=100)
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Environment reset: obs shape {obs.shape}, info keys: {list(info.keys())}")
    
    # Test step
    action = np.array([0.1, 0.0, 0.0, 0.0])  # Small pitch input
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"✅ Environment step: reward={reward:.2f}, terminated={terminated}")
    print(f"   Distance: {info['distance']/1000:.1f}km, Phase: {info['engagement_phase']}")
    
    env.close()
    return True


def test_training_components():
    """Test training components without full training"""
    print("\n🤖 Testing Training Components...")
    
    try:
        from agents.enhanced_ppo_agent import create_enhanced_ppo_config
        config = create_enhanced_ppo_config()
        print(f"✅ PPO config created: {len(config)} parameters")
        
        from agents.multi_rl_trainer import MultiRLTrainer
        print("✅ MultiRLTrainer importable")
        
        return True
    except Exception as e:
        print(f"❌ Training components failed: {e}")
        return False


def main():
    """Run comprehensive system test"""
    print("="*60)
    print("ENHANCED HARFANG RL-LLM SYSTEM TEST")
    print("="*60)
    
    tests = [
        ("LLM Integration", test_llm_integration),
        ("Tactical Assistant", test_tactical_assistant),
        ("Mock Environment", test_mock_environment),
        ("Training Components", test_training_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"🚀 System ready for training!")
        print(f"\nNext steps:")
        print(f"1. python enhanced_harfang_rl_llm.py --mode train --use_wandb=false")
        print(f"2. python enhanced_harfang_rl_llm.py --algorithm ALL")
        print(f"3. python data/dataset_expansion.py --target_count 2000")
    else:
        print(f"\n⚠️  Some tests failed - check errors above")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    print(f"\n{'='*60}")
    if success:
        print("🎯 SYSTEM READY FOR ENHANCED RL-LLM TRAINING!")
    else:
        print("🔧 Fix issues above before proceeding")
    print(f"{'='*60}")
