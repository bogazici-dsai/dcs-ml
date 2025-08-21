#!/usr/bin/env python3
# Test Advanced Features Implementation
import numpy as np
from typing import Dict, Any


def test_reward_system():
    """Test comprehensive reward system"""
    print("🎯 Testing Tactical Reward System...")
    
    try:
        from rewards.tactical_reward_system import TacticalRewardSystem, RewardWeights, create_reward_system
        
        # Create mock LLM assistant
        class MockLLMAssistant:
            def __init__(self):
                self.verbose = True
        
        mock_assistant = MockLLMAssistant()
        
        # Test reward system creation
        reward_system = create_reward_system(mock_assistant, use_curriculum=False)
        print("   ✅ Reward system created")
        
        # Test reward calculation
        mock_state = np.random.randn(25)
        mock_action = np.array([0.1, 0.0, 0.0, 0.0])
        mock_next_state = np.random.randn(25)
        mock_info = {
            'distance': 8000,
            'locked': True,
            'threat_level': 0.3,
            'engagement_phase': 'BVR',
            'energy_state': 'HIGH'
        }
        mock_llm_feedback = {
            'shaping_delta': 0.2,
            'tactical_assessment': {'situation': 'optimal_position'}
        }
        
        reward = reward_system.calculate_comprehensive_reward(
            mock_state, mock_action, mock_next_state, mock_info, mock_llm_feedback
        )
        
        print(f"   ✅ Reward calculation: {reward:.2f}")
        
        # Test curriculum reward system
        curriculum_reward_system = create_reward_system(mock_assistant, use_curriculum=True)
        print("   ✅ Curriculum reward system created")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Reward system test failed: {e}")
        return False


def test_curriculum_system():
    """Test curriculum learning system"""
    print("\n📚 Testing Curriculum Learning System...")
    
    try:
        from training.curriculum_manager import CombatCurriculum, CurriculumTrainingManager, create_combat_curriculum
        
        # Create curriculum
        curriculum = create_combat_curriculum(total_timesteps=1000000, difficulty_progression="standard")
        print("   ✅ Combat curriculum created")
        
        # Test stage progression
        current_stage = curriculum.get_current_stage()
        print(f"   ✅ Current stage: {current_stage.name}")
        
        # Test advancement criteria
        mock_performance = {
            'success_rate': 0.9,
            'energy_violations': 1,
            'control_smoothness': 0.8
        }
        should_advance = curriculum.should_advance_stage(mock_performance)
        print(f"   ✅ Advancement check: {should_advance}")
        
        # Test curriculum summary
        summary = curriculum.get_curriculum_summary()
        print(f"   ✅ Curriculum summary: {summary['total_stages']} stages")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Curriculum system test failed: {e}")
        return False


def test_mission_planning():
    """Test mission planning system"""
    print("\n🎯 Testing Mission Planning System...")
    
    try:
        from missions.mission_planner import MissionPlanner, MissionType
        
        # Create mission planner
        planner = MissionPlanner(verbose=False)
        print("   ✅ Mission planner created")
        
        # Generate test missions
        mission_types = [MissionType.AIR_SUPERIORITY, MissionType.INTERCEPT, MissionType.ESCORT]
        
        for mission_type in mission_types:
            mission = planner.generate_mission(mission_type, difficulty="medium")
            print(f"   ✅ {mission_type.value} mission: {len(mission.objectives)} objectives")
        
        # Test mission statistics
        stats = planner.get_mission_statistics()
        print(f"   ✅ Mission statistics: {stats['missions_generated']} generated")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Mission planning test failed: {e}")
        return False


def test_sensor_simulation():
    """Test advanced sensor simulation"""
    print("\n📡 Testing Advanced Sensor Simulation...")
    
    try:
        from sensors.advanced_sensor_suite import AdvancedSensorSuite, RadarSystem, RWRSystem
        
        # Create sensor suite
        sensors = AdvancedSensorSuite("F-16C")
        print("   ✅ Sensor suite created")
        
        # Test radar system
        own_pos = np.array([0, 0, 5000])
        own_att = np.array([0, 0, 0])
        
        targets = [{
            'id': 'enemy_1',
            'position': [10000, 2000, 5000],
            'type': 'fighter',
            'rcs': 3.0
        }]
        
        emitters = [{
            'id': 'enemy_radar',
            'position': [10000, 2000, 5000],
            'frequency': 9.0,
            'power': 1000,
            'platform': 'fighter'
        }]
        
        sensor_state = sensors.update_sensors(own_pos, own_att, targets, emitters)
        print(f"   ✅ Sensor update: {len(sensor_state)} state elements")
        
        # Test observation vector
        obs_vector = sensors.get_sensor_observation_vector()
        print(f"   ✅ Sensor observation vector: {obs_vector.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Sensor simulation test failed: {e}")
        return False


def test_performance_optimization():
    """Test performance optimization features"""
    print("\n⚡ Testing Performance Optimization...")
    
    try:
        from optimization.performance_optimizer import OptimizedTacticalAssistant, SituationCache
        
        # Create mock assistant
        class MockAssistant:
            def request_shaping(self, features, step=0):
                return 0.1, {"critique": "mock_response"}
            
            def extract_features(self, *args, **kwargs):
                return {"distance": 8000, "threat_level": 0.3}
        
        mock_assistant = MockAssistant()
        
        # Test situation cache
        cache = SituationCache(max_size=100)
        print("   ✅ Situation cache created")
        
        # Test optimized assistant
        optimized_assistant = OptimizedTacticalAssistant(
            mock_assistant,
            enable_caching=True,
            enable_async=True
        )
        print("   ✅ Optimized assistant created")
        
        # Test performance
        mock_features = {'distance': 8000, 'threat_level': 0.3}
        response = optimized_assistant.request_shaping(mock_features)
        print(f"   ✅ Optimized guidance: {response[0]:.2f}")
        
        # Test performance stats
        stats = optimized_assistant.get_optimization_stats()
        print(f"   ✅ Performance stats: {stats['total_calls']} calls")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance optimization test failed: {e}")
        return False


def test_hierarchical_actions():
    """Test hierarchical action space"""
    print("\n🎪 Testing Hierarchical Action Space...")
    
    try:
        from actions.hierarchical_action_space import HierarchicalActionSpace, MacroActionLibrary
        
        # Create hierarchical action space
        action_space = HierarchicalActionSpace(enable_macro_actions=True)
        print("   ✅ Hierarchical action space created")
        
        # Test macro action library
        library = MacroActionLibrary()
        print(f"   ✅ Macro library: {len(library.macro_actions)} actions")
        
        # Test action selection
        mock_agent_action = np.array([0.1, 0.0, 0.0, 0.0])
        mock_state = {
            'distance': 8000,
            'threat_level': 0.3,
            'engagement_phase': 'BVR',
            'energy_state': 'HIGH'
        }
        
        final_action, action_info = action_space.select_action(mock_agent_action, mock_state)
        print(f"   ✅ Action selection: {action_info['action_type']}")
        
        # Test action space info
        info = action_space.get_action_space_info()
        print(f"   ✅ Action space info: {info['available_macro_actions']} macro actions")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Hierarchical actions test failed: {e}")
        return False


def test_integration():
    """Test integration of all advanced features"""
    print("\n🔗 Testing Feature Integration...")
    
    try:
        # Test that all components can work together
        from llm.multi_llm_manager import MultiLLMManager
        from rewards.tactical_reward_system import create_reward_system
        from training.curriculum_manager import create_combat_curriculum
        from missions.mission_planner import MissionPlanner
        
        # Create components
        llm_manager = MultiLLMManager(verbose=False)
        
        class MockAssistant:
            def __init__(self):
                self.verbose = False
            def request_shaping(self, features, step=0):
                return 0.0, {"critique": "integration_test"}
        
        mock_assistant = MockAssistant()
        reward_system = create_reward_system(mock_assistant)
        curriculum = create_combat_curriculum(1000000)
        mission_planner = MissionPlanner(verbose=False)
        
        print("   ✅ All components integrated successfully")
        
        # Test component interaction
        mission = mission_planner.generate_mission()
        stage = curriculum.get_current_stage()
        
        print(f"   ✅ Integration test: Mission={mission.mission_type.value}, Stage={stage.name}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False


def main():
    """Run comprehensive test of advanced features"""
    
    print("="*80)
    print("ADVANCED FEATURES TEST SUITE")
    print("="*80)
    print("Testing newly implemented Phases 3-5...")
    
    tests = [
        ("Tactical Reward System", test_reward_system),
        ("Curriculum Learning", test_curriculum_system),
        ("Mission Planning", test_mission_planning),
        ("Sensor Simulation", test_sensor_simulation),
        ("Performance Optimization", test_performance_optimization),
        ("Hierarchical Actions", test_hierarchical_actions),
        ("Feature Integration", test_integration)
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
    print(f"\n{'='*80}")
    print("ADVANCED FEATURES TEST SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n🎉 ALL ADVANCED FEATURES WORKING!")
        print(f"🚀 System ready for advanced training!")
        print(f"\nNext steps:")
        print(f"1. Test full training: python enhanced_harfang_rl_llm.py --mode train")
        print(f"2. Install LoRA deps: pip install peft transformers datasets")
        print(f"3. Start fine-tuning: Setup Gemma 3 4B specialization")
    elif passed >= total - 1:
        print(f"\n⚠️  MOSTLY WORKING - Minor issues detected")
        print(f"🚀 System ready for testing with minor limitations")
    else:
        print(f"\n❌ SOME FEATURES FAILED - Check errors above")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    print(f"\n{'='*80}")
    if success:
        print("🎯 ADVANCED FEATURES READY!")
    else:
        print("🔧 Some features need attention")
    print(f"{'='*80}")
