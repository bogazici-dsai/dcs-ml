#!/usr/bin/env python3
# Complete Phase 6 Testing - Evaluation and Analysis Systems
import numpy as np
import json
import time
from typing import Dict, Any


def test_combat_evaluation_suite():
    """Test comprehensive combat evaluation suite"""
    print("🎯 Testing Combat Evaluation Suite...")
    
    try:
        from evaluation.combat_evaluation_suite import CombatEvaluationSuite, EvaluationScenario
        
        # Create evaluation suite
        eval_suite = CombatEvaluationSuite(verbose=False)
        print(f"   ✅ Evaluation suite created with {len(eval_suite.scenarios)} scenarios")
        
        # Test scenario configuration
        head_on_scenario = eval_suite.scenarios[EvaluationScenario.HEAD_ON_BVR]
        print(f"   ✅ Scenario config: {head_on_scenario.name} (difficulty: {head_on_scenario.difficulty_level})")
        
        # Test benchmark system
        benchmarks = eval_suite.benchmarks
        print(f"   ✅ Performance benchmarks: {len(benchmarks)} agent types")
        
        # Test mock agent evaluation
        class MockAgent:
            def predict(self, obs, deterministic=True):
                return np.random.uniform(-1, 1, 4), None
        
        class MockEnv:
            def __init__(self):
                self.action_space = type('ActionSpace', (), {'sample': lambda: np.random.uniform(-1, 1, 4)})()
            
            def reset(self, seed=None):
                obs = np.random.randn(25)
                info = {'distance': 8000, 'locked': False, 'threat_level': 0.3}
                return obs, info
            
            def step(self, action):
                obs = np.random.randn(25)
                reward = np.random.uniform(-1, 5)
                done = np.random.random() < 0.1  # 10% chance to end
                info = {
                    'distance': max(1000, np.random.uniform(2000, 15000)),
                    'locked': np.random.random() < 0.3,
                    'enemy_health': max(0, np.random.uniform(0, 1)),
                    'ego_health': 1.0,
                    'success': np.random.random() < 0.3
                }
                return obs, reward, done, False, info
        
        mock_agent = MockAgent()
        mock_env = MockEnv()
        
        # Test single scenario evaluation (reduced episodes for speed)
        print("   🧪 Testing scenario evaluation...")
        scenario_result = eval_suite._evaluate_scenario(
            mock_agent, mock_env, EvaluationScenario.HEAD_ON_BVR, num_episodes=3
        )
        print(f"   ✅ Scenario evaluation: {scenario_result['success_rate']:.1%} success rate")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Combat evaluation suite test failed: {e}")
        return False


def test_llm_effectiveness_analyzer():
    """Test LLM effectiveness analyzer"""
    print("\n🧠 Testing LLM Effectiveness Analyzer...")
    
    try:
        from analysis.llm_effectiveness_analyzer import LLMEffectivenessAnalyzer, LLMInteractionRecord
        
        # Create analyzer
        analyzer = LLMEffectivenessAnalyzer(analysis_window=100, verbose=False)
        print("   ✅ LLM analyzer created")
        
        # Test interaction recording
        for i in range(20):  # Record 20 interactions
            analyzer.record_llm_interaction(
                step=i,
                episode=0,
                tactical_situation="BVR_ENGAGEMENT" if i % 2 == 0 else "WVR_DOGFIGHT",
                llm_prompt=f"Mock prompt {i}",
                llm_response=f'{{"shaping_delta": {np.random.uniform(-0.3, 0.3):.2f}, "critique": "mock_response_{i}"}}',
                shaping_delta=np.random.uniform(-0.3, 0.3),
                response_time=np.random.uniform(0.5, 3.0),
                agent_action_before=np.random.uniform(-1, 1, 4),
                agent_action_after=np.random.uniform(-1, 1, 4),
                environment_reward=np.random.uniform(-1, 5),
                success_outcome=np.random.random() < 0.4
            )
        
        print(f"   ✅ Recorded {len(analyzer.interaction_history)} interactions")
        
        # Test analysis functions
        mock_episode_logs = [{'episode': i, 'success': np.random.random() < 0.4} for i in range(10)]
        
        analysis_result = analyzer.analyze_llm_impact(mock_episode_logs)
        print(f"   ✅ LLM impact analysis: {len(analysis_result)} analysis categories")
        
        # Test optimization summary
        optimization_summary = analyzer.get_optimization_summary()
        print(f"   ✅ Optimization summary: {optimization_summary['overall_llm_performance']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ LLM effectiveness analyzer test failed: {e}")
        return False


def test_comprehensive_integration():
    """Test integration of evaluation and analysis systems"""
    print("\n🔗 Testing Comprehensive Integration...")
    
    try:
        from evaluation.combat_evaluation_suite import create_comprehensive_evaluation_system
        from llm.multi_llm_manager import MultiLLMManager
        
        # Create mock LLM assistant
        class MockLLMAssistant:
            def __init__(self):
                self.verbose = False
            
            def request_shaping(self, features, step=0):
                return np.random.uniform(-0.3, 0.3), {
                    "critique": "mock_guidance",
                    "shaping_delta": np.random.uniform(-0.3, 0.3)
                }
        
        mock_assistant = MockLLMAssistant()
        
        # Test comprehensive evaluation system creation
        combat_eval, llm_analyzer = create_comprehensive_evaluation_system(mock_assistant)
        print("   ✅ Comprehensive evaluation system created")
        
        # Test that components work together
        print(f"   ✅ Combat eval scenarios: {len(combat_eval.scenarios)}")
        print(f"   ✅ LLM analyzer window: {llm_analyzer.analysis_window}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Comprehensive integration test failed: {e}")
        return False


def test_all_advanced_features_together():
    """Test that all advanced features work together"""
    print("\n🎪 Testing All Advanced Features Integration...")
    
    try:
        # Import all major components
        from rewards.tactical_reward_system import create_reward_system
        from training.curriculum_manager import create_combat_curriculum
        from missions.mission_planner import MissionPlanner
        from sensors.advanced_sensor_suite import AdvancedSensorSuite
        from optimization.performance_optimizer import OptimizedTacticalAssistant
        from actions.hierarchical_action_space import HierarchicalActionSpace
        from evaluation.combat_evaluation_suite import CombatEvaluationSuite
        from analysis.llm_effectiveness_analyzer import LLMEffectivenessAnalyzer
        
        print("   ✅ All advanced components imported successfully")
        
        # Create mock assistant
        class MockAssistant:
            def __init__(self):
                self.verbose = False
                self.max_rate_hz = 10.0
            def request_shaping(self, features, step=0):
                return 0.1, {"critique": "integrated_test"}
            def extract_features(self, *args, **kwargs):
                return {"distance": 8000, "threat_level": 0.3}
        
        mock_assistant = MockAssistant()
        
        # Test component creation
        reward_system = create_reward_system(mock_assistant)
        curriculum = create_combat_curriculum(500000)
        mission_planner = MissionPlanner(verbose=False)
        sensor_suite = AdvancedSensorSuite("F-16C")
        optimized_assistant = OptimizedTacticalAssistant(mock_assistant)
        action_space = HierarchicalActionSpace(enable_macro_actions=True)
        eval_suite = CombatEvaluationSuite(verbose=False)
        llm_analyzer = LLMEffectivenessAnalyzer(verbose=False)
        
        print("   ✅ All advanced components created successfully")
        
        # Test basic functionality of each
        mission = mission_planner.generate_mission()
        stage = curriculum.get_current_stage()
        mock_features = {"distance": 8000, "threat_level": 0.3}
        guidance = optimized_assistant.request_shaping(mock_features)
        
        print(f"   ✅ Integration test successful:")
        print(f"      Mission: {mission.mission_type.value}")
        print(f"      Curriculum stage: {stage.name}")
        print(f"      LLM guidance: {guidance[0]:.2f}")
        print(f"      Sensor suite: {sensor_suite.aircraft_type}")
        print(f"      Action space: {len(action_space.macro_library.macro_actions)} macro actions")
        print(f"      Evaluation: {len(eval_suite.scenarios)} scenarios")
        print(f"      Analysis: {llm_analyzer.analysis_window} window")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Advanced features integration test failed: {e}")
        return False


def run_final_system_validation():
    """Run final comprehensive system validation"""
    print("\n🔍 Final System Validation...")
    
    try:
        # Test that enhanced script works without errors
        import subprocess
        
        # Test help command
        result = subprocess.run([
            'python', 'enhanced_harfang_rl_llm.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("   ✅ Enhanced script help works")
        else:
            print(f"   ❌ Enhanced script help failed: {result.stderr}")
            return False
        
        # Test that all imports work
        test_imports = [
            'agents.enhanced_ppo_agent',
            'agents.multi_rl_trainer', 
            'llm.multi_llm_manager',
            'llm.multi_stage_tactical_assistant',
            'rewards.tactical_reward_system',
            'training.curriculum_manager',
            'missions.mission_planner',
            'sensors.advanced_sensor_suite',
            'optimization.performance_optimizer',
            'actions.hierarchical_action_space',
            'evaluation.combat_evaluation_suite',
            'analysis.llm_effectiveness_analyzer'
        ]
        
        for module_name in test_imports:
            try:
                __import__(module_name)
                print(f"   ✅ {module_name}")
            except Exception as e:
                print(f"   ❌ {module_name}: {e}")
                return False
        
        print("   ✅ All modules import successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Final validation failed: {e}")
        return False


def main():
    """Run complete Phase 6 testing and final validation"""
    
    print("="*80)
    print("PHASE 6 COMPLETION & FINAL SYSTEM VALIDATION")
    print("="*80)
    
    tests = [
        ("Combat Evaluation Suite", test_combat_evaluation_suite),
        ("LLM Effectiveness Analyzer", test_llm_effectiveness_analyzer),
        ("Comprehensive Integration", test_comprehensive_integration),
        ("All Advanced Features", test_all_advanced_features_together),
        ("Final System Validation", run_final_system_validation)
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
    print("PHASE 6 COMPLETION TEST SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n🎉 PHASE 6 COMPLETE - ALL SYSTEMS OPERATIONAL!")
        print(f"🚀 Harfang RL-LLM system fully prepared!")
        print(f"\n📋 SYSTEM READY FOR:")
        print(f"   1. Full-scale training")
        print(f"   2. Algorithm comparison")
        print(f"   3. LLM fine-tuning")
        print(f"   4. Performance optimization")
        print(f"\n🎯 Next: Start serious training or fine-tuning!")
    else:
        print(f"\n⚠️  {total-passed} issues remaining - address before training")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    print(f"\n{'='*80}")
    if success:
        print("🎯 PHASE 6 COMPLETE - SYSTEM READY FOR TRAINING!")
    else:
        print("🔧 Address remaining issues before proceeding")
    print(f"{'='*80}")
