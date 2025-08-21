#!/usr/bin/env python3
# Enhanced System Setup and Validation Script
import os
import sys
import subprocess
import json
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'numpy', 'gymnasium', 'stable_baselines3',
        'langchain', 'langchain_ollama', 'loguru'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n📦 Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    print("✅ All core dependencies available")
    return True


def check_ollama_installation():
    """Check if Ollama is installed and running"""
    print("\n🦙 Checking Ollama...")
    
    try:
        # Check if Ollama is installed
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Ollama installed")
        else:
            print("   ❌ Ollama not found")
            return False
    except FileNotFoundError:
        print("   ❌ Ollama not installed")
        print("   📥 Install: curl -fsSL https://ollama.com/install.sh | sh")
        return False
    
    # Check if Ollama service is running
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("   ✅ Ollama service running")
            
            # Check for default model
            if 'gemma3:4b' in result.stdout:
                print("   ✅ Gemma 3 4B available")
            else:
                print("   📥 Download default model: ollama pull gemma3:4b")
            
            return True
        else:
            print("   ❌ Ollama service not running")
            print("   🚀 Start service: ollama serve")
            return False
    except subprocess.TimeoutExpired:
        print("   ❌ Ollama service timeout")
        return False
    except Exception as e:
        print(f"   ❌ Ollama check failed: {e}")
        return False


def check_harfang_environment():
    """Check if Harfang environment is available"""
    print("\n🎮 Checking Harfang environment...")
    
    try:
        from env.hirl.environments.HarfangEnv_GYM_Enhanced import HarfangEnhancedEnv
        print("   ✅ HarfangEnhancedEnv available")
        return 'enhanced'
    except ImportError:
        try:
            from env.hirl.environments.HarfangEnv_GYM_ppo_v2 import HarfangEnv
            print("   ⚠️  Using HarfangEnv V2 (Enhanced not available)")
            return 'v2'
        except ImportError:
            try:
                from env.hirl.environments.HarfangEnv_GYM import HarfangEnv
                print("   ⚠️  Using basic HarfangEnv (V2 not available)")
                return 'basic'
            except ImportError:
                try:
                    from env.mock_harfang_env import MockHarfangEnhancedEnv
                    print("   ✅ Using mock Harfang environment for testing")
                    print("   📥 Install Harfang3D for full functionality")
                    return 'mock'
                except ImportError:
                    print("   ❌ No environment available (including mock)")
                    return None


def test_enhanced_components():
    """Test enhanced system components"""
    print("\n🧪 Testing enhanced components...")
    
    components = [
        ('agents.enhanced_ppo_agent', 'EnhancedPPOAgent'),
        ('agents.multi_rl_trainer', 'MultiRLTrainer'),
        ('llm.multi_llm_manager', 'MultiLLMManager'),
        ('llm.multi_stage_tactical_assistant', 'MultiStageTacticalAssistant'),
        ('llm.openai_api_integration', 'OpenAIAPIIntegration'),
        ('llm.lora_finetuning_foundation', 'LoRACombatTrainer')
    ]
    
    all_working = True
    
    for module_name, class_name in components:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"   ✅ {class_name}")
        except ImportError as e:
            print(f"   ❌ {class_name} - Import error: {e}")
            all_working = False
        except AttributeError as e:
            print(f"   ❌ {class_name} - Class not found: {e}")
            all_working = False
        except Exception as e:
            print(f"   ⚠️  {class_name} - Warning: {e}")
    
    return all_working


def setup_directories():
    """Setup required directories"""
    print("\n📁 Setting up directories...")
    
    directories = [
        'logs', 'models', 'data/enhanced_harfang_logs',
        'models/enhanced_ppo', 'models/multi_rl_comparison',
        'agents', 'llm'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}")
    
    return True


def run_system_test():
    """Run a quick system test"""
    print("\n🧪 Running system test...")
    
    try:
        # Test LLM manager
        from llm.multi_llm_manager import MultiLLMManager
        manager = MultiLLMManager(verbose=False)
        print("   ✅ MultiLLMManager initialized")
        
        # Test model recommendations
        recommendations = manager.get_model_recommendations('training')
        print(f"   ✅ Model recommendations: {recommendations}")
        
        # Test LoRA foundation
        from llm.lora_finetuning_foundation import setup_lora_foundation
        foundation = setup_lora_foundation()
        print(f"   ✅ LoRA foundation ready: {foundation['training_ready']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ System test failed: {e}")
        return False


def main():
    """Main setup and validation function"""
    
    print("="*80)
    print("ENHANCED HARFANG RL-LLM SYSTEM SETUP")
    print("="*80)
    
    # Check all components
    checks = [
        ("Dependencies", check_dependencies),
        ("Ollama", check_ollama_installation), 
        ("Harfang Environment", check_harfang_environment),
        ("Enhanced Components", test_enhanced_components),
        ("Directories", setup_directories),
        ("System Test", run_system_test)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"\n{'='*20} {check_name} {'='*20}")
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"❌ {check_name} failed: {e}")
            results[check_name] = False
    
    # Summary
    print(f"\n{'='*80}")
    print("SETUP SUMMARY")
    print(f"{'='*80}")
    
    passed = 0
    total = len(checks)
    
    for check_name, result in results.items():
        if result:
            print(f"✅ {check_name}")
            passed += 1
        else:
            print(f"❌ {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print(f"\n🎉 SYSTEM READY!")
        print(f"🚀 Start training: python enhanced_harfang_rl_llm.py --mode train")
        print(f"📖 Full guide: see ENHANCED_SYSTEM_GUIDE.md")
    elif passed >= total - 1:
        print(f"\n⚠️  MOSTLY READY - Minor issues detected")
        print(f"🚀 You can likely start training with: python enhanced_harfang_rl_llm.py --mode train")
    else:
        print(f"\n❌ SETUP INCOMPLETE - {total - passed} critical issues")
        print(f"🔧 Fix issues above before training")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
