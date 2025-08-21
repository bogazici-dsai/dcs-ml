# Multi-LLM Manager for Harfang Combat Training
import os
import time
import json
import subprocess
import numpy as np
from typing import Dict, Any, List, Optional, Union
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from dataclasses import dataclass


@dataclass
class LLMModelSpec:
    """Specification for an LLM model"""
    name: str
    ollama_name: str
    parameters: str
    context_length: int
    description: str
    recommended_use: str
    memory_requirement: str


class MultiLLMManager:
    """
    Manager for multiple LLM models optimized for tactical air combat guidance.
    Supports easy switching between different models and automatic model management.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.current_model = None
        self.current_llm = None
        self.model_cache = {}  # Cache loaded models for quick switching
        
        # Define supported models optimized for tactical reasoning
        self.supported_models = {
            # Gemma 3 models (Google) - Primary choices
            'gemma3:4b': LLMModelSpec(
                name="Gemma 3 4B",
                ollama_name="gemma3:4b",
                parameters="4B",
                context_length=8192,
                description="Latest Gemma model, excellent reasoning for size, optimized for efficiency",
                recommended_use="DEFAULT - Primary choice for tactical guidance",
                memory_requirement="~4GB"
            ),
            'gemma3:1b': LLMModelSpec(
                name="Gemma 3 1B", 
                ollama_name="gemma3:1b",
                parameters="1B",
                context_length=8192,
                description="Ultra-compact, fast inference, good for real-time applications",
                recommended_use="Ultra-fast inference, minimal resource environments",
                memory_requirement="~1GB"
            ),
            
            # Llama models (Meta) - Keep smaller ones
            'llama3.1:8b': LLMModelSpec(
                name="Llama 3.1 8B",
                ollama_name="llama3.1:8b",
                parameters="8B",
                context_length=128000,
                description="Excellent reasoning, good for complex tactical analysis",
                recommended_use="High-quality tactical guidance, research",
                memory_requirement="~8GB"
            ),
            'llama3.2:3b': LLMModelSpec(
                name="Llama 3.2 3B", 
                ollama_name="llama3.2:3b",
                parameters="3B",
                context_length=128000,
                description="Smaller, faster, good for real-time applications",
                recommended_use="Fast inference, resource-constrained environments",
                memory_requirement="~3GB"
            ),
            
            # Qwen 3 models (Alibaba) - Latest version
            'qwen3:7b': LLMModelSpec(
                name="Qwen 3 7B",
                ollama_name="qwen3:7b",
                parameters="7B",
                context_length=32768,
                description="Latest Qwen model, strong reasoning, good for tactical analysis",
                recommended_use="Alternative to Gemma/Llama, excellent performance",
                memory_requirement="~7GB"
            ),
            
            # GPT-OSS models (Open Source GPT)
            'gpt-oss:20b': LLMModelSpec(
                name="GPT-OSS 20B",
                ollama_name="gpt-oss:20b",
                parameters="20B",
                context_length=32768,
                description="Open source GPT model, high-quality reasoning",
                recommended_use="High-performance tactical analysis, research",
                memory_requirement="~20GB"
            ),
            
            # Phi models (Microsoft) - Keep for diversity
            'phi3:medium': LLMModelSpec(
                name="Phi 3 Medium",
                ollama_name="phi3:medium",
                parameters="14B",
                context_length=128000,
                description="Compact but powerful, optimized for reasoning",
                recommended_use="Balanced performance and efficiency",
                memory_requirement="~14GB"
            ),
            
            # OpenAI API models (requires API key)
            'gpt-5': LLMModelSpec(
                name="GPT-5 (OpenAI API)",
                ollama_name="gpt-5-api",  # Special identifier for API
                parameters="Unknown",
                context_length=200000,
                description="Latest OpenAI model, highest quality reasoning (requires API key)",
                recommended_use="Maximum performance, research, API-based deployment",
                memory_requirement="API-based (no local memory)"
            )
        }
        
        print(f"[MULTI LLM] Manager initialized with {len(self.supported_models)} supported models")
    
    def list_available_models(self) -> List[Dict[str, str]]:
        """List all supported models with their specifications"""
        
        print("\n" + "="*80)
        print("SUPPORTED LLM MODELS FOR TACTICAL COMBAT GUIDANCE")
        print("="*80)
        
        available_models = []
        
        for model_id, spec in self.supported_models.items():
            print(f"\n🤖 {spec.name} ({model_id})")
            print(f"   Parameters: {spec.parameters}")
            print(f"   Context: {spec.context_length:,} tokens")
            print(f"   Memory: {spec.memory_requirement}")
            print(f"   Use Case: {spec.recommended_use}")
            print(f"   Description: {spec.description}")
            
            # Check if model is available locally
            is_available = self._check_model_availability(spec.ollama_name)
            status = "✅ Available" if is_available else "❌ Not Downloaded"
            print(f"   Status: {status}")
            
            available_models.append({
                'id': model_id,
                'name': spec.name,
                'available': is_available,
                'parameters': spec.parameters,
                'memory': spec.memory_requirement,
                'recommended_use': spec.recommended_use
            })
        
        print("\n" + "="*80)
        return available_models
    
    def _check_model_availability(self, ollama_name: str) -> bool:
        """Check if a model is available locally in Ollama"""
        try:
            result = subprocess.run(
                ['ollama', 'list'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return ollama_name in result.stdout
        except Exception:
            return False
    
    def download_model(self, model_id: str) -> bool:
        """Download a model if not available"""
        if model_id not in self.supported_models:
            print(f"[ERROR] Model {model_id} not supported")
            return False
        
        spec = self.supported_models[model_id]
        
        if self._check_model_availability(spec.ollama_name):
            print(f"[INFO] Model {spec.name} already available")
            return True
        
        print(f"[DOWNLOAD] Downloading {spec.name} ({spec.parameters})...")
        print(f"[DOWNLOAD] This will require ~{spec.memory_requirement} of disk space")
        
        try:
            result = subprocess.run(
                ['ollama', 'pull', spec.ollama_name],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                print(f"[DOWNLOAD] {spec.name} downloaded successfully!")
                return True
            else:
                print(f"[ERROR] Download failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Download timed out after 30 minutes")
            return False
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            return False
    
    def initialize_model(self, model_id: str, temperature: float = 0.0, 
                        max_rate_hz: float = 10.0) -> Optional[RunnableLambda]:
        """
        Initialize a specific LLM model for tactical guidance
        
        Args:
            model_id: Model identifier from supported_models
            temperature: Response temperature (0.0 = deterministic)
            max_rate_hz: Maximum call rate in Hz
        
        Returns:
            Initialized LLM wrapped in RunnableLambda
        """
        if model_id not in self.supported_models:
            print(f"[ERROR] Model {model_id} not supported")
            print("Available models:", list(self.supported_models.keys()))
            return None
        
        spec = self.supported_models[model_id]
        
        # Special handling for OpenAI API models
        if model_id == 'gpt-5':
            from llm.openai_api_integration import OpenAIAPIIntegration
            api_integration = OpenAIAPIIntegration(verbose=self.verbose)
            llm = api_integration.initialize_gpt5(temperature=temperature)
            if llm:
                self.current_model = model_id
                self.current_llm = llm
                return llm
            else:
                print(f"[ERROR] Failed to initialize {spec.name}")
                return None
        
        # Check if model is available locally
        if not self._check_model_availability(spec.ollama_name):
            print(f"[WARNING] Model {spec.name} not available locally")
            download = input(f"Download {spec.name} (~{spec.memory_requirement})? [y/N]: ")
            if download.lower() == 'y':
                if not self.download_model(model_id):
                    return None
            else:
                return None
        
        # Check cache first
        cache_key = f"{model_id}_{temperature}"
        if cache_key in self.model_cache:
            print(f"[CACHE] Using cached {spec.name}")
            self.current_model = model_id
            self.current_llm = self.model_cache[cache_key]
            return self.current_llm
        
        # Initialize new model
        print(f"[INIT] Initializing {spec.name}...")
        
        try:
            chat = ChatOllama(
                model=spec.ollama_name,
                temperature=temperature,
                num_predict=500,  # Reasonable response length
                top_p=0.9,
                top_k=40,
                # Model-specific optimizations
                num_ctx=min(4096, spec.context_length),  # Reasonable context window
                repeat_penalty=1.1
            )
            
            llm = RunnableLambda(lambda x: chat.invoke(x))
            
            # Test the model
            test_start = time.time()
            test_response = llm.invoke("You are a fighter pilot. Respond with just 'READY FOR COMBAT'")
            test_time = time.time() - test_start
            
            if "READY FOR COMBAT" in str(test_response.content):
                print(f"[INIT] {spec.name} ready! Response time: {test_time:.2f}s")
                
                # Cache the model
                self.model_cache[cache_key] = llm
                self.current_model = model_id
                self.current_llm = llm
                
                return llm
            else:
                print(f"[WARNING] {spec.name} test failed: {test_response.content}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to initialize {spec.name}: {e}")
            return None
    
    def get_model_recommendations(self, use_case: str = "training") -> List[str]:
        """
        Get recommended models for specific use cases
        
        Args:
            use_case: 'training', 'inference', 'research', 'production'
        
        Returns:
            List of recommended model IDs
        """
        recommendations = {
            'training': ['gemma3:4b', 'llama3.1:8b', 'qwen3:7b'],
            'inference': ['gemma3:4b', 'gemma3:1b', 'llama3.2:3b'], 
            'research': ['gpt-5', 'gpt-oss:20b', 'phi3:medium'],
            'production': ['gemma3:4b', 'llama3.1:8b', 'qwen3:7b'],
            'fast': ['gemma3:1b', 'llama3.2:3b', 'gemma3:4b'],
            'quality': ['gpt-5', 'gpt-oss:20b', 'llama3.1:8b'],
            'default': ['gemma3:4b']  # Primary default choice
        }
        
        return recommendations.get(use_case, ['gemma3:4b'])
    
    def benchmark_models(self, model_ids: List[str], test_prompts: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Benchmark multiple models for response time and quality
        
        Args:
            model_ids: List of model IDs to benchmark
            test_prompts: Custom test prompts (optional)
        
        Returns:
            Benchmark results for each model
        """
        if test_prompts is None:
            test_prompts = [
                "You are an F-16 pilot. An enemy Su-27 is 5km ahead. What's your tactical recommendation?",
                "Analyze this air combat situation: Range 12km, enemy climbing, you have radar lock. Recommend action.",
                "Emergency: Missile incoming from 2 o'clock. Immediate defensive action required."
            ]
        
        print(f"\n[BENCHMARK] Testing {len(model_ids)} models with {len(test_prompts)} prompts")
        
        results = {}
        
        for model_id in model_ids:
            if model_id not in self.supported_models:
                print(f"[SKIP] {model_id} not supported")
                continue
            
            print(f"\n[BENCHMARK] Testing {self.supported_models[model_id].name}...")
            
            llm = self.initialize_model(model_id, temperature=0.0)
            if llm is None:
                print(f"[SKIP] {model_id} initialization failed")
                continue
            
            model_results = {
                'response_times': [],
                'response_lengths': [],
                'success_rate': 0.0,
                'avg_response_time': 0.0,
                'avg_response_length': 0.0
            }
            
            successful_responses = 0
            
            for i, prompt in enumerate(test_prompts):
                try:
                    start_time = time.time()
                    response = llm.invoke(prompt)
                    response_time = time.time() - start_time
                    
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    response_length = len(response_text)
                    
                    model_results['response_times'].append(response_time)
                    model_results['response_lengths'].append(response_length)
                    successful_responses += 1
                    
                    print(f"  Prompt {i+1}: {response_time:.2f}s, {response_length} chars")
                    
                except Exception as e:
                    print(f"  Prompt {i+1}: FAILED - {e}")
                    model_results['response_times'].append(float('inf'))
                    model_results['response_lengths'].append(0)
            
            # Calculate statistics
            valid_times = [t for t in model_results['response_times'] if t != float('inf')]
            valid_lengths = [l for l in model_results['response_lengths'] if l > 0]
            
            model_results['success_rate'] = successful_responses / len(test_prompts)
            model_results['avg_response_time'] = np.mean(valid_times) if valid_times else float('inf')
            model_results['avg_response_length'] = np.mean(valid_lengths) if valid_lengths else 0
            
            results[model_id] = model_results
            
            print(f"  Results: {model_results['success_rate']:.1%} success, "
                  f"{model_results['avg_response_time']:.2f}s avg time")
        
        # Print summary
        self._print_benchmark_summary(results)
        return results
    
    def _print_benchmark_summary(self, results: Dict[str, Dict[str, float]]):
        """Print formatted benchmark summary"""
        print(f"\n{'='*80}")
        print("LLM BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Success':<8} {'Avg Time':<10} {'Avg Length':<12} {'Recommendation'}")
        print("-" * 80)
        
        # Sort by success rate then by response time
        sorted_results = sorted(
            results.items(), 
            key=lambda x: (x[1]['success_rate'], -x[1]['avg_response_time']), 
            reverse=True
        )
        
        for model_id, result in sorted_results:
            spec = self.supported_models[model_id]
            success_rate = f"{result['success_rate']:.1%}"
            avg_time = f"{result['avg_response_time']:.2f}s"
            avg_length = f"{result['avg_response_length']:.0f}"
            
            # Simple recommendation
            if result['success_rate'] >= 0.9 and result['avg_response_time'] < 2.0:
                recommendation = "🟢 Excellent"
            elif result['success_rate'] >= 0.8 and result['avg_response_time'] < 5.0:
                recommendation = "🟡 Good"
            else:
                recommendation = "🔴 Poor"
            
            print(f"{spec.name:<20} {success_rate:<8} {avg_time:<10} {avg_length:<12} {recommendation}")
        
        print(f"\n[RECOMMENDATION] For training: Use top 2-3 models")
        print(f"[RECOMMENDATION] For production: Use fastest model with >90% success rate")
    
    def auto_select_best_model(self, use_case: str = "training", 
                              max_memory_gb: Optional[float] = None) -> Optional[str]:
        """
        Automatically select the best model for a given use case
        
        Args:
            use_case: 'training', 'inference', 'research', 'production'
            max_memory_gb: Maximum memory constraint in GB
        
        Returns:
            Best model ID or None if no suitable model found
        """
        candidates = self.get_model_recommendations(use_case)
        
        # Filter by memory constraint if specified
        if max_memory_gb is not None:
            filtered_candidates = []
            for model_id in candidates:
                spec = self.supported_models[model_id]
                # Extract memory requirement (rough parsing)
                memory_str = spec.memory_requirement.replace('~', '').replace('GB', '')
                try:
                    memory_gb = float(memory_str)
                    if memory_gb <= max_memory_gb:
                        filtered_candidates.append(model_id)
                except:
                    pass  # Skip if can't parse memory requirement
            candidates = filtered_candidates
        
        # Check availability and select first available
        for model_id in candidates:
            if self._check_model_availability(self.supported_models[model_id].ollama_name):
                print(f"[AUTO SELECT] Selected {self.supported_models[model_id].name} for {use_case}")
                return model_id
        
        # If no models available, suggest download
        if candidates:
            best_candidate = candidates[0]
            spec = self.supported_models[best_candidate]
            print(f"[AUTO SELECT] Best model {spec.name} not available")
            print(f"[SUGGESTION] Run: ollama pull {spec.ollama_name}")
            return best_candidate
        
        print(f"[ERROR] No suitable models found for {use_case}")
        return None
    
    def create_tactical_assistant(self, model_id: str, temperature: float = 0.0,
                                 max_rate_hz: float = 10.0, verbose: bool = True):
        """
        Create a HarfangTacticalAssistant with specified model
        
        Args:
            model_id: Model identifier
            temperature: Response temperature
            max_rate_hz: Maximum call rate
            verbose: Verbose output
        
        Returns:
            Configured HarfangTacticalAssistant
        """
        llm = self.initialize_model(model_id, temperature, max_rate_hz)
        if llm is None:
            raise ValueError(f"Failed to initialize model {model_id}")
        
        # Import here to avoid circular imports
        from HarfangAssistant_Enhanced import HarfangTacticalAssistant
        
        assistant = HarfangTacticalAssistant(
            llm=llm,
            verbose=verbose,
            max_rate_hz=max_rate_hz
        )
        
        # Add model info to assistant
        assistant.model_name = model_id
        assistant.model_spec = self.supported_models[model_id]
        
        print(f"[ASSISTANT] Created with {self.supported_models[model_id].name}")
        return assistant
    
    def get_system_requirements(self) -> Dict[str, Any]:
        """Get system requirements for running different models"""
        return {
            'minimum_ram_gb': 8,
            'recommended_ram_gb': 16,
            'gpu_recommended': True,
            'disk_space_per_model': "3-40GB depending on model size",
            'ollama_required': True,
            'supported_platforms': ['macOS', 'Linux', 'Windows'],
            'python_version': '3.8+',
            'torch_version': '1.12+'
        }


def interactive_model_selection() -> str:
    """Interactive model selection for users"""
    manager = MultiLLMManager()
    
    print("\n🎯 INTERACTIVE LLM MODEL SELECTION")
    print("="*50)
    
    available_models = manager.list_available_models()
    
    # Filter to only available models
    available_ids = [model['id'] for model in available_models if model['available']]
    
    if not available_ids:
        print("\n❌ No models available locally!")
        print("📥 Recommended downloads:")
        for model_id in ['llama3.1:8b', 'gemma2:9b', 'mistral:7b']:
            spec = manager.supported_models[model_id]
            print(f"   ollama pull {spec.ollama_name}  # {spec.name} ({spec.memory_requirement})")
        return None
    
    print(f"\n✅ Available models ({len(available_ids)}):")
    default_choice = None
    for i, model_id in enumerate(available_ids):
        spec = manager.supported_models[model_id]
        is_default = "DEFAULT" in spec.recommended_use
        default_marker = " [DEFAULT]" if is_default else ""
        print(f"   {i+1}. {spec.name} ({spec.parameters}) - {spec.recommended_use}{default_marker}")
        if is_default:
            default_choice = i + 1
    
    while True:
        try:
            prompt_text = f"\nSelect model (1-{len(available_ids)}), 'auto', or Enter for default"
            if default_choice:
                prompt_text += f" [{default_choice}]"
            prompt_text += ": "
            
            choice = input(prompt_text).strip()
            
            if choice == '' and default_choice:
                # Use default (Gemma 3 4B)
                selected = available_ids[default_choice - 1]
                break
            elif choice.lower() == 'auto':
                selected = manager.auto_select_best_model('training')
                break
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(available_ids):
                    selected = available_ids[idx]
                    break
                else:
                    print("Invalid selection!")
        except ValueError:
            print("Please enter a number, 'auto', or Enter for default")
    
    if selected:
        spec = manager.supported_models[selected]
        print(f"\n✅ Selected: {spec.name}")
        print(f"   Use Case: {spec.recommended_use}")
        print(f"   Memory: {spec.memory_requirement}")
    
    return selected


if __name__ == "__main__":
    # Example usage and testing
    manager = MultiLLMManager()
    
    print("Multi-LLM Manager for Harfang Combat Training")
    print("Run interactive_model_selection() to choose a model")
    
    # List available models
    manager.list_available_models()
    
    # Show system requirements
    requirements = manager.get_system_requirements()
    print("\nSystem Requirements:")
    for key, value in requirements.items():
        print(f"  {key}: {value}")
