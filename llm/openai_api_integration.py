# OpenAI API Integration for GPT-5 Support in Harfang RL-LLM
import os
import time
import json
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableLambda

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI integration not available. Install with: pip install langchain-openai")


class OpenAIAPIIntegration:
    """
    Integration for OpenAI API models (GPT-5) in the Harfang RL-LLM system.
    Provides seamless switching between local Ollama models and OpenAI API.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.api_key = None
        self.current_api_model = None
        
        # Check for API key
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            if self.verbose:
                print("[OPENAI] No API key found in OPENAI_API_KEY environment variable")
        else:
            if self.verbose:
                print("[OPENAI] API key found, OpenAI models available")
    
    def is_available(self) -> bool:
        """Check if OpenAI API integration is available"""
        return OPENAI_AVAILABLE and self.api_key is not None
    
    def initialize_gpt5(self, temperature: float = 0.0, max_tokens: int = 500) -> Optional[RunnableLambda]:
        """
        Initialize GPT-5 via OpenAI API
        
        Args:
            temperature: Response temperature (0.0 = deterministic)
            max_tokens: Maximum response tokens
        
        Returns:
            Initialized GPT-5 wrapped in RunnableLambda for compatibility
        """
        if not self.is_available():
            print("[ERROR] OpenAI API not available. Check API key and langchain-openai installation.")
            return None
        
        try:
            # Initialize GPT-5 (when available) or GPT-4 as fallback
            models_to_try = ["gpt-5", "gpt-4-turbo", "gpt-4"]
            
            for model_name in models_to_try:
                try:
                    chat = ChatOpenAI(
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=self.api_key,
                        timeout=30,
                        max_retries=3
                    )
                    
                    # Test the model
                    test_start = time.time()
                    test_response = chat.invoke("Respond with just 'TACTICAL READY'")
                    test_time = time.time() - test_start
                    
                    if "TACTICAL READY" in test_response.content:
                        print(f"[OPENAI] {model_name} ready! Response time: {test_time:.2f}s")
                        
                        # Wrap in RunnableLambda for compatibility
                        llm = RunnableLambda(lambda x: chat.invoke(x))
                        self.current_api_model = model_name
                        
                        return llm
                    else:
                        print(f"[WARNING] {model_name} test failed")
                        continue
                        
                except Exception as e:
                    if self.verbose:
                        print(f"[WARNING] Failed to initialize {model_name}: {e}")
                    continue
            
            print("[ERROR] All OpenAI models failed to initialize")
            return None
            
        except Exception as e:
            print(f"[ERROR] OpenAI initialization failed: {e}")
            return None
    
    def get_api_usage_info(self) -> Dict[str, Any]:
        """Get information about API usage and costs"""
        return {
            'api_available': self.is_available(),
            'current_model': self.current_api_model,
            'estimated_cost_per_1k_tokens': {
                'gpt-5': 'TBD',  # Will be announced by OpenAI
                'gpt-4-turbo': '$0.01',
                'gpt-4': '$0.03'
            },
            'recommended_usage': 'Research and high-quality evaluation, not continuous training',
            'rate_limits': 'Check OpenAI documentation for current limits'
        }
    
    def estimate_training_cost(self, num_episodes: int, steps_per_episode: int, 
                             llm_calls_per_step: float = 0.1) -> Dict[str, float]:
        """
        Estimate API costs for training
        
        Args:
            num_episodes: Number of training episodes
            steps_per_episode: Steps per episode
            llm_calls_per_step: LLM calls per step (with rate limiting)
        
        Returns:
            Cost estimates for different models
        """
        total_calls = num_episodes * steps_per_episode * llm_calls_per_step
        avg_tokens_per_call = 800  # Estimate: 300 input + 500 output
        total_tokens = total_calls * avg_tokens_per_call
        
        # Rough cost estimates (will need updating when GPT-5 pricing announced)
        cost_estimates = {
            'gpt-5': total_tokens * 0.00005,  # Estimated
            'gpt-4-turbo': total_tokens * 0.00001,
            'gpt-4': total_tokens * 0.00003,
            'total_calls': total_calls,
            'total_tokens': total_tokens
        }
        
        if self.verbose:
            print(f"[COST ESTIMATE] Total LLM calls: {total_calls:,.0f}")
            print(f"[COST ESTIMATE] Total tokens: {total_tokens:,.0f}")
            print(f"[COST ESTIMATE] GPT-5 estimated cost: ${cost_estimates['gpt-5']:.2f}")
            print(f"[COST ESTIMATE] GPT-4-turbo cost: ${cost_estimates['gpt-4-turbo']:.2f}")
        
        return cost_estimates


def create_hybrid_llm_system(primary_model: str = "gemma3:4b", 
                            api_model: str = "gpt-5",
                            use_api_for: str = "evaluation") -> Dict[str, Any]:
    """
    Create hybrid system using local models for training and API models for evaluation
    
    Args:
        primary_model: Local model for continuous training
        api_model: API model for high-quality evaluation
        use_api_for: When to use API model ('evaluation', 'research', 'never')
    
    Returns:
        Hybrid system configuration
    """
    from llm.multi_llm_manager import MultiLLMManager
    
    manager = MultiLLMManager()
    api_integration = OpenAIAPIIntegration()
    
    # Initialize primary local model
    primary_llm = manager.initialize_model(primary_model, temperature=0.0)
    
    # Initialize API model if available and requested
    api_llm = None
    if use_api_for != 'never' and api_integration.is_available():
        api_llm = api_integration.initialize_gpt5(temperature=0.0)
    
    hybrid_config = {
        'primary_model': primary_model,
        'primary_llm': primary_llm,
        'api_model': api_model if api_llm else None,
        'api_llm': api_llm,
        'use_api_for': use_api_for,
        'api_available': api_llm is not None,
        'cost_estimates': api_integration.get_api_usage_info() if api_llm else None
    }
    
    print(f"[HYBRID] Primary model: {manager.supported_models[primary_model].name}")
    if api_llm:
        print(f"[HYBRID] API model: {api_model} (for {use_api_for})")
    else:
        print(f"[HYBRID] API model not available")
    
    return hybrid_config


if __name__ == "__main__":
    # Test OpenAI integration
    api_integration = OpenAIAPIIntegration()
    
    if api_integration.is_available():
        print("Testing GPT-5 initialization...")
        gpt5 = api_integration.initialize_gpt5()
        if gpt5:
            print("✅ GPT-5 ready for tactical guidance")
        else:
            print("❌ GPT-5 initialization failed")
    else:
        print("OpenAI API not available - check API key and installation")
