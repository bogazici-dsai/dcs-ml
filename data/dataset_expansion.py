#!/usr/bin/env python3
# Dataset Expansion System for Combat LLM Fine-Tuning
import json
import numpy as np
import random
import time
import os
from typing import Dict, Any, List, Tuple
from pathlib import Path
import argparse


class TacticalDataExpander:
    """
    Expand existing GPT-4 tactical scenarios for comprehensive LLM fine-tuning.
    Takes ~200 base scenarios and expands to 2000+ through intelligent variation.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Expansion parameters
        self.aircraft_types = {
            'friendly': ['F-16C', 'F-18C', 'F-35A', 'F-22A', 'Eurofighter', 'Rafale'],
            'enemy': ['Su-27', 'Su-30', 'Su-35', 'MiG-29', 'J-20', 'J-16']
        }
        
        self.weapon_loadouts = {
            'F-16C': ['2 AIM-120C + 2 AIM-9X', '4 AIM-120C', '6 AIM-9X'],
            'F-18C': ['2 AIM-120C + 2 AIM-9X', '4 AIM-120C + 2 AIM-9X', '2 AIM-120C + 4 AIM-9X'],
            'Su-30': ['2 R-77 + 2 R-73', '4 R-77', '2 R-77 + 4 R-73'],
            'Su-35': ['4 R-77 + 2 R-73', '6 R-77', '2 R-77 + 4 R-73']
        }
        
        self.engagement_ranges = [2, 5, 8, 12, 20, 30, 50, 75, 100, 150]  # km
        self.altitude_bands = [500, 2000, 5000, 8000, 12000, 20000, 30000, 40000]  # feet
        self.weather_conditions = ['clear', 'cloudy', 'overcast', 'rain', 'snow', 'night', 'dawn', 'dusk']
        self.threat_environments = ['permissive', 'contested', 'denied', 'unknown']
        
        # Tactical scenario types
        self.scenario_types = [
            'head_on_bvr', 'beam_aspect_bvr', 'stern_aspect_bvr',
            'merge_geometry', 'wvr_dogfight', 'defensive_spiral',
            'multi_target', 'support_scenario', 'intercept_mission',
            'cap_patrol', 'escort_mission'
        ]
        
        print(f"[EXPANDER] Initialized with {len(self.aircraft_types['friendly'])} friendly aircraft types")
        print(f"[EXPANDER] {len(self.scenario_types)} scenario types available")
    
    def load_base_scenarios(self, main_branch_path: str = "../") -> List[Dict[str, Any]]:
        """Load base scenarios from main branch data files"""
        
        base_scenarios = []
        
        # Try to load from main branch
        data_files = [
            f"{main_branch_path}/data/dcs_gpt4_self_interpretation_dataset.jsonl",
            f"{main_branch_path}/data/dcs_ai_enhanced_training_data.jsonl"
        ]
        
        for filepath in data_files:
            if Path(filepath).exists():
                try:
                    with open(filepath, 'r') as f:
                        for line in f:
                            if line.strip():
                                scenario = json.loads(line)
                                base_scenarios.append(scenario)
                    
                    if self.verbose:
                        print(f"[LOAD] Loaded {len(base_scenarios)} scenarios from {filepath}")
                        
                except Exception as e:
                    print(f"[ERROR] Failed to load {filepath}: {e}")
            else:
                print(f"[WARNING] File not found: {filepath}")
        
        if not base_scenarios:
            print("[WARNING] No base scenarios found, will create synthetic data")
            base_scenarios = self._create_fallback_scenarios(100)
        
        print(f"[LOAD] Total base scenarios: {len(base_scenarios)}")
        return base_scenarios
    
    def expand_scenarios(self, base_scenarios: List[Dict[str, Any]], 
                        target_count: int = 2000) -> List[Dict[str, Any]]:
        """
        Expand base scenarios to target count through intelligent variation
        
        Args:
            base_scenarios: Original scenarios from GPT-4
            target_count: Target number of scenarios
        
        Returns:
            Expanded scenario list
        """
        if not base_scenarios:
            print("[ERROR] No base scenarios provided")
            return []
        
        print(f"[EXPAND] Expanding {len(base_scenarios)} → {target_count} scenarios")
        
        expanded = base_scenarios.copy()  # Keep originals
        
        # Calculate how many variations per base scenario
        variations_needed = target_count - len(base_scenarios)
        variations_per_base = max(1, variations_needed // len(base_scenarios))
        
        print(f"[EXPAND] Creating ~{variations_per_base} variations per base scenario")
        
        for base_scenario in base_scenarios:
            for _ in range(variations_per_base):
                if len(expanded) >= target_count:
                    break
                
                variation = self._create_intelligent_variation(base_scenario)
                expanded.append(variation)
        
        # Fill remaining slots with random variations
        while len(expanded) < target_count:
            base = random.choice(base_scenarios)
            variation = self._create_intelligent_variation(base)
            expanded.append(variation)
        
        print(f"[EXPAND] Expansion complete: {len(expanded)} scenarios")
        return expanded[:target_count]
    
    def _create_intelligent_variation(self, base_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Create an intelligent variation of a base scenario"""
        
        variation = base_scenario.copy()
        
        # Parse original scenario elements
        original_prompt = variation.get('prompt', '')
        original_completion = variation.get('completion', {})
        
        # Apply systematic variations
        modifications = []
        
        # 1. Range variation
        if 'Distance to enemy:' in original_prompt:
            new_range = random.choice(self.engagement_ranges)
            original_prompt = self._modify_distance(original_prompt, new_range)
            modifications.append(f"range_{new_range}km")
        
        # 2. Aircraft type variation
        friendly_aircraft = random.choice(self.aircraft_types['friendly'])
        enemy_aircraft = random.choice(self.aircraft_types['enemy'])
        original_prompt = self._modify_aircraft_types(original_prompt, friendly_aircraft, enemy_aircraft)
        modifications.append(f"{friendly_aircraft}_vs_{enemy_aircraft}")
        
        # 3. Loadout variation
        if friendly_aircraft in self.weapon_loadouts:
            new_loadout = random.choice(self.weapon_loadouts[friendly_aircraft])
            original_prompt = self._modify_loadout(original_prompt, friendly_aircraft, new_loadout)
            modifications.append(f"loadout_variant")
        
        # 4. Environmental variation
        weather = random.choice(self.weather_conditions)
        threat_env = random.choice(self.threat_environments)
        altitude = random.choice(self.altitude_bands)
        
        enhanced_prompt = f"""{original_prompt}

Environmental Conditions:
- Weather: {weather}
- Threat Environment: {threat_env}
- Operating Altitude: {altitude} feet
- Scenario Type: {random.choice(self.scenario_types)}"""
        
        modifications.extend([weather, threat_env, f"alt_{altitude}ft"])
        
        # 5. Modify completion based on variations
        modified_completion = self._adapt_completion_to_variations(
            original_completion, weather, threat_env, altitude, modifications
        )
        
        # Create variation
        variation.update({
            'prompt': enhanced_prompt,
            'completion': modified_completion,
            'variation_id': '_'.join(modifications[:3]),  # Unique identifier
            'base_scenario_id': base_scenario.get('scenario_id', 'unknown'),
            'expansion_type': 'intelligent_variation'
        })
        
        return variation
    
    def _modify_distance(self, prompt: str, new_distance: int) -> str:
        """Modify distance in prompt text"""
        import re
        
        # Find and replace distance
        pattern = r'Distance to enemy: (\d+) km'
        replacement = f'Distance to enemy: {new_distance} km'
        
        modified = re.sub(pattern, replacement, prompt)
        return modified
    
    def _modify_aircraft_types(self, prompt: str, friendly: str, enemy: str) -> str:
        """Modify aircraft types in prompt"""
        
        # Simple replacements (can be enhanced)
        modified = prompt.replace('F-16', friendly)
        modified = modified.replace('Su-30', enemy)
        
        return modified
    
    def _modify_loadout(self, prompt: str, aircraft: str, new_loadout: str) -> str:
        """Modify weapon loadout in prompt"""
        
        # Find and replace loadout
        if f'{aircraft} Loadout:' in prompt:
            lines = prompt.split('\n')
            for i, line in enumerate(lines):
                if f'{aircraft} Loadout:' in line:
                    lines[i] = f'- {aircraft} Loadout: {new_loadout}'
                    break
            modified = '\n'.join(lines)
        else:
            modified = f"{prompt}\n- {aircraft} Loadout: {new_loadout}"
        
        return modified
    
    def _adapt_completion_to_variations(self, original_completion: Dict[str, Any],
                                      weather: str, threat_env: str, altitude: int,
                                      modifications: List[str]) -> Dict[str, Any]:
        """Adapt tactical completion based on scenario variations"""
        
        if not isinstance(original_completion, dict):
            return original_completion
        
        adapted = original_completion.copy()
        
        # Adapt maneuver based on weather
        if weather in ['rain', 'snow', 'night']:
            if 'Maneuver' in adapted:
                adapted['Maneuver'] = f"{adapted['Maneuver']} (adapted for {weather} conditions)"
        
        # Adapt sensing based on weather
        if weather in ['cloudy', 'overcast', 'rain']:
            if 'Sensoring' in adapted:
                adapted['Sensoring'] = f"Enhanced radar scan (reduced visibility: {weather})"
        
        # Adapt countermeasures based on threat environment
        if threat_env in ['contested', 'denied']:
            if 'Countermeasuring' in adapted:
                adapted['Countermeasuring'] = f"Enhanced defensive measures ({threat_env} environment)"
        
        # Add environmental awareness
        adapted['environmental_factors'] = {
            'weather': weather,
            'threat_environment': threat_env,
            'altitude_band': altitude,
            'adaptations_applied': modifications[:3]
        }
        
        return adapted
    
    def _create_fallback_scenarios(self, count: int) -> List[Dict[str, Any]]:
        """Create fallback scenarios if no base data available"""
        
        fallback_scenarios = []
        
        for i in range(count):
            distance = random.choice(self.engagement_ranges)
            friendly = random.choice(self.aircraft_types['friendly'])
            enemy = random.choice(self.aircraft_types['enemy'])
            weather = random.choice(self.weather_conditions)
            
            scenario = {
                'prompt': f"""Tactical Situation:
- Distance to enemy: {distance} km
- Ego aircraft: {friendly}
- Enemy aircraft: {enemy}
- Weather: {weather}
- Threat environment: {random.choice(self.threat_environments)}
- Engagement type: {random.choice(self.scenario_types)}""",
                
                'completion': {
                    'Maneuver': self._generate_fallback_maneuver(distance),
                    'Sensoring': self._generate_fallback_sensing(distance),
                    'Firing': self._generate_fallback_firing(distance),
                    'Countermeasuring': self._generate_fallback_countermeasures(weather)
                },
                
                'scenario_id': f'fallback_{i}',
                'expansion_type': 'synthetic_fallback'
            }
            
            fallback_scenarios.append(scenario)
        
        print(f"[FALLBACK] Created {count} synthetic scenarios")
        return fallback_scenarios
    
    def _generate_fallback_maneuver(self, distance: int) -> str:
        """Generate appropriate maneuver for distance"""
        if distance > 50:
            return "Maintain course"
        elif distance > 20:
            return "Crank left"
        elif distance > 10:
            return "Notch right"
        else:
            return "Defensive spiral"
    
    def _generate_fallback_sensing(self, distance: int) -> str:
        """Generate appropriate sensing for distance"""
        if distance > 30:
            return "Scan with radar"
        elif distance > 10:
            return "Lock target STT"
        else:
            return "Lock target TWS"
    
    def _generate_fallback_firing(self, distance: int) -> str:
        """Generate appropriate firing for distance"""
        if distance > 25:
            return "Fire AIM-120C"
        elif distance > 8:
            return "Fire AIM-9L"
        elif distance > 2:
            return "Gun attack"
        else:
            return "Hold fire"
    
    def _generate_fallback_countermeasures(self, weather: str) -> str:
        """Generate appropriate countermeasures for weather"""
        if weather in ['rain', 'snow']:
            return "Deploy chaff"
        elif weather == 'night':
            return "Deploy flares"
        else:
            return "No action required"
    
    def save_expanded_dataset(self, scenarios: List[Dict[str, Any]], 
                            output_path: str = "data/combat_training_expanded.jsonl") -> str:
        """Save expanded dataset in JSONL format"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for scenario in scenarios:
                f.write(json.dumps(scenario) + '\n')
        
        print(f"[SAVE] Expanded dataset saved: {output_path} ({len(scenarios)} scenarios)")
        return output_path
    
    def create_evaluation_split(self, scenarios: List[Dict[str, Any]], 
                              eval_ratio: float = 0.1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Create train/evaluation split"""
        
        # Shuffle scenarios
        shuffled = scenarios.copy()
        random.shuffle(shuffled)
        
        # Split
        split_idx = int(len(shuffled) * (1 - eval_ratio))
        train_scenarios = shuffled[:split_idx]
        eval_scenarios = shuffled[split_idx:]
        
        print(f"[SPLIT] Train: {len(train_scenarios)}, Eval: {len(eval_scenarios)}")
        return train_scenarios, eval_scenarios
    
    def generate_statistics(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about the expanded dataset"""
        
        stats = {
            'total_scenarios': len(scenarios),
            'scenario_types': {},
            'distance_distribution': {},
            'aircraft_combinations': {},
            'weather_distribution': {},
            'completion_types': {}
        }
        
        # Analyze scenarios
        for scenario in scenarios:
            prompt = scenario.get('prompt', '')
            completion = scenario.get('completion', {})
            
            # Extract statistics
            if 'Scenario Type:' in prompt:
                scenario_type = prompt.split('Scenario Type:')[1].split('\n')[0].strip()
                stats['scenario_types'][scenario_type] = stats['scenario_types'].get(scenario_type, 0) + 1
            
            if 'Weather:' in prompt:
                weather = prompt.split('Weather:')[1].split('\n')[0].strip()
                stats['weather_distribution'][weather] = stats['weather_distribution'].get(weather, 0) + 1
            
            if isinstance(completion, dict):
                for key in completion.keys():
                    stats['completion_types'][key] = stats['completion_types'].get(key, 0) + 1
        
        return stats
    
    def export_expansion_report(self, scenarios: List[Dict[str, Any]], 
                              output_path: str = "data/expansion_report.json"):
        """Export detailed expansion report"""
        
        stats = self.generate_statistics(scenarios)
        
        report = {
            'expansion_summary': {
                'original_count': '~200 (estimated)',
                'expanded_count': len(scenarios),
                'expansion_factor': f"{len(scenarios) / 200:.1f}x",
                'expansion_date': time.time()
            },
            'dataset_statistics': stats,
            'quality_metrics': {
                'scenario_diversity': len(stats['scenario_types']),
                'weather_coverage': len(stats['weather_distribution']),
                'tactical_coverage': len(stats['completion_types'])
            },
            'fine_tuning_readiness': {
                'sufficient_data': len(scenarios) >= 1000,
                'diverse_scenarios': len(stats['scenario_types']) >= 5,
                'ready_for_lora': len(scenarios) >= 1000 and len(stats['scenario_types']) >= 5
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[REPORT] Expansion report saved: {output_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("DATASET EXPANSION SUMMARY")
        print(f"{'='*60}")
        print(f"Original scenarios: ~200")
        print(f"Expanded scenarios: {len(scenarios)}")
        print(f"Expansion factor: {len(scenarios) / 200:.1f}x")
        print(f"Scenario types: {len(stats['scenario_types'])}")
        print(f"Weather conditions: {len(stats['weather_distribution'])}")
        print(f"Ready for LoRA training: {report['fine_tuning_readiness']['ready_for_lora']}")
        
        return report


def main():
    """Main function for dataset expansion"""
    
    parser = argparse.ArgumentParser(description='Expand Combat Tactical Dataset for LLM Fine-Tuning')
    parser.add_argument('--target_count', type=int, default=2000,
                       help='Target number of scenarios')
    parser.add_argument('--main_branch_path', type=str, default="../",
                       help='Path to main branch for loading base data')
    parser.add_argument('--output_dir', type=str, default="data",
                       help='Output directory for expanded dataset')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TACTICAL DATASET EXPANSION FOR COMBAT LLM FINE-TUNING")
    print("="*80)
    print(f"Target scenarios: {args.target_count}")
    print(f"Main branch path: {args.main_branch_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize expander
    expander = TacticalDataExpander(verbose=args.verbose)
    
    # Load base scenarios
    print(f"\n[STEP 1] Loading base scenarios from main branch...")
    base_scenarios = expander.load_base_scenarios(args.main_branch_path)
    
    if not base_scenarios:
        print("[ERROR] No base scenarios available")
        return
    
    # Expand scenarios
    print(f"\n[STEP 2] Expanding scenarios to {args.target_count}...")
    expanded_scenarios = expander.expand_scenarios(base_scenarios, args.target_count)
    
    # Create train/eval split
    print(f"\n[STEP 3] Creating train/evaluation split...")
    train_scenarios, eval_scenarios = expander.create_evaluation_split(expanded_scenarios)
    
    # Save datasets
    print(f"\n[STEP 4] Saving expanded datasets...")
    train_path = f"{args.output_dir}/combat_training_expanded.jsonl"
    eval_path = f"{args.output_dir}/combat_evaluation_expanded.jsonl"
    
    expander.save_expanded_dataset(train_scenarios, train_path)
    expander.save_expanded_dataset(eval_scenarios, eval_path)
    
    # Generate report
    print(f"\n[STEP 5] Generating expansion report...")
    report = expander.export_expansion_report(expanded_scenarios, f"{args.output_dir}/expansion_report.json")
    
    print(f"\n✅ DATASET EXPANSION COMPLETE!")
    print(f"📁 Training data: {train_path}")
    print(f"📁 Evaluation data: {eval_path}")
    print(f"📊 Report: {args.output_dir}/expansion_report.json")
    
    if report['fine_tuning_readiness']['ready_for_lora']:
        print(f"\n🎯 READY FOR LORA FINE-TUNING!")
        print(f"   Base model: Gemma 3 4B")
        print(f"   Training examples: {len(train_scenarios)}")
        print(f"   Evaluation examples: {len(eval_scenarios)}")
    else:
        print(f"\n⚠️  Dataset may need more diversity for optimal fine-tuning")


if __name__ == "__main__":
    main()
