# Advanced Sensor Simulation for Realistic Air Combat Training
import numpy as np
import math
import random
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RadarMode(Enum):
    """Radar operating modes"""
    OFF = "off"
    STANDBY = "standby"
    SEARCH = "search"
    TRACK = "track"
    STT = "stt"  # Single Target Track
    TWS = "tws"  # Track While Scan
    FLOOD = "flood"  # Flood mode for jamming


class ThreatType(Enum):
    """Types of radar threats"""
    FIGHTER = "fighter"
    INTERCEPTOR = "interceptor"
    SAM = "sam"
    AWACS = "awacs"
    JAMMER = "jammer"
    UNKNOWN = "unknown"


@dataclass
class RadarContact:
    """Radar contact information"""
    contact_id: str
    azimuth: float          # degrees
    elevation: float        # degrees
    range: float           # meters
    range_rate: float      # m/s (closure rate)
    rcs: float             # radar cross section (m²)
    confidence: float      # detection confidence 0-1
    track_quality: float   # track quality 0-1
    first_detection_time: float
    last_update_time: float
    threat_type: ThreatType
    locked: bool = False


@dataclass
class RWRContact:
    """RWR (Radar Warning Receiver) contact"""
    emitter_id: str
    frequency: float       # GHz
    azimuth: float        # degrees (relative)
    signal_strength: float # dBm
    threat_type: ThreatType
    threat_level: int     # 1-5 (5 = immediate threat)
    lock_indication: bool
    missile_launch_warning: bool
    first_detection_time: float


class RadarSystem:
    """
    Realistic radar system simulation with proper detection physics,
    beam patterns, and electronic warfare effects.
    """
    
    def __init__(self, aircraft_type: str = "F-16C"):
        """
        Initialize radar system based on aircraft type
        
        Args:
            aircraft_type: Type of aircraft (determines radar specs)
        """
        self.aircraft_type = aircraft_type
        
        # Radar specifications (based on real systems)
        radar_specs = self._get_radar_specs(aircraft_type)
        
        self.max_range = radar_specs['max_range']
        self.beam_width = radar_specs['beam_width']
        self.update_rate = radar_specs['update_rate']
        self.peak_power = radar_specs['peak_power']
        self.frequency = radar_specs['frequency']
        self.antenna_gain = radar_specs['antenna_gain']
        
        # Operating state
        self.mode = RadarMode.OFF
        self.azimuth_center = 0.0  # degrees
        self.elevation_center = 0.0  # degrees
        self.scan_pattern = "raster"  # raster, circular, sector
        self.scan_volume = {"az_min": -60, "az_max": 60, "el_min": -20, "el_max": 20}
        
        # Contact tracking
        self.contacts = {}  # Dict[str, RadarContact]
        self.locked_target = None
        self.track_files = {}  # Persistent track data
        
        # Electronic warfare effects
        self.jamming_effects = {
            'noise_jamming': 0.0,     # 0-1 intensity
            'deception_jamming': 0.0,  # 0-1 intensity
            'chaff_density': 0.0       # 0-1 density
        }
        
        print(f"[RADAR] {aircraft_type} radar system initialized")
        print(f"[RADAR] Max range: {self.max_range/1000:.0f}km, Beam width: {self.beam_width:.1f}°")
    
    def _get_radar_specs(self, aircraft_type: str) -> Dict[str, float]:
        """Get radar specifications for aircraft type"""
        
        radar_database = {
            'F-16C': {  # AN/APG-68
                'max_range': 120000,    # 120km
                'beam_width': 2.8,      # degrees
                'update_rate': 4.0,     # Hz
                'peak_power': 1000,     # kW
                'frequency': 9.5,       # GHz (X-band)
                'antenna_gain': 36      # dB
            },
            'F-18C': {  # AN/APG-73
                'max_range': 130000,    # 130km
                'beam_width': 2.5,      # degrees
                'update_rate': 4.5,     # Hz
                'peak_power': 1200,     # kW
                'frequency': 9.3,       # GHz
                'antenna_gain': 38      # dB
            },
            'F-22A': {  # AN/APG-77
                'max_range': 200000,    # 200km
                'beam_width': 2.0,      # degrees
                'update_rate': 10.0,    # Hz
                'peak_power': 2000,     # kW
                'frequency': 9.8,       # GHz
                'antenna_gain': 42      # dB
            },
            'default': {  # Generic fighter radar
                'max_range': 100000,    # 100km
                'beam_width': 3.0,      # degrees
                'update_rate': 3.0,     # Hz
                'peak_power': 800,      # kW
                'frequency': 9.0,       # GHz
                'antenna_gain': 34      # dB
            }
        }
        
        return radar_database.get(aircraft_type, radar_database['default'])
    
    def set_mode(self, mode: RadarMode, target_azimuth: float = 0.0, 
                target_elevation: float = 0.0):
        """Set radar operating mode"""
        
        self.mode = mode
        
        if mode == RadarMode.STT:
            # Single Target Track - narrow beam on specific target
            self.azimuth_center = target_azimuth
            self.elevation_center = target_elevation
            self.scan_volume = {
                "az_min": target_azimuth - 2,
                "az_max": target_azimuth + 2,
                "el_min": target_elevation - 2,
                "el_max": target_elevation + 2
            }
        elif mode == RadarMode.TWS:
            # Track While Scan - wide area search with tracking
            self.scan_volume = {"az_min": -60, "az_max": 60, "el_min": -20, "el_max": 20}
        elif mode == RadarMode.SEARCH:
            # Search mode - maximum coverage
            self.scan_volume = {"az_min": -90, "az_max": 90, "el_min": -30, "el_max": 30}
        
        print(f"[RADAR] Mode set to {mode.value.upper()}")
    
    def update(self, own_position: np.ndarray, own_attitude: np.ndarray,
              targets: List[Dict[str, Any]], dt: float = 1.0) -> List[RadarContact]:
        """
        Update radar system and return detected contacts
        
        Args:
            own_position: Own aircraft position [x, y, z] in meters
            own_attitude: Own aircraft attitude [pitch, roll, yaw] in radians
            targets: List of potential targets with position and properties
            dt: Time delta in seconds
        
        Returns:
            List of radar contacts
        """
        
        if self.mode == RadarMode.OFF:
            return []
        
        detected_contacts = []
        current_time = time.time()
        
        for target in targets:
            # Calculate target geometry
            target_pos = np.array(target['position'])
            relative_pos = target_pos - own_position
            
            # Convert to spherical coordinates (aircraft frame)
            range_m = np.linalg.norm(relative_pos)
            azimuth, elevation = self._calculate_look_angles(relative_pos, own_attitude)
            
            # Check if target is within scan volume
            if not self._is_in_scan_volume(azimuth, elevation):
                continue
            
            # Calculate detection probability
            detection_prob = self._calculate_detection_probability(
                range_m, target.get('rcs', 1.0), azimuth, elevation
            )
            
            # Apply jamming effects
            detection_prob *= self._apply_jamming_effects(range_m, target.get('jamming', 0))
            
            # Detection decision
            if random.random() < detection_prob:
                contact = self._create_radar_contact(
                    target, azimuth, elevation, range_m, current_time
                )
                detected_contacts.append(contact)
                
                # Update persistent track
                self._update_track_file(contact)
        
        # Update contact list
        self._update_contact_list(detected_contacts, current_time)
        
        return list(self.contacts.values())
    
    def _calculate_look_angles(self, relative_pos: np.ndarray, 
                              own_attitude: np.ndarray) -> Tuple[float, float]:
        """Calculate azimuth and elevation angles to target"""
        
        # Transform to aircraft body frame
        # Simplified transformation (full implementation would use rotation matrices)
        range_m = np.linalg.norm(relative_pos)
        
        if range_m == 0:
            return 0.0, 0.0
        
        # Calculate angles (simplified)
        azimuth = math.degrees(math.atan2(relative_pos[1], relative_pos[0]))
        elevation = math.degrees(math.atan2(relative_pos[2], 
                                          math.sqrt(relative_pos[0]**2 + relative_pos[1]**2)))
        
        return azimuth, elevation
    
    def _is_in_scan_volume(self, azimuth: float, elevation: float) -> bool:
        """Check if angles are within current scan volume"""
        
        return (self.scan_volume["az_min"] <= azimuth <= self.scan_volume["az_max"] and
                self.scan_volume["el_min"] <= elevation <= self.scan_volume["el_max"])
    
    def _calculate_detection_probability(self, range_m: float, rcs: float,
                                       azimuth: float, elevation: float) -> float:
        """Calculate detection probability using radar equation"""
        
        # Simplified radar equation: Pd = f(Power, Gain, RCS, Range, Noise)
        
        # Range factor (4th power law)
        range_factor = (self.max_range / max(range_m, 1000)) ** 4
        
        # RCS factor
        rcs_factor = rcs / 1.0  # Normalized to 1 m² reference
        
        # Antenna gain factor (beam pattern)
        angle_offset = math.sqrt(azimuth**2 + elevation**2)
        gain_factor = max(0.1, 1.0 - (angle_offset / self.beam_width)**2)
        
        # Base detection probability
        base_pd = min(0.95, range_factor * rcs_factor * gain_factor * 0.8)
        
        # Mode-specific adjustments
        if self.mode == RadarMode.STT:
            base_pd *= 1.3  # Better detection in STT mode
        elif self.mode == RadarMode.SEARCH:
            base_pd *= 0.7  # Reduced detection in search mode
        
        return max(0.0, min(0.95, base_pd))
    
    def _apply_jamming_effects(self, range_m: float, target_jamming: float) -> float:
        """Apply electronic warfare effects to detection probability"""
        
        # Noise jamming effect
        noise_reduction = 1.0 - (self.jamming_effects['noise_jamming'] * 0.8)
        
        # Deception jamming effect
        deception_reduction = 1.0 - (self.jamming_effects['deception_jamming'] * 0.6)
        
        # Chaff effect (range-dependent)
        chaff_reduction = 1.0 - (self.jamming_effects['chaff_density'] * 
                                max(0, 1.0 - range_m / 10000))  # Chaff effective < 10km
        
        # Target self-jamming
        target_jamming_reduction = 1.0 - (target_jamming * 0.7)
        
        # Combined effect
        total_reduction = (noise_reduction * deception_reduction * 
                          chaff_reduction * target_jamming_reduction)
        
        return max(0.1, total_reduction)  # Minimum 10% effectiveness
    
    def _create_radar_contact(self, target: Dict[str, Any], azimuth: float,
                            elevation: float, range_m: float, 
                            current_time: float) -> RadarContact:
        """Create radar contact from detected target"""
        
        target_id = target.get('id', f"tgt_{random.randint(1000, 9999)}")
        
        # Calculate range rate (simplified)
        range_rate = target.get('closure_rate', random.uniform(-300, 300))
        
        # Determine threat type based on target characteristics
        threat_type = self._classify_threat_type(target)
        
        contact = RadarContact(
            contact_id=target_id,
            azimuth=azimuth,
            elevation=elevation,
            range=range_m,
            range_rate=range_rate,
            rcs=target.get('rcs', 1.0),
            confidence=self._calculate_track_confidence(range_m, target.get('rcs', 1.0)),
            track_quality=self._calculate_track_quality(range_m, azimuth, elevation),
            first_detection_time=current_time,
            last_update_time=current_time,
            threat_type=threat_type,
            locked=False
        )
        
        return contact
    
    def _classify_threat_type(self, target: Dict[str, Any]) -> ThreatType:
        """Classify target threat type based on characteristics"""
        
        # Simple classification based on target properties
        aircraft_type = target.get('type', 'unknown').lower()
        
        if 'fighter' in aircraft_type or 'f-' in aircraft_type or 'su-' in aircraft_type:
            return ThreatType.FIGHTER
        elif 'mig-' in aircraft_type or 'interceptor' in aircraft_type:
            return ThreatType.INTERCEPTOR
        elif 'sam' in aircraft_type or 'missile' in aircraft_type:
            return ThreatType.SAM
        elif 'awacs' in aircraft_type or 'aew' in aircraft_type:
            return ThreatType.AWACS
        else:
            return ThreatType.UNKNOWN
    
    def _calculate_track_confidence(self, range_m: float, rcs: float) -> float:
        """Calculate track confidence based on range and RCS"""
        
        # Base confidence from detection strength
        range_factor = max(0.1, 1.0 - (range_m / self.max_range))
        rcs_factor = min(1.0, rcs / 1.0)  # Normalized to 1 m²
        
        base_confidence = (range_factor * 0.7 + rcs_factor * 0.3)
        
        # Mode-specific confidence
        if self.mode == RadarMode.STT:
            base_confidence *= 1.2  # Higher confidence in STT
        elif self.mode == RadarMode.SEARCH:
            base_confidence *= 0.8  # Lower confidence in search
        
        return max(0.1, min(0.95, base_confidence))
    
    def _calculate_track_quality(self, range_m: float, azimuth: float, 
                               elevation: float) -> float:
        """Calculate track quality based on geometry and range"""
        
        # Range quality (better at medium ranges)
        optimal_range = self.max_range * 0.4  # 40% of max range is optimal
        range_quality = 1.0 - abs(range_m - optimal_range) / optimal_range
        range_quality = max(0.2, min(1.0, range_quality))
        
        # Angle quality (better at boresight)
        angle_offset = math.sqrt(azimuth**2 + elevation**2)
        angle_quality = max(0.3, 1.0 - (angle_offset / 45.0))  # Degrade beyond 45°
        
        # Combined quality
        track_quality = (range_quality * 0.6 + angle_quality * 0.4)
        
        return track_quality
    
    def lock_target(self, contact_id: str) -> bool:
        """Attempt to lock specific target"""
        
        if contact_id not in self.contacts:
            return False
        
        contact = self.contacts[contact_id]
        
        # Lock probability based on track quality and range
        lock_prob = contact.track_quality * contact.confidence
        
        # Apply jamming effects to lock probability
        lock_prob *= self._apply_jamming_effects(contact.range, 0.0)
        
        if random.random() < lock_prob:
            contact.locked = True
            self.locked_target = contact_id
            self.set_mode(RadarMode.STT, contact.azimuth, contact.elevation)
            
            print(f"[RADAR] Target {contact_id} locked at {contact.range/1000:.1f}km")
            return True
        else:
            print(f"[RADAR] Lock attempt failed on {contact_id}")
            return False
    
    def break_lock(self):
        """Break current target lock"""
        if self.locked_target:
            if self.locked_target in self.contacts:
                self.contacts[self.locked_target].locked = False
            self.locked_target = None
            self.set_mode(RadarMode.TWS)  # Return to TWS mode
            print("[RADAR] Target lock broken")
    
    def get_radar_state(self) -> Dict[str, Any]:
        """Get current radar state for environment observation"""
        
        state = {
            'mode': self.mode.value,
            'contacts_count': len(self.contacts),
            'locked_target': self.locked_target,
            'scan_azimuth': self.azimuth_center,
            'scan_elevation': self.elevation_center,
            'jamming_level': max(self.jamming_effects.values()),
            'max_range': self.max_range,
            'beam_width': self.beam_width
        }
        
        # Add locked target information
        if self.locked_target and self.locked_target in self.contacts:
            locked_contact = self.contacts[self.locked_target]
            state.update({
                'locked_range': locked_contact.range,
                'locked_azimuth': locked_contact.azimuth,
                'locked_track_quality': locked_contact.track_quality,
                'locked_threat_type': locked_contact.threat_type.value
            })
        
        return state


class RWRSystem:
    """
    Radar Warning Receiver system for threat detection and classification
    """
    
    def __init__(self, aircraft_type: str = "F-16C"):
        """Initialize RWR system"""
        self.aircraft_type = aircraft_type
        
        # RWR specifications
        self.frequency_coverage = (2.0, 18.0)  # GHz coverage
        self.azimuth_coverage = 360.0  # degrees (full coverage)
        self.sensitivity = -80  # dBm minimum detectable signal
        
        # Threat library
        self.threat_library = self._load_threat_library()
        
        # Current contacts
        self.rwr_contacts = {}  # Dict[str, RWRContact]
        self.threat_priorities = []
        
        print(f"[RWR] System initialized for {aircraft_type}")
        print(f"[RWR] Frequency coverage: {self.frequency_coverage[0]}-{self.frequency_coverage[1]} GHz")
    
    def _load_threat_library(self) -> Dict[str, Dict[str, Any]]:
        """Load threat emitter library"""
        
        # Simplified threat library (real systems have hundreds of emitters)
        library = {
            'AN/APG-68': {  # F-16 radar
                'frequency': 9.5,
                'threat_level': 3,
                'platform': 'F-16',
                'engagement_range': 120000
            },
            'N001': {  # Su-27 radar
                'frequency': 9.0,
                'threat_level': 3,
                'platform': 'Su-27',
                'engagement_range': 100000
            },
            'SA-10': {  # S-300 SAM
                'frequency': 5.6,
                'threat_level': 5,
                'platform': 'SAM',
                'engagement_range': 150000
            },
            'SA-15': {  # Tor SAM
                'frequency': 15.0,
                'threat_level': 4,
                'platform': 'SAM',
                'engagement_range': 12000
            }
        }
        
        return library
    
    def update(self, own_position: np.ndarray, emitters: List[Dict[str, Any]]) -> List[RWRContact]:
        """
        Update RWR system with current emitter environment
        
        Args:
            own_position: Own aircraft position
            emitters: List of radar emitters in environment
        
        Returns:
            List of RWR contacts
        """
        
        detected_contacts = []
        current_time = time.time()
        
        for emitter in emitters:
            # Calculate emitter geometry
            emitter_pos = np.array(emitter['position'])
            relative_pos = emitter_pos - own_position
            range_m = np.linalg.norm(relative_pos)
            
            # Calculate azimuth (simplified - relative to aircraft nose)
            azimuth = math.degrees(math.atan2(relative_pos[1], relative_pos[0]))
            
            # Check if emitter is detectable
            if self._is_emitter_detectable(emitter, range_m):
                contact = self._create_rwr_contact(emitter, azimuth, range_m, current_time)
                detected_contacts.append(contact)
        
        # Update contact list and prioritize threats
        self.rwr_contacts = {contact.emitter_id: contact for contact in detected_contacts}
        self._prioritize_threats()
        
        return list(self.rwr_contacts.values())
    
    def _prioritize_threats(self):
        """Prioritize RWR threats by threat level"""
        # Simple prioritization - could be enhanced
        pass
    
    def _is_emitter_detectable(self, emitter: Dict[str, Any], range_m: float) -> bool:
        """Check if emitter is detectable by RWR"""
        
        frequency = emitter.get('frequency', 10.0)
        power = emitter.get('power', 1000)  # kW
        
        # Frequency coverage check
        if not (self.frequency_coverage[0] <= frequency <= self.frequency_coverage[1]):
            return False
        
        # Signal strength calculation (simplified)
        # Real calculation: received_power = transmitted_power * gain / (4π * range²)²
        signal_strength = 10 * math.log10(power / (range_m / 1000)**2)  # Simplified
        
        return signal_strength > self.sensitivity
    
    def _create_rwr_contact(self, emitter: Dict[str, Any], azimuth: float,
                          range_m: float, current_time: float) -> RWRContact:
        """Create RWR contact from detected emitter"""
        
        emitter_id = emitter.get('id', f"emit_{random.randint(1000, 9999)}")
        frequency = emitter.get('frequency', 10.0)
        
        # Classify threat
        threat_type = self._classify_emitter_threat(emitter)
        threat_level = self._assess_threat_level(emitter, range_m)
        
        # Check for lock/launch indications
        lock_indication = emitter.get('tracking_mode', False)
        missile_warning = emitter.get('missile_launched', False)
        
        contact = RWRContact(
            emitter_id=emitter_id,
            frequency=frequency,
            azimuth=azimuth,
            signal_strength=emitter.get('signal_strength', -60),
            threat_type=threat_type,
            threat_level=threat_level,
            lock_indication=lock_indication,
            missile_launch_warning=missile_warning,
            first_detection_time=current_time
        )
        
        return contact
    
    def _classify_emitter_threat(self, emitter: Dict[str, Any]) -> ThreatType:
        """Classify emitter threat type"""
        
        frequency = emitter.get('frequency', 10.0)
        platform = emitter.get('platform', 'unknown').lower()
        
        # Classification based on frequency and platform
        if 'sam' in platform:
            return ThreatType.SAM
        elif 'fighter' in platform or 8.0 <= frequency <= 12.0:
            return ThreatType.FIGHTER
        elif 'awacs' in platform or frequency < 5.0:
            return ThreatType.AWACS
        else:
            return ThreatType.UNKNOWN
    
    def _assess_threat_level(self, emitter: Dict[str, Any], range_m: float) -> int:
        """Assess threat level (1-5 scale)"""
        
        base_threat = 1
        
        # Range-based threat
        if range_m < 20000:
            base_threat += 2  # Close threats are dangerous
        elif range_m < 50000:
            base_threat += 1
        
        # Platform-based threat
        platform = emitter.get('platform', 'unknown').lower()
        if 'sam' in platform:
            base_threat += 2  # SAMs are high threat
        elif 'fighter' in platform:
            base_threat += 1
        
        # Mode-based threat
        if emitter.get('tracking_mode', False):
            base_threat += 1  # Being tracked
        if emitter.get('missile_launched', False):
            base_threat = 5  # Immediate threat
        
        return min(5, base_threat)


class ElectronicWarfareSuite:
    """
    Electronic Warfare suite including jammers and countermeasures
    """
    
    def __init__(self, aircraft_type: str = "F-16C"):
        """Initialize EW suite"""
        self.aircraft_type = aircraft_type
        
        # Countermeasure inventory
        self.chaff_count = 60
        self.flare_count = 60
        self.max_chaff = 60
        self.max_flares = 60
        
        # Jammer specifications
        self.jammer_power = 1000  # watts
        self.jammer_frequency_range = (8.0, 12.0)  # GHz
        self.jammer_active = False
        
        # Countermeasure effectiveness
        self.chaff_effectiveness = 0.8  # vs radar missiles
        self.flare_effectiveness = 0.9  # vs IR missiles
        
        print(f"[EW SUITE] Initialized for {aircraft_type}")
        print(f"[EW SUITE] Countermeasures: {self.chaff_count} chaff, {self.flare_count} flares")
    
    def deploy_chaff(self, count: int = 1) -> bool:
        """Deploy chaff countermeasures"""
        
        if self.chaff_count >= count:
            self.chaff_count -= count
            print(f"[EW] Deployed {count} chaff, {self.chaff_count} remaining")
            return True
        else:
            print(f"[EW] Insufficient chaff ({self.chaff_count} remaining)")
            return False
    
    def deploy_flares(self, count: int = 1) -> bool:
        """Deploy flare countermeasures"""
        
        if self.flare_count >= count:
            self.flare_count -= count
            print(f"[EW] Deployed {count} flares, {self.flare_count} remaining")
            return True
        else:
            print(f"[EW] Insufficient flares ({self.flare_count} remaining)")
            return False
    
    def activate_jammer(self, target_frequency: float = 9.5) -> bool:
        """Activate ECM jammer"""
        
        if (self.jammer_frequency_range[0] <= target_frequency <= 
            self.jammer_frequency_range[1]):
            self.jammer_active = True
            print(f"[EW] Jammer active on {target_frequency} GHz")
            return True
        else:
            print(f"[EW] Cannot jam {target_frequency} GHz (out of range)")
            return False
    
    def get_ew_state(self) -> Dict[str, Any]:
        """Get EW suite state"""
        return {
            'chaff_count': self.chaff_count,
            'flare_count': self.flare_count,
            'chaff_percentage': self.chaff_count / self.max_chaff,
            'flare_percentage': self.flare_count / self.max_flares,
            'jammer_active': self.jammer_active,
            'jammer_frequency': self.jammer_frequency_range
        }


class AdvancedSensorSuite:
    """
    Complete sensor suite combining radar, RWR, and EW systems
    for realistic air combat simulation.
    """
    
    def __init__(self, aircraft_type: str = "F-16C"):
        """
        Initialize complete sensor suite
        
        Args:
            aircraft_type: Aircraft type (determines sensor specifications)
        """
        self.aircraft_type = aircraft_type
        
        # Initialize subsystems
        self.radar = RadarSystem(aircraft_type)
        self.rwr = RWRSystem(aircraft_type)
        self.ew_suite = ElectronicWarfareSuite(aircraft_type)
        
        # Sensor fusion
        self.fused_picture = {}
        self.threat_assessment = {}
        
        print(f"[SENSOR SUITE] Advanced sensor suite initialized for {aircraft_type}")
    
    def update_sensors(self, own_position: np.ndarray, own_attitude: np.ndarray,
                      targets: List[Dict[str, Any]], emitters: List[Dict[str, Any]],
                      dt: float = 1.0) -> Dict[str, Any]:
        """
        Update all sensors and provide fused tactical picture
        
        Args:
            own_position: Own aircraft position [x, y, z]
            own_attitude: Own aircraft attitude [pitch, roll, yaw]
            targets: List of potential targets
            emitters: List of radar emitters
            dt: Time delta
        
        Returns:
            Fused sensor picture
        """
        
        # Update individual sensors
        radar_contacts = self.radar.update(own_position, own_attitude, targets, dt)
        rwr_contacts = self.rwr.update(own_position, emitters)
        
        # Sensor fusion
        fused_picture = self._fuse_sensor_data(radar_contacts, rwr_contacts)
        
        # Threat assessment
        threat_assessment = self._assess_threats(fused_picture)
        
        # Create comprehensive sensor state
        sensor_state = {
            'radar_state': self.radar.get_radar_state(),
            'rwr_contacts': len(rwr_contacts),
            'ew_state': self.ew_suite.get_ew_state(),
            'fused_contacts': len(fused_picture),
            'highest_threat_level': max([c.get('threat_level', 0) for c in fused_picture.values()] + [0]),
            'threats_within_engagement_range': sum(1 for c in fused_picture.values() 
                                                  if c.get('range', float('inf')) < 50000),
            'missile_warning': any(contact.missile_launch_warning for contact in rwr_contacts)
        }
        
        self.fused_picture = fused_picture
        self.threat_assessment = threat_assessment
        
        return sensor_state
    
    def _fuse_sensor_data(self, radar_contacts: List[RadarContact],
                         rwr_contacts: List[RWRContact]) -> Dict[str, Dict[str, Any]]:
        """Fuse radar and RWR data into coherent picture"""
        
        fused_contacts = {}
        
        # Add radar contacts
        for contact in radar_contacts:
            fused_contacts[contact.contact_id] = {
                'id': contact.contact_id,
                'source': 'radar',
                'range': contact.range,
                'azimuth': contact.azimuth,
                'elevation': contact.elevation,
                'range_rate': contact.range_rate,
                'threat_type': contact.threat_type.value,
                'confidence': contact.confidence,
                'locked': contact.locked,
                'track_quality': contact.track_quality
            }
        
        # Correlate RWR contacts with radar contacts
        for rwr_contact in rwr_contacts:
            # Simple correlation based on azimuth (real systems use complex algorithms)
            correlated = False
            
            for contact_id, radar_data in fused_contacts.items():
                azimuth_diff = abs(radar_data['azimuth'] - rwr_contact.azimuth)
                if azimuth_diff < 10.0:  # Within 10 degrees
                    # Correlate with radar contact
                    radar_data['rwr_correlation'] = True
                    radar_data['threat_level'] = rwr_contact.threat_level
                    radar_data['lock_indication'] = rwr_contact.lock_indication
                    radar_data['missile_warning'] = rwr_contact.missile_launch_warning
                    correlated = True
                    break
            
            # Add uncorrelated RWR contact
            if not correlated:
                fused_contacts[rwr_contact.emitter_id] = {
                    'id': rwr_contact.emitter_id,
                    'source': 'rwr_only',
                    'azimuth': rwr_contact.azimuth,
                    'threat_type': rwr_contact.threat_type.value,
                    'threat_level': rwr_contact.threat_level,
                    'lock_indication': rwr_contact.lock_indication,
                    'missile_warning': rwr_contact.missile_launch_warning,
                    'frequency': rwr_contact.frequency
                }
        
        return fused_contacts
    
    def _assess_threats(self, fused_picture: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall threat situation"""
        
        if not fused_picture:
            return {'overall_threat': 0, 'immediate_threats': 0, 'primary_threat': None}
        
        # Analyze threats
        threat_levels = [contact.get('threat_level', 0) for contact in fused_picture.values()]
        immediate_threats = sum(1 for level in threat_levels if level >= 4)
        
        # Find primary threat (highest threat level, closest range)
        primary_threat = None
        max_threat_score = 0
        
        for contact_id, contact in fused_picture.items():
            threat_level = contact.get('threat_level', 0)
            range_m = contact.get('range', float('inf'))
            
            # Threat score combines level and proximity
            threat_score = threat_level + (1.0 - min(range_m / 100000, 1.0))
            
            if threat_score > max_threat_score:
                max_threat_score = threat_score
                primary_threat = contact_id
        
        return {
            'overall_threat': max(threat_levels) if threat_levels else 0,
            'immediate_threats': immediate_threats,
            'primary_threat': primary_threat,
            'threat_count': len(fused_picture),
            'max_threat_score': max_threat_score
        }
    
    def get_sensor_observation_vector(self) -> np.ndarray:
        """
        Get sensor data as observation vector for RL agent
        
        Returns:
            Sensor observation vector for integration with environment state
        """
        
        # Create sensor observation vector (12 dimensions)
        sensor_obs = np.zeros(12, dtype=np.float32)
        
        radar_state = self.radar.get_radar_state()
        ew_state = self.ew_suite.get_ew_state()
        
        # Radar information [0-5]
        sensor_obs[0] = 1.0 if radar_state['mode'] != 'off' else 0.0
        sensor_obs[1] = min(1.0, radar_state['contacts_count'] / 10.0)  # Normalized contact count
        sensor_obs[2] = 1.0 if radar_state['locked_target'] else 0.0
        sensor_obs[3] = radar_state.get('locked_range', 0) / 100000.0  # Normalized range
        sensor_obs[4] = (radar_state.get('locked_azimuth', 0) + 180) / 360.0  # Normalized azimuth
        sensor_obs[5] = radar_state.get('locked_track_quality', 0)
        
        # RWR information [6-9]
        rwr_contact_count = len(getattr(self.rwr, 'rwr_contacts', {}))
        sensor_obs[6] = min(1.0, rwr_contact_count / 5.0)  # Normalized RWR contact count
        sensor_obs[7] = self.threat_assessment.get('overall_threat', 0) / 5.0  # Normalized threat level
        sensor_obs[8] = min(1.0, self.threat_assessment.get('immediate_threats', 0) / 3.0)
        sensor_obs[9] = 1.0 if any(c.get('missile_warning', False) for c in self.fused_picture.values()) else 0.0
        
        # EW suite information [10-11]
        sensor_obs[10] = ew_state['chaff_percentage']
        sensor_obs[11] = ew_state['flare_percentage']
        
        return sensor_obs
    
    def get_tactical_recommendations(self) -> Dict[str, str]:
        """Get sensor-based tactical recommendations"""
        
        recommendations = {
            'radar_recommendation': 'continue_current_mode',
            'threat_response': 'monitor',
            'countermeasure_recommendation': 'none',
            'primary_action': 'continue'
        }
        
        # Analyze current situation
        if self.threat_assessment.get('immediate_threats', 0) > 0:
            recommendations['threat_response'] = 'defensive_action'
            recommendations['primary_action'] = 'evade'
        
        if any(c.get('missile_warning', False) for c in self.fused_picture.values()):
            recommendations['countermeasure_recommendation'] = 'deploy_chaff_flares'
            recommendations['primary_action'] = 'notch_and_chaff'
        
        if self.radar.locked_target and len(self.fused_picture) == 1:
            recommendations['radar_recommendation'] = 'maintain_lock'
            recommendations['primary_action'] = 'engage_target'
        
        return recommendations


def integrate_sensors_with_environment(env, sensor_suite: AdvancedSensorSuite) -> np.ndarray:
    """
    Integrate sensor suite with Harfang environment to create enhanced observation space
    
    Args:
        env: Harfang environment
        sensor_suite: Advanced sensor suite
    
    Returns:
        Enhanced observation vector (25 + 12 = 37 dimensions)
    """
    
    # Get base environment observation (25D)
    base_obs = env._get_observation() if hasattr(env, '_get_observation') else np.zeros(25)
    
    # Get sensor observation (12D)
    sensor_obs = sensor_suite.get_sensor_observation_vector()
    
    # Combine into enhanced observation
    enhanced_obs = np.concatenate([base_obs, sensor_obs])
    
    return enhanced_obs.astype(np.float32)


def create_sensor_enhanced_environment(base_env_class, aircraft_type: str = "F-16C"):
    """
    Factory function to create sensor-enhanced environment wrapper
    
    Args:
        base_env_class: Base environment class to wrap
        aircraft_type: Aircraft type for sensor specifications
    
    Returns:
        Enhanced environment class with sensor integration
    """
    
    class SensorEnhancedEnvironment(base_env_class):
        """Environment wrapper with advanced sensor simulation"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Add sensor suite
            self.sensor_suite = AdvancedSensorSuite(aircraft_type)
            
            # Expand observation space to include sensors (25 + 12 = 37D)
            import gymnasium as gym
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32
            )
            
            print(f"[SENSOR ENV] Enhanced with {aircraft_type} sensor suite")
            print(f"[SENSOR ENV] Observation space expanded to {self.observation_space.shape}")
        
        def reset(self, **kwargs):
            """Reset with sensor initialization"""
            base_result = super().reset(**kwargs)
            
            # Reset sensors
            self.sensor_suite.radar.set_mode(RadarMode.SEARCH)
            
            # Get enhanced observation
            if isinstance(base_result, tuple):
                base_obs, info = base_result
                enhanced_obs = self._get_enhanced_observation()
                return enhanced_obs, info
            else:
                return self._get_enhanced_observation()
        
        def step(self, action):
            """Step with sensor updates"""
            
            # Handle sensor control actions (if extended action space)
            sensor_actions = self._extract_sensor_actions(action)
            self._apply_sensor_actions(sensor_actions)
            
            # Step base environment
            result = super().step(action)
            obs, reward, terminated, truncated, info = result
            
            # Update sensors
            self._update_sensors_from_env_state(obs, info)
            
            # Get enhanced observation
            enhanced_obs = self._get_enhanced_observation()
            
            return enhanced_obs, reward, terminated, truncated, info
        
        def _get_enhanced_observation(self) -> np.ndarray:
            """Get observation with sensor data"""
            base_obs = super()._get_observation() if hasattr(super(), '_get_observation') else np.zeros(25)
            sensor_obs = self.sensor_suite.get_sensor_observation_vector()
            return np.concatenate([base_obs, sensor_obs]).astype(np.float32)
        
        def _extract_sensor_actions(self, action: np.ndarray) -> Dict[str, Any]:
            """Extract sensor control actions from extended action space"""
            # For now, use simple heuristics
            # Could be extended to include radar mode, countermeasure deployment
            return {
                'radar_mode': 'auto',
                'deploy_countermeasures': action[3] > 0.8 if len(action) > 3 else False
            }
        
        def _apply_sensor_actions(self, sensor_actions: Dict[str, Any]):
            """Apply sensor control actions"""
            
            # Auto radar mode management
            if sensor_actions['radar_mode'] == 'auto':
                contacts = len(self.sensor_suite.radar.contacts)
                if contacts == 0:
                    self.sensor_suite.radar.set_mode(RadarMode.SEARCH)
                elif contacts == 1 and not self.sensor_suite.radar.locked_target:
                    # Try to lock single contact
                    contact_id = list(self.sensor_suite.radar.contacts.keys())[0]
                    self.sensor_suite.radar.lock_target(contact_id)
                elif contacts > 1:
                    self.sensor_suite.radar.set_mode(RadarMode.TWS)
            
            # Countermeasure deployment
            if sensor_actions['deploy_countermeasures']:
                # Deploy both chaff and flares
                self.sensor_suite.ew_suite.deploy_chaff(2)
                self.sensor_suite.ew_suite.deploy_flares(1)
        
        def _update_sensors_from_env_state(self, obs: np.ndarray, info: Dict[str, Any]):
            """Update sensors based on environment state"""
            
            # Create mock targets for sensor simulation
            targets = []
            if info.get('enemy_health', 0) > 0:
                # Create enemy target
                enemy_distance = info.get('distance', 10000)
                enemy_azimuth = random.uniform(-30, 30)  # Simplified
                
                targets.append({
                    'id': 'enemy_1',
                    'position': [enemy_distance * math.cos(math.radians(enemy_azimuth)),
                               enemy_distance * math.sin(math.radians(enemy_azimuth)),
                               0],
                    'type': 'fighter',
                    'rcs': 3.0,  # m²
                    'closure_rate': random.uniform(-200, 200)
                })
            
            # Create mock emitters
            emitters = []
            if targets:
                for target in targets:
                    emitters.append({
                        'id': f"radar_{target['id']}",
                        'position': target['position'],
                        'frequency': 9.0 + random.uniform(-0.5, 0.5),
                        'power': 1000,
                        'platform': 'fighter',
                        'tracking_mode': info.get('locked', False)
                    })
            
            # Update sensor suite
            own_pos = np.array([0, 0, 5000])  # Simplified own position
            own_att = np.array([0, 0, 0])    # Simplified own attitude
            
            self.sensor_suite.update_sensors(own_pos, own_att, targets, emitters)
    
    return SensorEnhancedEnvironment


if __name__ == "__main__":
    # Test sensor suite
    print("Advanced Sensor Suite for Realistic Air Combat")
    
    # Create sensor suite
    sensors = AdvancedSensorSuite("F-16C")
    
    # Simulate sensor update
    own_pos = np.array([0, 0, 5000])
    own_att = np.array([0, 0, 0])
    
    # Mock targets and emitters
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
    
    # Update sensors
    sensor_state = sensors.update_sensors(own_pos, own_att, targets, emitters)
    
    print(f"\nSensor State: {sensor_state}")
    print(f"Observation Vector Shape: {sensors.get_sensor_observation_vector().shape}")
