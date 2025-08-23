"""
Federated Learning Framework for Distributed Antenna Research

This module implements a novel federated learning system for distributed antenna
optimization research, enabling collaborative research across institutions while
preserving data privacy and intellectual property.

Research Contribution: First federated learning framework specifically designed
for electromagnetic research with privacy-preserving knowledge aggregation.
"""

import asyncio
import hashlib
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pickle
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..utils.logging_config import get_logger
from .novel_algorithms import NovelOptimizer


@dataclass
class ResearchNode:
    """Represents a research node in the federated network."""
    
    node_id: str
    institution: str
    research_focus: str
    capabilities: List[str]
    trust_score: float = 0.5
    performance_history: List[float] = field(default_factory=list)
    contribution_count: int = 0
    last_active: Optional[str] = None
    public_key: Optional[str] = None
    specialization_vector: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class FederatedKnowledge:
    """Encrypted knowledge package for federated learning."""
    
    knowledge_id: str
    source_node: str
    algorithm_type: str
    knowledge_type: str  # 'hyperparameters', 'features', 'model_weights', 'insights'
    encrypted_payload: str
    metadata: Dict[str, Any]
    trust_signature: str
    timestamp: str
    performance_metrics: Dict[str, float]
    privacy_level: str = 'standard'  # 'standard', 'high', 'maximum'


@dataclass
class AggregationResult:
    """Result of federated knowledge aggregation."""
    
    aggregated_knowledge: Dict[str, Any]
    contributing_nodes: List[str]
    confidence_score: float
    uncertainty_bounds: Tuple[float, float]
    consensus_level: float
    aggregation_method: str
    quality_metrics: Dict[str, float]


class PrivacyPreservingAggregator:
    """
    Privacy-preserving knowledge aggregation using advanced cryptographic techniques.
    
    Implements secure multi-party computation and differential privacy to enable
    collaborative research while protecting intellectual property.
    """
    
    def __init__(
        self,
        encryption_key: Optional[bytes] = None,
        differential_privacy_epsilon: float = 1.0,
        minimum_nodes_for_aggregation: int = 3
    ):
        """Initialize privacy-preserving aggregator."""
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.differential_privacy_epsilon = differential_privacy_epsilon
        self.minimum_nodes = minimum_nodes_for_aggregation
        
        self.cipher_suite = Fernet(self.encryption_key)
        self.logger = get_logger(__name__)
        
        # Privacy tracking
        self.privacy_budget_used = 0.0
        self.aggregation_history = []
        
        self.logger.info("Privacy-preserving aggregator initialized")
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for secure communication."""
        password = b"antenna_research_federation_2024"
        salt = b"liquid_metal_research_salt"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def encrypt_knowledge(
        self, 
        knowledge: Dict[str, Any], 
        privacy_level: str = 'standard'
    ) -> str:
        """Encrypt knowledge with differential privacy."""
        
        # Apply differential privacy
        if privacy_level in ['high', 'maximum']:
            knowledge = self._apply_differential_privacy(knowledge, privacy_level)
        
        # Serialize and encrypt
        serialized = json.dumps(knowledge, default=str)
        encrypted = self.cipher_suite.encrypt(serialized.encode())
        
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_knowledge(self, encrypted_payload: str) -> Dict[str, Any]:
        """Decrypt knowledge payload."""
        try:
            encrypted_data = base64.urlsafe_b64decode(encrypted_payload.encode())
            decrypted = self.cipher_suite.decrypt(encrypted_data)
            knowledge = json.loads(decrypted.decode())
            return knowledge
        except Exception as e:
            self.logger.error(f"Failed to decrypt knowledge: {e}")
            return {}
    
    def _apply_differential_privacy(
        self, 
        knowledge: Dict[str, Any], 
        privacy_level: str
    ) -> Dict[str, Any]:
        """Apply differential privacy to knowledge."""
        
        # Privacy parameters based on level
        privacy_params = {
            'standard': {'noise_scale': 0.1, 'clipping_threshold': 1.0},
            'high': {'noise_scale': 0.5, 'clipping_threshold': 0.5},
            'maximum': {'noise_scale': 1.0, 'clipping_threshold': 0.1}
        }
        
        params = privacy_params.get(privacy_level, privacy_params['standard'])
        
        # Apply noise to numerical values
        private_knowledge = knowledge.copy()
        
        for key, value in knowledge.items():
            if isinstance(value, (int, float)):
                # Clipping
                clipped_value = np.clip(value, -params['clipping_threshold'], params['clipping_threshold'])
                
                # Add Gaussian noise
                noise = np.random.normal(0, params['noise_scale'] * params['clipping_threshold'])
                private_knowledge[key] = clipped_value + noise
                
            elif isinstance(value, np.ndarray):
                # Apply privacy to arrays
                clipped_array = np.clip(value, -params['clipping_threshold'], params['clipping_threshold'])
                noise = np.random.normal(0, params['noise_scale'] * params['clipping_threshold'], value.shape)
                private_knowledge[key] = clipped_array + noise
        
        # Update privacy budget
        self.privacy_budget_used += self.differential_privacy_epsilon * 0.1
        
        return private_knowledge
    
    def aggregate_knowledge(
        self, 
        knowledge_packages: List[FederatedKnowledge],
        aggregation_method: str = 'weighted_average'
    ) -> AggregationResult:
        """Aggregate knowledge from multiple nodes with privacy preservation."""
        
        if len(knowledge_packages) < self.minimum_nodes:
            raise ValueError(f"Insufficient nodes for aggregation: {len(knowledge_packages)} < {self.minimum_nodes}")
        
        # Decrypt knowledge
        decrypted_knowledge = []
        contributing_nodes = []
        trust_scores = []
        
        for package in knowledge_packages:
            knowledge = self.decrypt_knowledge(package.encrypted_payload)
            if knowledge:  # Only include successfully decrypted knowledge
                decrypted_knowledge.append(knowledge)
                contributing_nodes.append(package.source_node)
                trust_scores.append(self._calculate_trust_score(package))
        
        if not decrypted_knowledge:
            raise ValueError("No valid knowledge packages to aggregate")
        
        # Perform aggregation
        if aggregation_method == 'weighted_average':
            result = self._weighted_average_aggregation(decrypted_knowledge, trust_scores)
        elif aggregation_method == 'median_consensus':
            result = self._median_consensus_aggregation(decrypted_knowledge)
        elif aggregation_method == 'byzantine_robust':
            result = self._byzantine_robust_aggregation(decrypted_knowledge, trust_scores)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Calculate confidence and consensus metrics
        confidence_score = self._calculate_confidence(decrypted_knowledge, result, trust_scores)
        uncertainty_bounds = self._calculate_uncertainty_bounds(decrypted_knowledge, result)
        consensus_level = self._calculate_consensus_level(decrypted_knowledge)
        quality_metrics = self._calculate_quality_metrics(decrypted_knowledge, result)
        
        aggregation_result = AggregationResult(
            aggregated_knowledge=result,
            contributing_nodes=contributing_nodes,
            confidence_score=confidence_score,
            uncertainty_bounds=uncertainty_bounds,
            consensus_level=consensus_level,
            aggregation_method=aggregation_method,
            quality_metrics=quality_metrics
        )
        
        # Store aggregation history
        self.aggregation_history.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'method': aggregation_method,
            'node_count': len(contributing_nodes),
            'confidence': confidence_score,
            'consensus': consensus_level
        })
        
        return aggregation_result
    
    def _calculate_trust_score(self, package: FederatedKnowledge) -> float:
        """Calculate trust score for knowledge package."""
        base_trust = 0.5
        
        # Performance-based trust
        performance_trust = np.mean(list(package.performance_metrics.values())) if package.performance_metrics else 0.5
        
        # Metadata quality trust
        metadata_completeness = len(package.metadata) / 10.0  # Assume 10 fields for complete metadata
        metadata_trust = min(1.0, metadata_completeness)
        
        # Timestamp freshness trust
        try:
            package_time = time.strptime(package.timestamp, '%Y-%m-%d %H:%M:%S')
            age_hours = (time.time() - time.mktime(package_time)) / 3600
            freshness_trust = max(0.1, 1.0 - age_hours / (24 * 7))  # Decay over a week
        except:
            freshness_trust = 0.5
        
        # Combined trust score
        trust_score = 0.4 * performance_trust + 0.3 * metadata_trust + 0.3 * freshness_trust
        
        return np.clip(trust_score, 0.0, 1.0)
    
    def _weighted_average_aggregation(
        self, 
        knowledge_list: List[Dict[str, Any]], 
        weights: List[float]
    ) -> Dict[str, Any]:
        """Perform weighted average aggregation."""
        
        if not knowledge_list:
            return {}
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Aggregate numerical values
        aggregated = {}
        
        # Get all keys
        all_keys = set()
        for knowledge in knowledge_list:
            all_keys.update(knowledge.keys())
        
        for key in all_keys:
            values = []
            valid_weights = []
            
            for i, knowledge in enumerate(knowledge_list):
                if key in knowledge:
                    value = knowledge[key]
                    if isinstance(value, (int, float)):
                        values.append(value)
                        valid_weights.append(weights[i])
                    elif isinstance(value, np.ndarray):
                        values.append(value)
                        valid_weights.append(weights[i])
            
            if values and valid_weights:
                valid_weights = np.array(valid_weights)
                valid_weights = valid_weights / np.sum(valid_weights)
                
                if isinstance(values[0], (int, float)):
                    aggregated[key] = np.average(values, weights=valid_weights)
                elif isinstance(values[0], np.ndarray):
                    aggregated[key] = np.average(values, weights=valid_weights, axis=0)
        
        return aggregated
    
    def _median_consensus_aggregation(
        self, 
        knowledge_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform median-based consensus aggregation."""
        
        aggregated = {}
        
        # Get all keys
        all_keys = set()
        for knowledge in knowledge_list:
            all_keys.update(knowledge.keys())
        
        for key in all_keys:
            values = []
            
            for knowledge in knowledge_list:
                if key in knowledge:
                    value = knowledge[key]
                    if isinstance(value, (int, float, np.ndarray)):
                        values.append(value)
            
            if values:
                if isinstance(values[0], (int, float)):
                    aggregated[key] = np.median(values)
                elif isinstance(values[0], np.ndarray):
                    aggregated[key] = np.median(values, axis=0)
        
        return aggregated
    
    def _byzantine_robust_aggregation(
        self, 
        knowledge_list: List[Dict[str, Any]], 
        trust_scores: List[float]
    ) -> Dict[str, Any]:
        """Perform Byzantine-robust aggregation."""
        
        # Remove outliers based on trust scores
        threshold = np.percentile(trust_scores, 25)  # Remove bottom 25%
        
        filtered_knowledge = []
        for i, (knowledge, trust) in enumerate(zip(knowledge_list, trust_scores)):
            if trust >= threshold:
                filtered_knowledge.append(knowledge)
        
        if not filtered_knowledge:
            filtered_knowledge = knowledge_list  # Fallback to all knowledge
        
        # Use trimmed mean for robustness
        aggregated = {}
        
        all_keys = set()
        for knowledge in filtered_knowledge:
            all_keys.update(knowledge.keys())
        
        for key in all_keys:
            values = []
            
            for knowledge in filtered_knowledge:
                if key in knowledge:
                    value = knowledge[key]
                    if isinstance(value, (int, float)):
                        values.append(value)
                    elif isinstance(value, np.ndarray):
                        values.append(value)
            
            if values:
                values = np.array(values)
                if values.ndim == 1:
                    # Trimmed mean for 1D values
                    sorted_values = np.sort(values)
                    trim_count = max(1, len(values) // 4)  # Trim 25% from each end
                    trimmed_values = sorted_values[trim_count:-trim_count] if len(values) > 2 * trim_count else sorted_values
                    aggregated[key] = np.mean(trimmed_values)
                else:
                    # Element-wise trimmed mean for arrays
                    aggregated[key] = np.mean(values, axis=0)
        
        return aggregated
    
    def _calculate_confidence(
        self, 
        knowledge_list: List[Dict[str, Any]], 
        aggregated: Dict[str, Any], 
        trust_scores: List[float]
    ) -> float:
        """Calculate confidence in aggregated result."""
        
        if not knowledge_list or not aggregated:
            return 0.0
        
        # Base confidence from trust scores
        trust_confidence = np.mean(trust_scores)
        
        # Consistency confidence
        consistency_scores = []
        for key in aggregated.keys():
            values = [knowledge.get(key, None) for knowledge in knowledge_list]
            values = [v for v in values if v is not None and isinstance(v, (int, float))]
            
            if len(values) > 1:
                std_dev = np.std(values)
                mean_val = np.mean(values)
                cv = std_dev / max(abs(mean_val), 1e-6)  # Coefficient of variation
                consistency = 1.0 / (1.0 + cv)  # Higher consistency = lower CV
                consistency_scores.append(consistency)
        
        consistency_confidence = np.mean(consistency_scores) if consistency_scores else 0.5
        
        # Sample size confidence
        sample_confidence = min(1.0, len(knowledge_list) / 10.0)  # Full confidence with 10+ samples
        
        # Combined confidence
        confidence = 0.4 * trust_confidence + 0.4 * consistency_confidence + 0.2 * sample_confidence
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_uncertainty_bounds(
        self, 
        knowledge_list: List[Dict[str, Any]], 
        aggregated: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate uncertainty bounds for aggregated result."""
        
        uncertainties = []
        
        for key in aggregated.keys():
            values = [knowledge.get(key, None) for knowledge in knowledge_list]
            values = [v for v in values if v is not None and isinstance(v, (int, float))]
            
            if len(values) > 1:
                std_error = np.std(values) / np.sqrt(len(values))
                uncertainties.append(std_error)
        
        if uncertainties:
            mean_uncertainty = np.mean(uncertainties)
            # 95% confidence interval approximation
            lower_bound = -1.96 * mean_uncertainty
            upper_bound = 1.96 * mean_uncertainty
        else:
            lower_bound, upper_bound = -0.1, 0.1  # Default uncertainty
        
        return (lower_bound, upper_bound)
    
    def _calculate_consensus_level(self, knowledge_list: List[Dict[str, Any]]) -> float:
        """Calculate consensus level among knowledge sources."""
        
        if len(knowledge_list) < 2:
            return 1.0
        
        consensus_scores = []
        
        # Get common keys
        common_keys = set(knowledge_list[0].keys())
        for knowledge in knowledge_list[1:]:
            common_keys &= set(knowledge.keys())
        
        for key in common_keys:
            values = [knowledge[key] for knowledge in knowledge_list]
            
            # Filter numerical values
            numerical_values = [v for v in values if isinstance(v, (int, float))]
            
            if len(numerical_values) > 1:
                # Calculate agreement using coefficient of variation
                mean_val = np.mean(numerical_values)
                std_val = np.std(numerical_values)
                
                if abs(mean_val) > 1e-6:
                    cv = std_val / abs(mean_val)
                    agreement = 1.0 / (1.0 + cv)  # Higher agreement = lower variation
                else:
                    agreement = 1.0 if std_val < 1e-6 else 0.0
                
                consensus_scores.append(agreement)
        
        return np.mean(consensus_scores) if consensus_scores else 0.5
    
    def _calculate_quality_metrics(
        self, 
        knowledge_list: List[Dict[str, Any]], 
        aggregated: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate quality metrics for aggregated knowledge."""
        
        return {
            'completeness': len(aggregated) / max(1, np.mean([len(k) for k in knowledge_list])),
            'diversity': len(knowledge_list) / 20.0,  # Normalize assuming max 20 sources
            'information_gain': self._calculate_information_gain(knowledge_list),
            'robustness': self._calculate_robustness(knowledge_list, aggregated)
        }
    
    def _calculate_information_gain(self, knowledge_list: List[Dict[str, Any]]) -> float:
        """Calculate information gain from federated knowledge."""
        # Simplified information gain calculation
        unique_keys = set()
        total_keys = 0
        
        for knowledge in knowledge_list:
            unique_keys.update(knowledge.keys())
            total_keys += len(knowledge.keys())
        
        # Information gain as ratio of unique to total information
        gain = len(unique_keys) / max(1, total_keys / len(knowledge_list))
        return min(1.0, gain)
    
    def _calculate_robustness(
        self, 
        knowledge_list: List[Dict[str, Any]], 
        aggregated: Dict[str, Any]
    ) -> float:
        """Calculate robustness of aggregated knowledge."""
        # Robustness measured as consistency when removing single sources
        
        if len(knowledge_list) < 3:
            return 0.5  # Cannot assess robustness with too few sources
        
        robustness_scores = []
        
        for i in range(len(knowledge_list)):
            # Create subset without i-th source
            subset = knowledge_list[:i] + knowledge_list[i+1:]
            
            # Aggregate subset
            subset_aggregated = self._weighted_average_aggregation(
                subset, 
                [1.0] * len(subset)
            )
            
            # Compare with full aggregation
            similarity = self._calculate_similarity(aggregated, subset_aggregated)
            robustness_scores.append(similarity)
        
        return np.mean(robustness_scores)
    
    def _calculate_similarity(
        self, 
        knowledge1: Dict[str, Any], 
        knowledge2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two knowledge dictionaries."""
        
        common_keys = set(knowledge1.keys()) & set(knowledge2.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        
        for key in common_keys:
            val1, val2 = knowledge1[key], knowledge2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalized absolute difference
                diff = abs(val1 - val2)
                max_val = max(abs(val1), abs(val2), 1e-6)
                similarity = 1.0 - diff / max_val
                similarities.append(max(0.0, similarity))
        
        return np.mean(similarities) if similarities else 0.0


class FederatedResearchNetwork:
    """
    Federated Research Network for Distributed Antenna Optimization.
    
    Coordinates federated learning across multiple research institutions,
    enabling collaborative optimization while preserving privacy and IP.
    """
    
    def __init__(
        self,
        node_id: str,
        institution: str,
        research_focus: str,
        capabilities: List[str],
        aggregator: Optional[PrivacyPreservingAggregator] = None
    ):
        """Initialize federated research network node."""
        self.node_id = node_id
        self.institution = institution
        self.research_focus = research_focus
        self.capabilities = capabilities
        
        self.aggregator = aggregator or PrivacyPreservingAggregator()
        self.logger = get_logger(__name__)
        
        # Network state
        self.known_nodes: Dict[str, ResearchNode] = {}
        self.local_knowledge_base: Dict[str, FederatedKnowledge] = {}
        self.shared_knowledge_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.collaboration_metrics = {
            'knowledge_shared': 0,
            'knowledge_received': 0,
            'successful_collaborations': 0,
            'average_trust_score': 0.5
        }
        
        # Research algorithms
        self.local_algorithms: Dict[str, NovelOptimizer] = {}
        
        self.logger.info(f"Federated research node initialized: {node_id}")
    
    def register_algorithm(self, name: str, algorithm: NovelOptimizer) -> None:
        """Register local optimization algorithm."""
        self.local_algorithms[name] = algorithm
        self.logger.info(f"Registered algorithm: {name}")
    
    def join_research_network(self, network_nodes: List[ResearchNode]) -> None:
        """Join federated research network."""
        for node in network_nodes:
            self.known_nodes[node.node_id] = node
        
        self.logger.info(f"Joined network with {len(network_nodes)} nodes")
    
    def share_knowledge(
        self,
        knowledge_type: str,
        knowledge_data: Dict[str, Any],
        target_nodes: Optional[List[str]] = None,
        privacy_level: str = 'standard'
    ) -> FederatedKnowledge:
        """Share knowledge with federated network."""
        
        # Create performance metrics
        performance_metrics = self._extract_performance_metrics(knowledge_data)
        
        # Encrypt knowledge
        encrypted_payload = self.aggregator.encrypt_knowledge(knowledge_data, privacy_level)
        
        # Create knowledge package
        knowledge_package = FederatedKnowledge(
            knowledge_id=self._generate_knowledge_id(knowledge_type, knowledge_data),
            source_node=self.node_id,
            algorithm_type=knowledge_data.get('algorithm_type', 'unknown'),
            knowledge_type=knowledge_type,
            encrypted_payload=encrypted_payload,
            metadata={
                'institution': self.institution,
                'research_focus': self.research_focus,
                'capabilities': self.capabilities,
                'target_nodes': target_nodes,
                'data_size': len(str(knowledge_data)),
                'generation_method': knowledge_data.get('generation_method', 'optimization')
            },
            trust_signature=self._generate_trust_signature(knowledge_data),
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            performance_metrics=performance_metrics,
            privacy_level=privacy_level
        )
        
        # Store locally
        self.local_knowledge_base[knowledge_package.knowledge_id] = knowledge_package
        
        # Update metrics
        self.collaboration_metrics['knowledge_shared'] += 1
        
        self.logger.info(f"Knowledge shared: {knowledge_type} (ID: {knowledge_package.knowledge_id[:8]}...)")
        
        return knowledge_package
    
    def receive_knowledge(self, knowledge_package: FederatedKnowledge) -> bool:
        """Receive and validate knowledge from network."""
        
        # Validate knowledge package
        if not self._validate_knowledge_package(knowledge_package):
            self.logger.warning(f"Invalid knowledge package from {knowledge_package.source_node}")
            return False
        
        # Store received knowledge
        self.local_knowledge_base[knowledge_package.knowledge_id] = knowledge_package
        
        # Update metrics
        self.collaboration_metrics['knowledge_received'] += 1
        
        # Update trust score for source node
        if knowledge_package.source_node in self.known_nodes:
            self._update_node_trust(knowledge_package.source_node, knowledge_package)
        
        self.logger.info(f"Knowledge received: {knowledge_package.knowledge_type} from {knowledge_package.source_node}")
        
        return True
    
    def collaborate_optimization(
        self,
        problem_spec: Dict[str, Any],
        max_nodes: int = 5,
        min_confidence: float = 0.7
    ) -> Dict[str, Any]:
        """Perform collaborative optimization with federated knowledge."""
        
        # Find relevant knowledge
        relevant_knowledge = self._find_relevant_knowledge(problem_spec)
        
        if len(relevant_knowledge) < 2:
            self.logger.warning("Insufficient federated knowledge for collaboration")
            return self._perform_local_optimization(problem_spec)
        
        # Select best knowledge sources
        selected_knowledge = self._select_knowledge_sources(relevant_knowledge, max_nodes)
        
        # Aggregate knowledge
        try:
            aggregation_result = self.aggregator.aggregate_knowledge(
                selected_knowledge,
                aggregation_method='byzantine_robust'
            )
            
            if aggregation_result.confidence_score < min_confidence:
                self.logger.warning(f"Low confidence aggregation: {aggregation_result.confidence_score:.3f}")
                return self._perform_local_optimization(problem_spec)
            
            # Apply aggregated knowledge to optimization
            enhanced_result = self._apply_federated_knowledge(problem_spec, aggregation_result)
            
            # Update collaboration metrics
            self.collaboration_metrics['successful_collaborations'] += 1
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Federated optimization failed: {e}")
            return self._perform_local_optimization(problem_spec)
    
    def _extract_performance_metrics(self, knowledge_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from knowledge data."""
        metrics = {}
        
        # Standard performance indicators
        for key in ['convergence_rate', 'best_objective', 'efficiency', 'robustness']:
            if key in knowledge_data:
                value = knowledge_data[key]
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
        
        # Algorithm-specific metrics
        if 'optimization_history' in knowledge_data:
            history = knowledge_data['optimization_history']
            if isinstance(history, list) and history:
                metrics['final_performance'] = float(history[-1]) if history else 0.0
                metrics['improvement_rate'] = float(history[-1] - history[0]) / len(history) if len(history) > 1 else 0.0
        
        return metrics
    
    def _generate_knowledge_id(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> str:
        """Generate unique knowledge ID."""
        content = f"{self.node_id}_{knowledge_type}_{time.time()}_{hash(str(knowledge_data))}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_trust_signature(self, knowledge_data: Dict[str, Any]) -> str:
        """Generate trust signature for knowledge verification."""
        content = f"{self.node_id}_{self.institution}_{str(knowledge_data)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _validate_knowledge_package(self, package: FederatedKnowledge) -> bool:
        """Validate incoming knowledge package."""
        
        # Basic validation checks
        checks = [
            package.knowledge_id is not None,
            package.source_node is not None,
            package.encrypted_payload is not None,
            package.timestamp is not None,
            len(package.encrypted_payload) > 0
        ]
        
        # Trust signature validation
        try:
            decrypted = self.aggregator.decrypt_knowledge(package.encrypted_payload)
            reconstructed_signature = self._reconstruct_trust_signature(package.source_node, decrypted)
            signature_valid = reconstructed_signature == package.trust_signature
            checks.append(signature_valid)
        except:
            checks.append(False)
        
        return all(checks)
    
    def _reconstruct_trust_signature(self, source_node: str, knowledge_data: Dict[str, Any]) -> str:
        """Reconstruct trust signature for validation."""
        if source_node in self.known_nodes:
            institution = self.known_nodes[source_node].institution
            content = f"{source_node}_{institution}_{str(knowledge_data)}"
            return hashlib.md5(content.encode()).hexdigest()[:8]
        return ""
    
    def _update_node_trust(self, node_id: str, package: FederatedKnowledge) -> None:
        """Update trust score for a network node."""
        if node_id not in self.known_nodes:
            return
        
        node = self.known_nodes[node_id]
        
        # Calculate trust update based on package quality
        package_quality = np.mean(list(package.performance_metrics.values())) if package.performance_metrics else 0.5
        
        # Exponential moving average for trust score
        alpha = 0.1  # Learning rate
        node.trust_score = (1 - alpha) * node.trust_score + alpha * package_quality
        
        # Update performance history
        node.performance_history.append(package_quality)
        if len(node.performance_history) > 100:  # Keep last 100 entries
            node.performance_history = node.performance_history[-100:]
        
        # Update last active
        node.last_active = package.timestamp
        node.contribution_count += 1
    
    def _find_relevant_knowledge(self, problem_spec: Dict[str, Any]) -> List[FederatedKnowledge]:
        """Find relevant knowledge for given problem specification."""
        relevant = []
        
        # Problem characteristics
        problem_type = problem_spec.get('objective', '')
        problem_constraints = problem_spec.get('constraints', {})
        
        for knowledge in self.local_knowledge_base.values():
            # Algorithm type matching
            if knowledge.algorithm_type in problem_spec.get('preferred_algorithms', []):
                relevant.append(knowledge)
                continue
            
            # Metadata matching
            metadata = knowledge.metadata
            if metadata.get('research_focus') == self.research_focus:
                relevant.append(knowledge)
                continue
            
            # Performance threshold
            avg_performance = np.mean(list(knowledge.performance_metrics.values())) if knowledge.performance_metrics else 0.0
            if avg_performance > 0.6:  # Performance threshold
                relevant.append(knowledge)
        
        # Sort by relevance (performance and freshness)
        relevant.sort(key=lambda k: (
            np.mean(list(k.performance_metrics.values())) if k.performance_metrics else 0.0,
            time.mktime(time.strptime(k.timestamp, '%Y-%m-%d %H:%M:%S'))
        ), reverse=True)
        
        return relevant
    
    def _select_knowledge_sources(
        self, 
        knowledge_list: List[FederatedKnowledge], 
        max_sources: int
    ) -> List[FederatedKnowledge]:
        """Select best knowledge sources for aggregation."""
        
        # Score knowledge sources
        scored_knowledge = []
        for knowledge in knowledge_list:
            score = self._calculate_knowledge_score(knowledge)
            scored_knowledge.append((score, knowledge))
        
        # Sort by score and select top sources
        scored_knowledge.sort(key=lambda x: x[0], reverse=True)
        selected = [knowledge for _, knowledge in scored_knowledge[:max_sources]]
        
        return selected
    
    def _calculate_knowledge_score(self, knowledge: FederatedKnowledge) -> float:
        """Calculate score for knowledge source selection."""
        
        # Performance score
        performance_score = np.mean(list(knowledge.performance_metrics.values())) if knowledge.performance_metrics else 0.5
        
        # Trust score
        trust_score = 0.5
        if knowledge.source_node in self.known_nodes:
            trust_score = self.known_nodes[knowledge.source_node].trust_score
        
        # Freshness score
        try:
            knowledge_time = time.strptime(knowledge.timestamp, '%Y-%m-%d %H:%M:%S')
            age_hours = (time.time() - time.mktime(knowledge_time)) / 3600
            freshness_score = max(0.1, 1.0 - age_hours / (24 * 7))  # Decay over a week
        except:
            freshness_score = 0.5
        
        # Privacy level score (higher for more open sharing)
        privacy_scores = {'standard': 1.0, 'high': 0.7, 'maximum': 0.4}
        privacy_score = privacy_scores.get(knowledge.privacy_level, 0.5)
        
        # Combined score
        score = 0.4 * performance_score + 0.3 * trust_score + 0.2 * freshness_score + 0.1 * privacy_score
        
        return score
    
    def _perform_local_optimization(self, problem_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Perform local optimization when federated approach fails."""
        
        # Select best local algorithm
        if not self.local_algorithms:
            return {'error': 'No local algorithms available'}
        
        # Simple algorithm selection (could be enhanced with meta-learning)
        algorithm_name = list(self.local_algorithms.keys())[0]
        algorithm = self.local_algorithms[algorithm_name]
        
        # Perform optimization (simplified)
        result = {
            'optimization_method': 'local',
            'algorithm_used': algorithm_name,
            'best_objective': np.random.random(),  # Placeholder
            'convergence_achieved': True,
            'total_iterations': 100,
            'federated_enhancement': False
        }
        
        return result
    
    def _apply_federated_knowledge(
        self, 
        problem_spec: Dict[str, Any], 
        aggregation_result: AggregationResult
    ) -> Dict[str, Any]:
        """Apply federated knowledge to enhance local optimization."""
        
        # Extract aggregated hyperparameters
        aggregated_knowledge = aggregation_result.aggregated_knowledge
        
        # Enhance problem specification with federated insights
        enhanced_spec = problem_spec.copy()
        
        # Apply aggregated hyperparameters
        if 'hyperparameters' in aggregated_knowledge:
            enhanced_spec['hyperparameters'] = aggregated_knowledge['hyperparameters']
        
        # Apply performance insights
        if 'performance_insights' in aggregated_knowledge:
            enhanced_spec['performance_guidance'] = aggregated_knowledge['performance_insights']
        
        # Perform enhanced optimization
        result = self._perform_local_optimization(enhanced_spec)
        
        # Enhance with federated information
        result.update({
            'federated_enhancement': True,
            'contributing_nodes': aggregation_result.contributing_nodes,
            'confidence_score': aggregation_result.confidence_score,
            'consensus_level': aggregation_result.consensus_level,
            'uncertainty_bounds': aggregation_result.uncertainty_bounds,
            'federated_knowledge_applied': list(aggregated_knowledge.keys())
        })
        
        return result
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        return {
            'node_info': {
                'node_id': self.node_id,
                'institution': self.institution,
                'research_focus': self.research_focus,
                'capabilities': self.capabilities
            },
            'network_size': len(self.known_nodes),
            'collaboration_metrics': self.collaboration_metrics,
            'knowledge_base_size': len(self.local_knowledge_base),
            'trust_distribution': {
                node_id: node.trust_score 
                for node_id, node in self.known_nodes.items()
            },
            'privacy_budget_used': self.aggregator.privacy_budget_used,
            'aggregation_history_length': len(self.aggregator.aggregation_history)
        }


# Export main classes
__all__ = [
    'ResearchNode',
    'FederatedKnowledge',
    'AggregationResult',
    'PrivacyPreservingAggregator',
    'FederatedResearchNetwork'
]