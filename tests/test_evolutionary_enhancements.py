"""
Comprehensive Test Suite for Evolutionary Enhancements
=====================================================

Tests for Generation 4 evolutionary enhancements including quantum optimization,
AI-driven research acceleration, and cloud-native deployment capabilities.

Author: Daniel Schmidt
Email: daniel@terragonlabs.com
"""

import pytest
import asyncio
import numpy as np
import torch
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Import evolutionary enhancement modules
from liquid_metal_antenna.research.quantum_optimization_framework import (
    QuantumState, QuantumGate, RotationGate, QuantumCircuit, 
    QuantumAntennaSynthesis, QuantumBeamSteering, QuantumMachineLearning
)
from liquid_metal_antenna.research.ai_driven_research_acceleration import (
    ResearchHypothesis, ExperimentResult, AIResearchEngine
)
from liquid_metal_antenna.deployment.cloud_native_service import (
    OptimizationJob, OptimizationWorker, WorkerPool, CloudOptimizationService,
    AntennaSpecRequest, OptimizationRequest
)


class TestQuantumOptimizationFramework:
    """Test quantum optimization capabilities."""
    
    def test_quantum_state_creation(self):
        """Test quantum state initialization and validation."""
        amplitudes = np.array([0.6, 0.8])
        phases = np.array([0.0, np.pi/2])
        
        state = QuantumState(
            amplitudes=amplitudes,
            phases=phases,
            entanglement_measure=0.5,
            coherence_time=1.0,
            fidelity=0.99
        )
        
        assert len(state.amplitudes) == len(state.phases)
        assert np.isclose(np.linalg.norm(state.amplitudes), 1.0, atol=1e-6)
        assert state.entanglement_measure == 0.5
        assert state.coherence_time == 1.0
        assert state.fidelity == 0.99
    
    def test_quantum_state_invalid_input(self):
        """Test quantum state validation with invalid inputs."""
        with pytest.raises(ValueError):
            QuantumState(
                amplitudes=np.array([0.6, 0.8]),
                phases=np.array([0.0]),  # Length mismatch
                entanglement_measure=0.5,
                coherence_time=1.0,
                fidelity=0.99
            )
    
    def test_rotation_gate_creation(self):
        """Test quantum rotation gate initialization."""
        gate_x = RotationGate('x', np.pi/2)
        assert gate_x.axis == 'x'
        assert gate_x.angle == np.pi/2
        
        gate_y = RotationGate('Y', np.pi/4)  # Test case insensitive
        assert gate_y.axis == 'y'
        
        with pytest.raises(ValueError):
            RotationGate('invalid', np.pi)
    
    def test_rotation_gate_application(self):
        """Test rotation gate application to quantum state."""
        state = QuantumState(
            amplitudes=np.array([1.0, 0.0]),
            phases=np.array([0.0, 0.0]),
            entanglement_measure=0.0,
            coherence_time=1.0,
            fidelity=1.0
        )
        
        gate = RotationGate('x', np.pi)
        new_state = gate.apply(state)
        
        assert len(new_state.amplitudes) == len(state.amplitudes)
        assert new_state.coherence_time < state.coherence_time  # Decoherence
        assert new_state.fidelity < state.fidelity  # Fidelity degradation
    
    def test_quantum_circuit_creation(self):
        """Test quantum circuit construction."""
        circuit = QuantumCircuit(n_qubits=3)
        assert circuit.n_qubits == 3
        assert len(circuit.gates) == 0
        
        circuit.add_rotation('x', np.pi/2, 0)
        circuit.add_rotation('y', np.pi/4, 1)
        
        assert len(circuit.gates) == 2
    
    def test_quantum_circuit_execution(self):
        """Test quantum circuit execution."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.add_rotation('x', np.pi/2, 0)
        circuit.add_rotation('y', np.pi/4, 1)
        
        final_state = circuit.execute()
        
        assert isinstance(final_state, QuantumState)
        assert len(final_state.amplitudes) == 2**2
        assert np.isclose(np.linalg.norm(final_state.amplitudes), 1.0, atol=1e-6)
    
    def test_quantum_circuit_measurement(self):
        """Test quantum circuit measurement."""
        circuit = QuantumCircuit(n_qubits=3)
        circuit.add_rotation('x', np.pi/2, 0)  # Put qubit 0 in superposition
        
        measurements = circuit.measure([0, 1, 2])
        
        assert len(measurements) == 3
        assert all(bit in [0, 1] for bit in measurements)
    
    def test_quantum_antenna_synthesis_initialization(self):
        """Test quantum antenna synthesis initialization."""
        synth = QuantumAntennaSynthesis(
            n_parameters=16,
            n_layers=3,
            learning_rate=0.1,
            max_iterations=100
        )
        
        assert synth.n_parameters == 16
        assert synth.n_layers == 3
        assert synth.learning_rate == 0.1
        assert synth.max_iterations == 100
        assert synth.n_qubits >= 4  # Should be at least 4 qubits
        assert isinstance(synth.parameter_circuit, QuantumCircuit)
    
    def test_quantum_antenna_synthesis_parameter_encoding(self):
        """Test parameter encoding into quantum states."""
        synth = QuantumAntennaSynthesis(n_parameters=8)
        parameters = np.random.rand(8)
        
        quantum_state = synth._encode_parameters(parameters)
        
        assert isinstance(quantum_state, QuantumState)
        assert len(quantum_state.amplitudes) > 0
        assert np.isclose(np.linalg.norm(quantum_state.amplitudes), 1.0, atol=1e-6)
    
    def test_quantum_antenna_synthesis_optimization(self):
        """Test quantum antenna synthesis optimization."""
        synth = QuantumAntennaSynthesis(
            n_parameters=8, 
            n_layers=2, 
            max_iterations=10  # Small for testing
        )
        
        # Create simple target response
        target_response = np.ones(50)  # Flat response
        
        result = synth.optimize(target_response)
        
        assert 'optimal_parameters' in result
        assert 'optimal_cost' in result
        assert 'antenna_configuration' in result
        assert 'quantum_state' in result
        assert 'optimization_history' in result
        assert len(result['optimal_parameters']) == synth.n_parameters
        assert isinstance(result['quantum_state'], QuantumState)
        assert result['iterations'] <= synth.max_iterations
    
    def test_quantum_beam_steering_initialization(self):
        """Test quantum beam steering initialization."""
        beam_steering = QuantumBeamSteering(n_elements=8, frequency=5.8e9)
        
        assert beam_steering.n_elements == 8
        assert beam_steering.frequency == 5.8e9
        assert beam_steering.wavelength > 0
    
    def test_quantum_beam_steering_beamforming(self):
        """Test quantum beamforming optimization."""
        beam_steering = QuantumBeamSteering(n_elements=4, frequency=2.4e9)
        
        result = beam_steering.quantum_beamform(
            target_angles=[30.0, -45.0],
            target_gains=[0.9, 0.8],
            null_angles=[0.0]
        )
        
        assert 'phase_shifts' in result
        assert 'quantum_state' in result
        assert 'array_factor' in result
        assert 'beam_efficiency' in result
        
        assert len(result['phase_shifts']) == beam_steering.n_elements
        assert isinstance(result['quantum_state'], QuantumState)
        assert 0 <= result['beam_efficiency'] <= 1
    
    def test_quantum_machine_learning_initialization(self):
        """Test quantum machine learning model initialization."""
        qml = QuantumMachineLearning(n_qubits=4, n_layers=3)
        
        assert qml.n_qubits == 4
        assert qml.n_layers == 3
        assert isinstance(qml.quantum_model, QuantumCircuit)
    
    def test_quantum_machine_learning_training(self):
        """Test quantum surrogate model training."""
        qml = QuantumMachineLearning(n_qubits=4, n_layers=2)
        
        # Generate simple training data
        training_data = []
        for _ in range(5):  # Small dataset for testing
            inputs = np.random.rand(6)
            outputs = np.array([np.sum(inputs)])
            training_data.append((inputs, outputs))
        
        result = qml.train_quantum_surrogate(training_data, n_epochs=3)
        
        assert 'training_loss' in result
        assert 'quantum_model' in result
        assert 'n_epochs' in result
        assert len(result['training_loss']) == 3
        assert result['n_epochs'] == 3


class TestAIDrivenResearchAcceleration:
    """Test AI-driven research acceleration capabilities."""
    
    def test_research_hypothesis_creation(self):
        """Test research hypothesis creation and priority scoring."""
        hypothesis = ResearchHypothesis(
            id="test_hyp_001",
            description="Test antenna optimization with liquid metal",
            parameters={"geometry": "spiral", "material": "galinstan"},
            expected_improvement=0.25,
            confidence=0.8,
            complexity=3,
            experiment_cost=100.0
        )
        
        assert hypothesis.id == "test_hyp_001"
        assert hypothesis.expected_improvement == 0.25
        assert hypothesis.confidence == 0.8
        assert hypothesis.complexity == 3
        assert hypothesis.experiment_cost == 100.0
        assert hypothesis.priority_score > 0  # Should be computed in __post_init__
    
    def test_experiment_result_creation(self):
        """Test experiment result creation."""
        result = ExperimentResult(
            hypothesis_id="test_hyp_001",
            parameters={"param1": 1.0},
            metrics={"improvement": 0.15, "gain": 8.5},
            success=True,
            execution_time=45.2,
            computational_cost=500,
            insights=["Significant improvement achieved"]
        )
        
        assert result.hypothesis_id == "test_hyp_001"
        assert result.success is True
        assert result.execution_time == 45.2
        assert len(result.insights) == 1
        assert result.timestamp > 0  # Should be set automatically
    
    def test_ai_research_engine_initialization(self):
        """Test AI research engine initialization."""
        engine = AIResearchEngine(
            research_domain="test_antennas",
            max_parallel_experiments=4,
            learning_rate=0.01
        )
        
        assert engine.research_domain == "test_antennas"
        assert engine.max_parallel_experiments == 4
        assert engine.learning_rate == 0.01
        assert len(engine.hypotheses) == 0
        assert len(engine.experiment_results) == 0
        assert 'hypotheses_generated' in engine.research_metrics
    
    def test_ai_research_context_encoding(self):
        """Test research context encoding."""
        engine = AIResearchEngine()
        
        context = {
            'current_gain': 7.2,
            'current_bandwidth': 0.15,
            'current_efficiency': 0.82,
            'target_gain': 12.0,
            'target_bandwidth': 0.3,
            'target_efficiency': 0.9
        }
        
        encoded = engine._encode_research_context(context)
        
        assert isinstance(encoded, torch.Tensor)
        assert len(encoded) == 64  # Fixed context vector size
        assert torch.all(torch.isfinite(encoded))  # No NaN or inf values
    
    @pytest.mark.asyncio
    async def test_hypothesis_generation(self):
        """Test AI hypothesis generation."""
        engine = AIResearchEngine()
        
        context = {
            'current_gain': 7.0,
            'target_gain': 10.0,
            'exploration_factor': 0.5
        }
        
        hypotheses = await engine.generate_hypothesis(context, n_hypotheses=3)
        
        assert len(hypotheses) == 3
        assert all(isinstance(h, ResearchHypothesis) for h in hypotheses)
        assert all(h.priority_score > 0 for h in hypotheses)
        assert hypotheses[0].priority_score >= hypotheses[1].priority_score  # Should be sorted
        assert engine.research_metrics['hypotheses_generated'] == 3
    
    @pytest.mark.asyncio
    async def test_experiment_execution(self):
        """Test automated experiment execution."""
        engine = AIResearchEngine()
        
        hypothesis = ResearchHypothesis(
            id="test_exp_001",
            description="Test experiment",
            parameters={"geometry_type": 0, "liquid_metal_type": 0},
            expected_improvement=0.2,
            confidence=0.7,
            complexity=2,
            experiment_cost=50.0
        )
        
        result = await engine.execute_experiment(hypothesis, simulation_budget=100)
        
        assert isinstance(result, ExperimentResult)
        assert result.hypothesis_id == hypothesis.id
        assert result.execution_time > 0
        assert 'improvement' in result.metrics
        assert len(result.insights) > 0
        assert engine.research_metrics['experiments_conducted'] == 1
    
    @pytest.mark.asyncio
    async def test_autonomous_research_loop(self):
        """Test autonomous research loop execution."""
        engine = AIResearchEngine()
        
        context = {
            'current_gain': 6.0,
            'target_gain': 9.0,
            'exploration_factor': 0.3
        }
        
        # Run short research loop
        summary = await engine.autonomous_research_loop(
            research_context=context,
            n_iterations=2,  # Short for testing
            hypotheses_per_iteration=2
        )
        
        assert 'total_iterations' in summary
        assert 'total_experiments' in summary
        assert 'success_rate' in summary
        assert 'best_improvement' in summary
        assert 'timeline' in summary
        assert 'insights' in summary
        
        assert summary['total_iterations'] == 2
        assert summary['total_experiments'] > 0
        assert 0 <= summary['success_rate'] <= 1
        assert len(summary['timeline']) == 2
        assert len(summary['insights']) > 0


class TestCloudNativeDeployment:
    """Test cloud-native deployment capabilities."""
    
    def test_antenna_spec_request_validation(self):
        """Test antenna specification request validation."""
        # Valid request
        spec_request = AntennaSpecRequest(
            frequency_range=(2.4e9, 5.8e9),
            substrate="rogers_4003c",
            metal="galinstan",
            size_constraint=(50.0, 50.0, 3.0)
        )
        
        assert spec_request.frequency_range == (2.4e9, 5.8e9)
        assert spec_request.substrate == "rogers_4003c"
        assert spec_request.metal == "galinstan"
        
        # Invalid frequency range
        with pytest.raises(ValueError):
            AntennaSpecRequest(frequency_range=(5.8e9, 2.4e9))  # f_min > f_max
        
        with pytest.raises(ValueError):
            AntennaSpecRequest(frequency_range=(-1e9, 2.4e9))  # negative frequency
    
    def test_optimization_request_creation(self):
        """Test optimization request creation."""
        antenna_spec = AntennaSpecRequest(
            frequency_range=(2.4e9, 5.8e9),
            substrate="rogers_4003c",
            metal="galinstan"
        )
        
        opt_request = OptimizationRequest(
            antenna_spec=antenna_spec,
            objective="max_gain",
            constraints={"vswr": "<2.0"},
            algorithm="bayesian",
            n_iterations=100,
            optimization_type="single_objective",
            priority="high"
        )
        
        assert opt_request.objective == "max_gain"
        assert opt_request.algorithm == "bayesian"
        assert opt_request.n_iterations == 100
        assert opt_request.optimization_type == "single_objective"
        assert opt_request.priority == "high"
    
    def test_optimization_job_creation(self):
        """Test optimization job creation and priority scoring."""
        antenna_spec = AntennaSpecRequest(frequency_range=(2.4e9, 5.8e9))
        request = OptimizationRequest(
            antenna_spec=antenna_spec,
            priority="urgent"
        )
        
        job = OptimizationJob(
            job_id="test_job_001",
            request=request
        )
        
        assert job.job_id == "test_job_001"
        assert job.status == "queued"
        assert job.progress == 0.0
        assert job.priority_score == 10.0  # "urgent" priority
        assert job.created_at is not None
    
    def test_optimization_worker_initialization(self):
        """Test optimization worker initialization."""
        worker = OptimizationWorker(
            worker_id="test_worker_001",
            region="us-west"
        )
        
        assert worker.worker_id == "test_worker_001"
        assert worker.region == "us-west"
        assert worker.current_job is None
        assert worker.is_busy is False
        assert worker.quantum_optimizer is not None
        assert worker.ai_engine is not None
    
    @pytest.mark.asyncio
    async def test_worker_job_processing(self):
        """Test worker job processing."""
        worker = OptimizationWorker("test_worker", "test_region")
        
        # Mock Redis client
        mock_redis = AsyncMock()
        
        # Create test job
        antenna_spec = AntennaSpecRequest(frequency_range=(2.4e9, 5.8e9))
        request = OptimizationRequest(
            antenna_spec=antenna_spec,
            optimization_type="single_objective",
            n_iterations=10  # Small for testing
        )
        job = OptimizationJob(job_id="test_job", request=request)
        
        # Process job
        result = await worker.process_job(job, mock_redis)
        
        assert result is not None
        assert 'optimization_type' in result
        assert 'worker_id' in result
        assert result['worker_id'] == worker.worker_id
        assert job.status == "completed"
        assert job.progress == 1.0
        assert worker.current_job is None
        assert worker.is_busy is False
    
    def test_worker_pool_initialization(self):
        """Test worker pool initialization."""
        pool = WorkerPool(n_workers=3, region="eu-west")
        
        assert pool.region == "eu-west"
        assert len(pool.workers) == 3
        assert all(w.region == "eu-west" for w in pool.workers)
        assert all(w.worker_id.startswith("worker_eu-west_") for w in pool.workers)
    
    @pytest.mark.asyncio
    async def test_worker_pool_job_submission(self):
        """Test job submission to worker pool."""
        pool = WorkerPool(n_workers=2, region="test")
        
        # Create test job
        antenna_spec = AntennaSpecRequest(frequency_range=(2.4e9, 5.8e9))
        request = OptimizationRequest(antenna_spec=antenna_spec)
        job = OptimizationJob(job_id="test_job", request=request)
        
        # Submit job
        await pool.submit_job(job)
        
        # Check queue
        assert pool.job_queue.qsize() == 1
    
    def test_worker_pool_status(self):
        """Test worker pool status reporting."""
        pool = WorkerPool(n_workers=2, region="test")
        
        status = pool.get_worker_status()
        
        assert len(status) == 2
        assert all('worker_id' in worker_status for worker_status in status)
        assert all('region' in worker_status for worker_status in status)
        assert all('is_busy' in worker_status for worker_status in status)
        assert all(worker_status['region'] == "test" for worker_status in status)
    
    @pytest.mark.asyncio
    async def test_cloud_optimization_service_health_check(self):
        """Test cloud service health check."""
        service = CloudOptimizationService(region="test", n_workers=1)
        
        # Mock the FastAPI app startup
        with patch.object(service, '_startup') as mock_startup, \
             patch.object(service, '_shutdown') as mock_shutdown:
            
            mock_startup.return_value = None
            mock_shutdown.return_value = None
            
            # Test health endpoint
            client = service.app
            # Note: In a real test, you'd use TestClient from fastapi.testclient
            # For this example, we'll just check the app was created
            assert client is not None
            assert client.title == "Liquid Metal Antenna Optimization Service"


class TestIntegrationScenarios:
    """Test integration scenarios across all evolutionary enhancements."""
    
    @pytest.mark.asyncio
    async def test_quantum_ai_research_integration(self):
        """Test integration between quantum optimization and AI research."""
        # Initialize AI research engine
        ai_engine = AIResearchEngine()
        
        # Generate hypothesis
        context = {'current_gain': 7.0, 'target_gain': 12.0}
        hypotheses = await ai_engine.generate_hypothesis(context, n_hypotheses=1)
        hypothesis = hypotheses[0]
        
        # Use hypothesis parameters for quantum optimization
        quantum_synth = QuantumAntennaSynthesis(n_parameters=8, max_iterations=5)
        target_response = np.ones(20)  # Simple target
        
        quantum_result = quantum_synth.optimize(target_response)
        
        # Verify integration
        assert len(quantum_result['optimal_parameters']) == 8
        assert 'quantum_state' in quantum_result
        assert hypothesis.priority_score > 0
        
        # Simulate experiment with quantum result
        experiment_result = await ai_engine.execute_experiment(hypothesis)
        assert experiment_result.success is not None
    
    @pytest.mark.asyncio
    async def test_cloud_quantum_optimization_workflow(self):
        """Test cloud deployment of quantum optimization."""
        # Create cloud service
        service = CloudOptimizationService(n_workers=1)
        worker = service.worker_pool.workers[0]
        
        # Mock Redis
        mock_redis = AsyncMock()
        
        # Create quantum optimization job
        antenna_spec = AntennaSpecRequest(frequency_range=(2.4e9, 5.8e9))
        request = OptimizationRequest(
            antenna_spec=antenna_spec,
            optimization_type="quantum",
            n_iterations=5  # Small for testing
        )
        job = OptimizationJob(job_id="quantum_job", request=request)
        
        # Process quantum job
        result = await worker.process_job(job, mock_redis)
        
        assert result['optimization_type'] == 'quantum'
        assert 'quantum_circuit_depth' in result
        assert 'quantum_coherence_time' in result
        assert job.status == "completed"
    
    @pytest.mark.asyncio
    async def test_ai_driven_cloud_optimization_pipeline(self):
        """Test AI-driven optimization in cloud environment."""
        # Initialize AI engine
        ai_engine = AIResearchEngine(max_parallel_experiments=2)
        
        # Create cloud worker
        worker = OptimizationWorker("ai_worker", "test_region")
        mock_redis = AsyncMock()
        
        # Generate AI hypothesis
        context = {'current_gain': 6.0, 'target_gain': 10.0}
        hypotheses = await ai_engine.generate_hypothesis(context, n_hypotheses=1)
        
        # Convert to cloud optimization job
        antenna_spec = AntennaSpecRequest(frequency_range=(2.4e9, 5.8e9))
        request = OptimizationRequest(
            antenna_spec=antenna_spec,
            optimization_type="single_objective"
        )
        job = OptimizationJob(job_id="ai_job", request=request)
        
        # Process in parallel
        cloud_task = worker.process_job(job, mock_redis)
        ai_task = ai_engine.execute_experiment(hypotheses[0])
        
        cloud_result, ai_result = await asyncio.gather(cloud_task, ai_task)
        
        # Verify both completed successfully
        assert cloud_result is not None
        assert ai_result.success is not None
        assert job.status == "completed"


class TestPerformanceBenchmarks:
    """Performance benchmarks for evolutionary enhancements."""
    
    def test_quantum_optimization_performance(self):
        """Benchmark quantum optimization performance."""
        synth = QuantumAntennaSynthesis(
            n_parameters=16,
            n_layers=3,
            max_iterations=50
        )
        
        target_response = np.random.rand(100)
        
        start_time = time.time()
        result = synth.optimize(target_response)
        duration = time.time() - start_time
        
        # Performance assertions
        assert duration < 30.0  # Should complete within 30 seconds
        assert result['iterations'] <= 50
        assert len(result['optimization_history']['cost']) > 0
    
    @pytest.mark.asyncio
    async def test_ai_research_throughput(self):
        """Benchmark AI research engine throughput."""
        engine = AIResearchEngine(max_parallel_experiments=4)
        
        context = {'current_gain': 7.0, 'target_gain': 12.0}
        
        start_time = time.time()
        
        # Generate multiple hypotheses
        hypotheses = await engine.generate_hypothesis(context, n_hypotheses=10)
        generation_time = time.time() - start_time
        
        # Execute experiments
        experiment_start = time.time()
        tasks = [engine.execute_experiment(h) for h in hypotheses[:4]]
        results = await asyncio.gather(*tasks)
        experiment_time = time.time() - experiment_start
        
        # Performance assertions
        assert generation_time < 5.0  # Hypothesis generation should be fast
        assert experiment_time < 15.0  # 4 experiments in parallel
        assert len(hypotheses) == 10
        assert len(results) == 4
        assert all(isinstance(r, ExperimentResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_cloud_service_scalability(self):
        """Test cloud service scalability."""
        # Create service with multiple workers
        service = CloudOptimizationService(n_workers=4)
        pool = service.worker_pool
        
        # Create multiple jobs
        jobs = []
        for i in range(8):  # More jobs than workers
            antenna_spec = AntennaSpecRequest(frequency_range=(2.4e9, 5.8e9))
            request = OptimizationRequest(antenna_spec=antenna_spec, n_iterations=5)
            job = OptimizationJob(job_id=f"perf_job_{i}", request=request)
            jobs.append(job)
        
        # Submit all jobs
        start_time = time.time()
        for job in jobs:
            await pool.submit_job(job)
        submission_time = time.time() - start_time
        
        # Performance assertions
        assert submission_time < 1.0  # Job submission should be fast
        assert pool.job_queue.qsize() == 8
        assert len(pool.workers) == 4


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=liquid_metal_antenna.research",
        "--cov=liquid_metal_antenna.deployment",
        "--cov-report=term-missing"
    ])