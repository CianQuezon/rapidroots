import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
import platform
import os
from typing import Callable, Dict, List, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import the function to benchmark
from rapidroots.utils.validation.validate_bracket_numeric import validate_brackets_numeric


class PerformanceBenchmark:
    """Professional benchmarking suite for validate_brackets_numeric function"""
    
    def __init__(self, seed_fin: int = 42, seed_clim: int = 123):
        self.results = []
        # Use numpy's new RNG for better reproducibility
        self.rng_fin = np.random.default_rng(seed_fin)
        self.rng_clim = np.random.default_rng(seed_clim)
        # Store seeds for metadata
        self.seed_fin = seed_fin
        self.seed_clim = seed_clim
        
    def generate_financial_data(self, n: int, scenario: str = "normal") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate realistic financial time series data for bracket validation"""
        r = self.rng_fin  # shorthand
        
        if scenario == "normal":
            # Normal market conditions
            a = r.uniform(50, 100, n)
            b = a + r.uniform(5, 20, n)
            
            # Option payoff functions (simplified Black-Scholes style)
            strike = r.uniform(60, 110, n)
            volatility = r.uniform(0.1, 0.4, n)
            
            fa = (a - strike) * np.exp(-volatility * 0.25)
            fb = (b - strike) * np.exp(-volatility * 0.25)
            
        elif scenario == "crisis":
            # Financial crisis conditions - high volatility, extreme values
            a = r.uniform(10, 200, n)
            b = a + r.uniform(1, 50, n)
            
            # Crisis scenario with fat tails and extreme movements
            shock = r.choice([-1, 1], n) * r.exponential(0.3, n)
            fa = r.normal(-2, 5, n) + shock
            fb = r.normal(2, 5, n) + shock
            
        elif scenario == "stress":
            # Stress test scenario with some invalid brackets
            a = r.uniform(20, 150, n)
            b = a + r.uniform(-10, 30, n)  # Some negative intervals
            
            # Mix of valid and invalid brackets
            fa = r.normal(0, 3, n)
            fb = r.normal(0, 3, n)
            # Introduce same signs for stress testing
            same_sign_mask = r.random(n) < 0.3
            fb[same_sign_mask] = np.abs(fa[same_sign_mask]) + 0.1
            
        else:
            raise ValueError(f"Unknown financial scenario: {scenario}")
            
        return a, b, fa, fb
    
    def generate_climate_data(self, n: int, scenario: str = "temperature") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate realistic climate/environmental data for bracket validation"""
        r = self.rng_clim
        
        if scenario == "temperature":
            a = r.normal(15, 10, n)  # Min daily temperature
            b = a + r.uniform(5, 15, n)  # Max daily temperature
            
            # Heat transfer functions
            fa = np.sin(a * np.pi / 180) * r.normal(1, 0.2, n)
            fb = np.cos(b * np.pi / 180) * r.normal(1, 0.2, n)
            
        elif scenario == "precipitation":
            a = r.exponential(2, n)  # precip_min
            b = a + r.exponential(5, n)  # precip_max
            
            # Hydrological response functions
            fa = -np.log(a + 0.1) + r.normal(0, 0.5, n)
            fb = np.log(b + 0.1) + r.normal(0, 0.5, n)
            
        elif scenario == "co2":
            a = r.uniform(350, 420, n)  # co2_lower
            b = a + r.uniform(10, 100, n)  # co2_upper
            
            # Atmospheric absorption functions
            fa = -np.sqrt(a - 300) + r.normal(0, 1, n)
            fb = np.sqrt(b - 300) + r.normal(0, 1, n)
            
        else:
            raise ValueError(f"Unknown climate scenario: {scenario}")
            
        return a, b, fa, fb
    
    def benchmark_single_run(self, data_generator: Callable, data_type: str, n: int, scenario: str, 
                           tolerance: float = 0.0, dtype=np.float64, 
                           make_messages: bool = False,
                           repeats: int = 5, warmup: bool = True, track_memory: bool = False) -> Dict:
        """Benchmark a single run with professional timing methodology"""
        
        # Track memory if requested and psutil is available
        memory_before_mb = np.nan
        memory_after_mb = np.nan
        memory_peak_mb = np.nan
        
        if track_memory and PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            memory_before_mb = process.memory_info().rss / (1024**2)
        
        # Generate data
        data_gen_start = time.perf_counter()
        try:
            a, b, fa, fb = data_generator(n, scenario)
        except Exception as e:
            return {
                'n': n, 'scenario': scenario,
                'data_type': data_type,
                'kernel_ms': np.nan, 'kernel_std_ms': np.nan,
                'data_gen_ms': np.nan, 'throughput': np.nan,
                'memory_mb': np.nan, 'valid_ratio': np.nan,
                'tolerance': tolerance, 'dtype': str(dtype),
                'make_messages': make_messages, 
                'seed_used': self.seed_fin if data_type == "financial" else self.seed_clim,
                'memory_before_mb': memory_before_mb,
                'memory_after_mb': np.nan,
                'memory_peak_mb': np.nan,
                'error': repr(e),
            }
        data_gen_time = (time.perf_counter() - data_gen_start) * 1000.0
        
        # Validate shapes early
        if not (a.shape == b.shape == fa.shape == fb.shape):
            return {
                'n': n, 'scenario': scenario, 'data_type': data_type,
                'kernel_ms': np.nan, 'kernel_std_ms': np.nan,
                'data_gen_ms': data_gen_time, 'throughput': np.nan,
                'memory_mb': np.nan, 'valid_ratio': np.nan,
                'tolerance': tolerance, 'dtype': str(dtype),
                'make_messages': make_messages,
                'seed_used': self.seed_fin if data_type == "financial" else self.seed_clim,
                'memory_before_mb': memory_before_mb,
                'memory_after_mb': np.nan,
                'memory_peak_mb': np.nan,
                'error': f"Shape mismatch: a={a.shape}, b={b.shape}, fa={fa.shape}, fb={fb.shape}",
            }
        
        # Pre-cast arrays to target dtype for pure kernel benchmarking
        if dtype is not None:
            a = a.astype(dtype, copy=False)
            b = b.astype(dtype, copy=False)  
            fa = fa.astype(dtype, copy=False)
            fb = fb.astype(dtype, copy=False)
        
        # Calculate memory usage
        memory_usage = (a.nbytes + b.nbytes + fa.nbytes + fb.nbytes) / (1024**2)  # MB
        
        # Warmup run (not measured)
        if warmup:
            try:
                validate_brackets_numeric(
                    a, b, fa=fa, fb=fb, 
                    tolerance=tolerance, 
                    dtype=None,  # Skip re-casting since we pre-cast
                    make_messages=False
                )
            except Exception:
                pass  # Warmup failures are not critical
        
        # Benchmark runs with repetition for statistical stability
        timings = []
        is_valid = None
        result = None
        max_memory_during = memory_before_mb  # Track peak memory during benchmark
        
        for _ in range(repeats):
            start_time = time.perf_counter()
            try:
                is_valid, result = validate_brackets_numeric(
                    a, b, fa=fa, fb=fb, 
                    tolerance=tolerance, 
                    dtype=None,  # Skip re-casting
                    make_messages=make_messages
                )
                execution_time = (time.perf_counter() - start_time) * 1000.0  # ms
                timings.append(execution_time)
                
                # Track peak memory during execution
                if track_memory and PSUTIL_AVAILABLE:
                    current_memory = process.memory_info().rss / (1024**2)
                    max_memory_during = max(max_memory_during, current_memory)
                    
            except Exception as e:
                return {
                    'n': n, 'scenario': scenario,
                    'data_type': data_type,
                    'kernel_ms': np.nan, 'kernel_std_ms': np.nan,
                    'data_gen_ms': data_gen_time, 'throughput': np.nan,
                    'memory_mb': memory_usage, 'valid_ratio': np.nan,
                    'tolerance': tolerance, 'dtype': str(dtype),
                    'make_messages': make_messages,
                    'seed_used': self.seed_fin if data_type == "financial" else self.seed_clim,
                    'memory_before_mb': memory_before_mb,
                    'memory_after_mb': np.nan,
                    'memory_peak_mb': np.nan,
                    'error': repr(e),
                }
        
        # Final memory measurement
        if track_memory and PSUTIL_AVAILABLE:
            memory_after_mb = process.memory_info().rss / (1024**2)
            memory_peak_mb = max_memory_during
        
        # Use median for robust central tendency, std for variability
        timings = np.array(timings)
        kernel_ms = np.median(timings)
        kernel_std = np.std(timings, ddof=1) if len(timings) > 1 else 0.0
        
        # Guard against zero kernel time
        denom = max(kernel_ms / 1000.0, 1e-12)
        throughput = n / denom  # items per second
        valid_ratio = float(np.mean(is_valid)) if is_valid is not None else np.nan
        
        return {
            'n': n,
            'scenario': scenario,
            'data_type': data_type,
            'kernel_ms': kernel_ms,
            'kernel_std_ms': kernel_std,
            'data_gen_ms': data_gen_time,
            'throughput': throughput,
            'memory_mb': memory_usage,
            'valid_ratio': valid_ratio,
            'tolerance': tolerance,
            'dtype': str(dtype),
            'make_messages': make_messages,
            'seed_used': self.seed_fin if data_type == "financial" else self.seed_clim,
            'memory_before_mb': memory_before_mb,
            'memory_after_mb': memory_after_mb,
            'memory_peak_mb': memory_peak_mb,
            'error': None,
        }
    
    def run_scaling_benchmark(self, max_n: int = 1_000_000, include_messages: bool = False) -> pd.DataFrame:
        """Run comprehensive scaling benchmark with professional methodology"""
        
        print("🚀 Professional Performance Benchmark Suite")
        print("=" * 60)
        print("Methodology:")
        print("• Separate random number generators for reproducibility")
        print("• Warmup runs to stabilize performance")
        print("• Multiple repetitions with median timing")
        print("• Pre-cast arrays for pure kernel benchmarking")
        print("• Error capture and reporting")
        print("=" * 60)
        
        # Test sizes following logarithmic scaling for performance analysis
        sizes = np.array([
            1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
            100_000, 200_000, 500_000, 1_000_000
        ])
        
        if max_n > 1_000_000:
            sizes = np.append(sizes, [2_000_000, 5_000_000])
            
        # Focused scenarios for realistic testing
        scenarios = {
            'financial': ['normal', 'crisis'],
            'climate': ['temperature', 'precipitation']
        }
        
        # Key parameter variations
        dtypes = [np.float32, np.float64]
        tolerances = [0.0, 1e-6]
        message_flags = [False, True] if include_messages else [False]
        
        results = []
        total_configs = 0
        for data_type, scenario_list in scenarios.items():
            total_configs += len(sizes) * len(scenario_list) * len(dtypes) * len(tolerances) * len(message_flags)
        
        test_count = 0
        
        for n in sizes:
            print(f"\n📊 Testing with {n:,} data points...")
            
            # Financial data benchmarks
            for scenario in scenarios['financial']:
                for dtype in dtypes:
                    for tolerance in tolerances:
                        for make_messages in message_flags:
                            test_count += 1
                            progress = f"{test_count}/{total_configs}"
                            msg_label = "msgs" if make_messages else "codes"
                            print(f"  {progress} - Financial {scenario} ({dtype.__name__}, tol={tolerance}, {msg_label})")
                            
                            result = self.benchmark_single_run(
                                self.generate_financial_data, "financial", n, scenario, 
                                tolerance, dtype, make_messages=make_messages,
                                repeats=5, warmup=True, track_memory=(n >= 1_000_000)
                            )
                            results.append(result)
                            
                            if result['error']:
                                print(f"    ❌ Error: {result['error']}")
                            else:
                                print(f"    ✅ {result['kernel_ms']:.2f}ms, {result['throughput']:.0f} items/s")
            
            # Climate data benchmarks  
            for scenario in scenarios['climate']:
                for dtype in dtypes:
                    for tolerance in tolerances:
                        for make_messages in message_flags:
                            test_count += 1
                            progress = f"{test_count}/{total_configs}"
                            msg_label = "msgs" if make_messages else "codes"
                            print(f"  {progress} - Climate {scenario} ({dtype.__name__}, tol={tolerance}, {msg_label})")
                            
                            result = self.benchmark_single_run(
                                self.generate_climate_data, "climate", n, scenario,
                                tolerance, dtype, make_messages=make_messages,
                                repeats=5, warmup=True, track_memory=(n >= 1_000_000)
                            )
                            results.append(result)
                            
                            if result['error']:
                                print(f"    ❌ Error: {result['error']}")
                            else:
                                print(f"    ✅ {result['kernel_ms']:.2f}ms, {result['throughput']:.0f} items/s")
        
        self.results = pd.DataFrame(results)
        
        # Enforce proper dtypes for numerical columns
        numeric_columns = ['kernel_ms', 'kernel_std_ms', 'data_gen_ms', 'throughput', 'memory_mb', 'valid_ratio',
                          'memory_before_mb', 'memory_after_mb', 'memory_peak_mb']
        for col in numeric_columns:
            if col in self.results.columns:
                self.results[col] = pd.to_numeric(self.results[col], errors='coerce')
        
        # Summary of results
        successful = self.results['error'].isna().sum()
        failed = self.results['error'].notna().sum()
        print(f"\n✅ Benchmark completed! {successful} successful, {failed} failed tests")
        
        return self.results
    
    def run_linearity_test(self, target_size: int = 30_000_000) -> pd.DataFrame:
        """Run high-end linearity test to verify scaling doesn't break at extreme sizes"""
        
        print(f"\n🔍 HIGH-END LINEARITY TEST ({target_size:,} data points)")
        print("=" * 60)
        print("Testing for cache effects, NUMA issues, and OS paging at extreme scale...")
        
        if not PSUTIL_AVAILABLE:
            print("⚠️  psutil not available - memory tracking disabled")
        
        # Test both float32 and float64 at extreme scale
        dtypes_to_test = [np.float32, np.float64]
        linearity_results = []
        
        for dtype in dtypes_to_test:
            print(f"\n📊 Testing {dtype.__name__} with {target_size:,} data points...")
            
            result = self.benchmark_single_run(
                self.generate_financial_data, "financial", target_size, "normal",
                tolerance=0.0, dtype=dtype, make_messages=False,
                repeats=3, warmup=True, track_memory=True
            )
            linearity_results.append(result)
            
            if result['error']:
                print(f"    ❌ Failed: {result['error']}")
            else:
                print(f"    ✅ Success:")
                print(f"       Execution time: {result['kernel_ms']:.2f} ms")
                print(f"       Throughput: {result['throughput']:.0f} items/s")
                print(f"       Array memory: {result['memory_mb']:.1f} MB")
                
                if PSUTIL_AVAILABLE and not np.isnan(result['memory_peak_mb']):
                    memory_delta = result['memory_peak_mb'] - result['memory_before_mb']
                    print(f"       RSS before: {result['memory_before_mb']:.1f} MB")
                    print(f"       RSS peak: {result['memory_peak_mb']:.1f} MB")
                    print(f"       RSS delta: {memory_delta:.1f} MB")
        
        # Convert to DataFrame and combine with main results
        linearity_df = pd.DataFrame(linearity_results)
        
        # Enforce dtypes
        numeric_columns = ['kernel_ms', 'kernel_std_ms', 'data_gen_ms', 'throughput', 'memory_mb', 'valid_ratio',
                          'memory_before_mb', 'memory_after_mb', 'memory_peak_mb']
        for col in numeric_columns:
            if col in linearity_df.columns:
                linearity_df[col] = pd.to_numeric(linearity_df[col], errors='coerce')
        
        print(f"\n✅ Linearity test completed!")
        return linearity_df
    
    def analyze_performance(self) -> None:
        """Analyze and report benchmark results with professional insights"""
        
        if self.results.empty:
            print("No benchmark results available. Run benchmark first.")
            return
            
        print("\n" + "=" * 60)
        print("📈 PROFESSIONAL PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Filter successful results for analysis
        successful_results = self.results[self.results['error'].isna()].copy()
        
        if successful_results.empty:
            print("❌ No successful benchmark results to analyze.")
            return
        
        # Summary statistics with proper statistical measures
        print(f"\n🔍 STATISTICAL SUMMARY ({len(successful_results)} measurements):")
        summary_stats = successful_results.groupby(['data_type', 'dtype']).agg({
            'kernel_ms': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'throughput': ['mean', 'median', 'std'],
            'memory_mb': ['mean', 'max'],
            'valid_ratio': 'mean'
        }).round(3)
        print(summary_stats)
        
        # Performance at 1M data points with confidence intervals
        million_data = successful_results[successful_results['n'] == 1_000_000]
        if not million_data.empty:
            print(f"\n🎯 PERFORMANCE AT 1 MILLION DATA POINTS:")
            kernel_times = million_data['kernel_ms']
            throughputs = million_data['throughput']
            
            print(f"  Execution time: {kernel_times.median():.2f} ms (median)")
            print(f"                  {kernel_times.mean():.2f} ± {kernel_times.std():.2f} ms (mean ± std)")
            print(f"  Throughput: {throughputs.median():.0f} items/second (median)")
            print(f"  Memory usage: {million_data['memory_mb'].mean():.1f} MB")
            
            # Best performing configuration
            best_config = million_data.loc[million_data['kernel_ms'].idxmin()]
            print(f"  Fastest config: {best_config['data_type']} {best_config['scenario']} ({best_config['dtype']})")
        
        # Scaling analysis with correlation coefficient
        print(f"\n📏 SCALING CHARACTERISTICS:")
        large_data = successful_results[successful_results['n'] >= 100_000]
        if len(large_data) > 2:
            scaling_corr = np.corrcoef(large_data['n'], large_data['kernel_ms'])[0,1]
            print(f"  Size vs execution time correlation: {scaling_corr:.3f}")
            
            # Interpretation based on research best practices
            if scaling_corr > 0.95:
                print("  ✅ Excellent linear scaling (r > 0.95)")
            elif scaling_corr > 0.85:
                print("  ✅ Good scaling characteristics (r > 0.85)")
            else:
                print(f"  ⚠️  Non-linear scaling detected (r = {scaling_corr:.3f})")
        
        # Error analysis
        errors = self.results[self.results['error'].notna()]
        if not errors.empty:
            print(f"\n❌ ERROR ANALYSIS ({len(errors)} failures):")
            error_summary = errors.groupby(['data_type', 'scenario', 'dtype']).size()
            print(error_summary)
        
        # Data type performance comparison
        print(f"\n🔬 DATA TYPE PERFORMANCE COMPARISON:")
        dtype_stats = successful_results.groupby('dtype').agg({
            'kernel_ms': ['mean', 'median', 'std'],
            'throughput': ['mean', 'median'],
            'memory_mb': 'mean'
        }).round(3)
        print(dtype_stats)
    
    def generate_visualizations(self) -> None:
        """Generate professional performance visualization plots"""
        
        successful_results = self.results[self.results['error'].isna()]
        if successful_results.empty:
            print("No successful benchmark results for visualization.")
            return
            
        plt.style.use('default')  # More professional than seaborn for benchmarks
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('validate_brackets_numeric Performance Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1. Scaling Performance (Log-Log)
        ax1 = axes[0, 0]
        for dtype in successful_results['dtype'].unique():
            data = successful_results[successful_results['dtype'] == dtype]
            scaling_data = data.groupby('n').agg({
                'kernel_ms': ['mean', 'std']
            }).reset_index()
            
            means = scaling_data[('kernel_ms', 'mean')]
            stds = scaling_data[('kernel_ms', 'std')]
            
            ax1.loglog(scaling_data['n'], means, 'o-', 
                      label=f'{dtype}', alpha=0.8, linewidth=2, markersize=6)
            ax1.fill_between(scaling_data['n'], means - stds, means + stds, 
                           alpha=0.2)
        
        ax1.set_xlabel('Data Points (n)', fontsize=12)
        ax1.set_ylabel('Execution Time (ms)', fontsize=12)
        ax1.set_title('Scaling Performance with Error Bars', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Throughput Analysis
        ax2 = axes[0, 1]
        throughput_data = successful_results.groupby(['n', 'data_type'])['throughput'].mean().reset_index()
        for data_type in throughput_data['data_type'].unique():
            data = throughput_data[throughput_data['data_type'] == data_type]
            ax2.semilogx(data['n'], data['throughput'], 'o-', 
                        label=f'{data_type.capitalize()}', 
                        alpha=0.8, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Data Points (n)', fontsize=12)
        ax2.set_ylabel('Throughput (items/second)', fontsize=12)
        ax2.set_title('Throughput vs Dataset Size', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Memory Scaling
        ax3 = axes[1, 0]
        memory_data = successful_results.groupby('n')['memory_mb'].mean()
        ax3.loglog(memory_data.index, memory_data.values, 'ro-', 
                  alpha=0.8, linewidth=2, markersize=6)
        ax3.set_xlabel('Data Points (n)', fontsize=12)
        ax3.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax3.set_title('Memory Scaling', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Distribution
        ax4 = axes[1, 1]
        large_data = successful_results[successful_results['n'] >= 100_000]
        if not large_data.empty:
            performance_by_type = [
                large_data[large_data['data_type'] == dt]['kernel_ms'].values 
                for dt in large_data['data_type'].unique()
            ]
            labels = large_data['data_type'].unique()
            
            box_plot = ax4.boxplot(performance_by_type, labels=labels, 
                                  patch_artist=True, notch=True)
            ax4.set_ylabel('Execution Time (ms)', fontsize=12)
            ax4.set_title('Performance Distribution (100K+ data points)', fontsize=14)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def export_results(self, filename: str = "benchmark_results.csv") -> None:
        """Export benchmark results with comprehensive metadata"""
        if not self.results.empty:
            # Add comprehensive metadata
            metadata = {
                'benchmark_version': '2.1',
                'methodology': 'warmup + 5 repeats + median',
                'timer': 'time.perf_counter',
                'python_version': sys.version.split()[0],
                'numpy_version': np.__version__,
                'platform': platform.platform(),
                'processor': platform.processor() or "unknown",
                'psutil_available': PSUTIL_AVAILABLE,
                'seed_financial': self.seed_fin,
                'seed_climate': self.seed_clim,
                'total_tests': len(self.results),
                'successful_tests': self.results['error'].isna().sum(),
                'failed_tests': self.results['error'].notna().sum(),
            }
            
            # Export results
            self.results.to_csv(filename, index=False)
            
            # Export metadata
            metadata_filename = filename.replace('.csv', '_metadata.txt')
            with open(metadata_filename, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"📁 Results exported to {filename}")
            print(f"📋 Metadata exported to {metadata_filename}")


def run_comprehensive_benchmark(max_size: int = 1_000_000, include_messages: bool = False, 
                              test_linearity: bool = False, linearity_size: int = 30_000_000):
    """Execute the professional benchmark suite"""
    
    print("🎯 VALIDATE_BRACKETS_NUMERIC PROFESSIONAL BENCHMARK")
    print("=" * 60)
    print("Professional methodology based on Python benchmarking best practices:")
    print("• Separate RNG instances for reproducible data generation") 
    print("• Warmup runs to eliminate cold-start effects")
    print("• Multiple repetitions with robust statistics (median)")
    print("• Pre-cast arrays for pure kernel performance measurement")
    print("• Error capture and statistical analysis")
    print("• Memory tracking with psutil (RSS monitoring)")
    print("• Logarithmic scaling analysis for performance characterization")
    if test_linearity:
        print(f"• High-end linearity test at {linearity_size:,} data points")
    print("=" * 60)
    
    # Initialize benchmark with professional configuration
    benchmark = PerformanceBenchmark(seed_fin=42, seed_clim=123)
    
    # Run scaling benchmark
    results_df = benchmark.run_scaling_benchmark(max_n=max_size, include_messages=include_messages)
    
    # Optional linearity test at extreme scale
    if test_linearity:
        linearity_df = benchmark.run_linearity_test(target_size=linearity_size)
        # Combine results for comprehensive analysis
        benchmark.results = pd.concat([benchmark.results, linearity_df], ignore_index=True)
    
    # Professional analysis
    benchmark.analyze_performance()
    
    # Generate visualizations
    benchmark.generate_visualizations()
    
    # Export with metadata
    benchmark.export_results("validate_brackets_professional_benchmark.csv")
    
    return benchmark


# Execute if run as main script
if __name__ == "__main__":
    # Run the professional benchmark with linearity test
    benchmark_suite = run_comprehensive_benchmark(
        max_size=1_000_000, 
        include_messages=False,
        test_linearity=True,  # Enable 30M data point linearity test
        linearity_size=30_000_000
    )
    
    print("\n" + "="*60)
    print("🏁 BENCHMARK COMPLETE")
    print("="*60)
    print("Results ready for analysis and performance optimization decisions.")
    
    if PSUTIL_AVAILABLE:
        print("📊 Memory tracking data available in results")
    else:
        print("⚠️  Install psutil for memory tracking: pip install psutil")