import numpy as np
import time
import platform
import sys
import sympy as sp
from typing import Dict, List, Tuple, Optional
from itertools import product
import matplotlib.pyplot as plt

# Import the actual SympyFamily class
from rapidroots.families.sympy_family import SympyFamily


class SympyFamilyBenchmark:
    """Scientific benchmark suite for SympyFamily performance analysis"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.results = []
        
    def create_test_functions(self) -> Dict[str, Tuple[sp.Expr, Tuple[sp.Symbol, ...]]]:
        """Create test functions representing financial and climate models"""
        x = sp.Symbol('x')
        
        # Financial models
        a, b, c = sp.symbols('a b c')
        k, sigma, r = sp.symbols('k sigma r')
        mu, theta = sp.symbols('mu theta')
        
        # Climate models  
        alpha, beta, gamma = sp.symbols('alpha beta gamma')
        
        functions = {
            # Financial: Black-Scholes option pricing components
            'black_scholes': (a * sp.exp(-r * x) * sp.erf(sigma * sp.sqrt(x)), (a, r, sigma)),
            
            # Financial: Interest rate models (Vasicek-like)
            'interest_rate': (a * sp.exp(-b * x) + c * (1 - sp.exp(-b * x)), (a, b, c)),
            
            # Financial: Volatility surface
            'volatility_surface': (k * sp.sqrt(x) * sp.exp(-sigma * x), (k, sigma)),
            
            # Climate: Temperature anomaly model
            'temperature_model': (alpha * sp.sin(2 * sp.pi * x / 365) + beta * x + gamma, (alpha, beta, gamma)),
            
            # Climate: Carbon cycle model  
            'carbon_cycle': (a * sp.exp(-b * x) + c * sp.log(1 + x), (a, b, c)),
            
            # Climate: Sea level rise
            'sea_level': (mu * x + theta * x**2, (mu, theta)),
        }
        
        return functions
    
    def generate_realistic_data(self, n: int, model_type: str, param_count: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic data for financial and climate models"""
        if model_type == 'financial':
            # Financial data: time in years, parameters for market conditions
            x_data = self.rng.uniform(0, 10, n)  # 0-10 years
            
            if param_count == 2:
                params = np.column_stack([
                    self.rng.uniform(0.1, 2.0, n),   # volatility/growth parameter
                    self.rng.uniform(0.01, 0.15, n)  # rate parameter
                ])
            else:  # param_count == 3
                params = np.column_stack([
                    self.rng.uniform(50, 150, n),     # asset price
                    self.rng.uniform(0.01, 0.15, n), # interest rate
                    self.rng.uniform(0.1, 0.5, n)    # volatility
                ])
        
        elif model_type == 'climate':
            # Climate data: time in days/years, parameters for environmental conditions
            x_data = self.rng.uniform(0, 365*5, n)  # 5 years in days
            
            if param_count == 2:
                params = np.column_stack([
                    self.rng.uniform(0.5, 5.0, n),   # amplitude parameter
                    self.rng.uniform(0.001, 0.01, n) # decay/growth rate
                ])
            else:  # param_count == 3
                params = np.column_stack([
                    self.rng.uniform(-5, 5, n),      # seasonal amplitude
                    self.rng.uniform(-0.01, 0.01, n), # trend coefficient
                    self.rng.uniform(-2, 2, n)       # offset
                ])
        
        return x_data, params
    
    def benchmark_single_configuration(self, family: SympyFamily, function_name: str, 
                                      model_type: str, size: int, dtype: np.dtype, 
                                      derivative: int, x_data: np.ndarray, params_data: np.ndarray, 
                                      repeats: int = 5) -> Optional[Dict]:
        """Benchmark a single configuration (size, dtype, derivative)"""
        
        # Warmup run (not measured)
        try:
            family.evaluate(x_data[:100], params_data[:100], derivative=derivative, dtype=dtype)
        except Exception:
            return None
        
        # Benchmark runs
        timings = []
        result = None
        
        for _ in range(repeats):
            start_time = time.perf_counter()
            try:
                result = family.evaluate(x_data, params_data, derivative=derivative, dtype=dtype)
                end_time = time.perf_counter()
                timings.append((end_time - start_time) * 1000)  # Convert to ms
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                return None
        
        if not timings or result is None:
            return None
        
        # Use median for robust timing, std for variability
        median_time = np.median(timings)
        std_time = np.std(timings, ddof=1) if len(timings) > 1 else 0.0
        throughput = size / (median_time / 1000.0)  # items/second
        dtype_correct = result.dtype == dtype
        
        return {
            'function': function_name,
            'model_type': model_type,
            'size': size,
            'dtype': str(dtype),
            'derivative': derivative,
            'param_count': family.param_count,
            'median_time_ms': median_time,
            'std_time_ms': std_time,
            'throughput': throughput,
            'dtype_correct': dtype_correct,
            'memory_mb': (x_data.nbytes + params_data.nbytes + result.nbytes) / (1024**2)
        }

    def benchmark_function(self, function_name: str, expr: sp.Expr, param_syms: Tuple[sp.Symbol, ...], 
                          test_configurations: List[Tuple]) -> List[Dict]:
        """Benchmark a single function across all test configurations"""
        
        print(f"\n🔬 Benchmarking {function_name}")
        print("-" * 50)
        
        # Determine model type for realistic data generation
        model_type = 'financial' if any(name in function_name for name in ['black_scholes', 'interest', 'volatility']) else 'climate'
        
        # Create function family
        family = SympyFamily(function_name, expr, param_syms)
        function_results = []
        
        # Group configurations by size for efficient data generation
        configs_by_size = {}
        for size, dtype, derivative in test_configurations:
            if size not in configs_by_size:
                configs_by_size[size] = []
            configs_by_size[size].append((dtype, derivative))
        
        for size in sorted(configs_by_size.keys()):
            print(f"  Size: {size:,} points")
            
            # Generate realistic test data once per size
            x_data, params_data = self.generate_realistic_data(size, model_type, len(param_syms))
            
            # Test all dtype/derivative combinations for this size
            for dtype, derivative in configs_by_size[size]:
                result = self.benchmark_single_configuration(
                    family, function_name, model_type, size, dtype, derivative, 
                    x_data, params_data, repeats=5
                )
                
                if result:
                    function_results.append(result)
                    print(f"    {dtype.__name__} deriv={derivative}: {result['median_time_ms']:.2f}ms, "
                          f"{result['throughput']:.0f} items/s, dtype_ok={result['dtype_correct']}")
        
        return function_results
    
    def run_comprehensive_benchmark(self, max_size: int = 1_000_000) -> None:
        """Run comprehensive benchmark following scientific best practices"""
        
        print("🚀 SYMPYFAMILY SCIENTIFIC PERFORMANCE BENCHMARK")
        print("=" * 60)
        print("Methodology based on scientific computing best practices:")
        print("• Multiple repetitions with median timing for robustness")
        print("• Warmup runs to eliminate JIT compilation overhead") 
        print("• Realistic financial and climate model test cases")
        print("• dtype correctness verification")
        print("• Scaling analysis for performance characterization")
        print("=" * 60)
        
        # Test configurations following scientific methodology
        sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
        if max_size >= 1_000_000:
            sizes.append(1_000_000)
        
        dtypes = [np.float32, np.float64]
        derivatives = [0, 1, 2]  # Function, first derivative, second derivative
        
        # Create all test configurations using itertools.product (parameterized approach)
        test_configurations = list(product(sizes, dtypes, derivatives))
        print(f"Total configurations per function: {len(test_configurations)}")
        
        # Get test functions
        test_functions = self.create_test_functions()
        
        # Benchmark each function with all configurations
        for func_name, (expr, param_syms) in test_functions.items():
            try:
                results = self.benchmark_function(func_name, expr, param_syms, test_configurations)
                self.results.extend(results)
                
            except Exception as e:
                print(f"❌ Failed to benchmark {func_name}: {e}")
                continue
        
        print(f"\n✅ Benchmark completed! {len(self.results)} successful measurements")
    
    def analyze_performance(self) -> None:
        """Analyze benchmark results following scientific methodology"""
        
        if not self.results:
            print("No results to analyze")
            return
            
        print("\n" + "=" * 60)
        print("📊 SCIENTIFIC PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        import pandas as pd
        df = pd.DataFrame(self.results)
        
        # Performance at 100K data points (reference size)
        ref_size = 100_000
        ref_data = df[df['size'] == ref_size]
        
        if not ref_data.empty:
            print(f"\n🎯 PERFORMANCE AT {ref_size:,} DATA POINTS:")
            by_config = ref_data.groupby(['function', 'dtype', 'derivative']).agg({
                'median_time_ms': ['mean', 'std'],
                'throughput': 'mean',
                'dtype_correct': 'all'
            }).round(2)
            print(by_config)
        
        # Scaling analysis - check linearity
        print(f"\n📏 SCALING CHARACTERISTICS:")
        large_data = df[df['size'] >= 50_000]
        
        for func in df['function'].unique():
            func_data = large_data[large_data['function'] == func]
            if len(func_data) > 3:
                corr = np.corrcoef(func_data['size'], func_data['median_time_ms'])[0,1]
                print(f"  {func}: size vs time correlation = {corr:.3f}")
                
                if corr > 0.95:
                    print(f"    ✅ Excellent linear scaling")
                elif corr > 0.85:
                    print(f"    ✅ Good scaling")
                else:
                    print(f"    ⚠️  Non-linear scaling detected")
        
        # Data type performance comparison
        print(f"\n🔬 DTYPE PERFORMANCE COMPARISON:")
        dtype_perf = df.groupby(['dtype', 'derivative']).agg({
            'median_time_ms': 'mean',
            'throughput': 'mean',
            'dtype_correct': 'mean'
        }).round(2)
        print(dtype_perf)
        
        # Model type analysis
        print(f"\n🌍 MODEL TYPE ANALYSIS:")
        model_perf = df.groupby(['model_type', 'derivative']).agg({
            'median_time_ms': 'mean',
            'throughput': 'mean'
        }).round(2)
        print(model_perf)
    
    def generate_visualizations(self) -> None:
        """Generate scientific performance visualizations"""
        
        if not self.results:
            print("No results to visualize")
            return
            
        import pandas as pd
        df = pd.DataFrame(self.results)
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SympyFamily Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Scaling performance (log-log)
        ax1 = axes[0, 0]
        for dtype in df['dtype'].unique():
            for deriv in [0, 1, 2]:
                data = df[(df['dtype'] == dtype) & (df['derivative'] == deriv)]
                if not data.empty:
                    scaling_data = data.groupby('size')['median_time_ms'].mean()
                    ax1.loglog(scaling_data.index, scaling_data.values, 'o-', 
                              label=f'{dtype} deriv={deriv}', alpha=0.8, linewidth=2, markersize=4)
        
        ax1.set_xlabel('Data Points (n)')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Scaling Performance')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Throughput comparison
        ax2 = axes[0, 1]
        throughput_data = df.groupby(['function', 'dtype'])['throughput'].mean().unstack()
        throughput_data.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_xlabel('Function')
        ax2.set_ylabel('Throughput (items/second)')
        ax2.set_title('Throughput by Function and Dtype')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Dtype')
        
        # 3. Derivative computation overhead
        ax3 = axes[1, 0]
        deriv_overhead = df.groupby(['function', 'derivative'])['median_time_ms'].mean().unstack()
        deriv_overhead.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_xlabel('Function')
        ax3.set_ylabel('Execution Time (ms)')
        ax3.set_title('Derivative Computation Overhead')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Derivative Order')
        
        # 4. Memory vs Performance
        ax4 = axes[1, 1]
        for model_type in df['model_type'].unique():
            data = df[df['model_type'] == model_type]
            ax4.scatter(data['memory_mb'], data['throughput'], 
                       label=model_type.capitalize(), alpha=0.7, s=50)
        
        ax4.set_xlabel('Memory Usage (MB)')
        ax4.set_ylabel('Throughput (items/second)')
        ax4.set_title('Memory vs Performance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filename: str = "sympyfamily_benchmark.csv") -> None:
        """Export results with metadata"""
        
        if not self.results:
            print("No results to export")
            return
            
        import pandas as pd
        df = pd.DataFrame(self.results)
        
        # Add metadata
        metadata = {
            'benchmark_version': '1.0',
            'methodology': 'warmup + 5 repeats + median timing',
            'timer': 'time.perf_counter',
            'python_version': sys.version.split()[0],
            'numpy_version': np.__version__,
            'platform': platform.platform(),
            'seed': self.seed,
            'total_measurements': len(self.results)
        }
        
        # Export results
        df.to_csv(filename, index=False)
        
        # Export metadata
        metadata_filename = filename.replace('.csv', '_metadata.txt')
        with open(metadata_filename, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print(f"📁 Results exported to {filename}")
        print(f"📋 Metadata exported to {metadata_filename}")


def run_benchmark(max_size: int = 1_000_000):
    """Execute the scientific benchmark suite"""
    
    benchmark = SympyFamilyBenchmark(seed=42)
    benchmark.run_comprehensive_benchmark(max_size=max_size)
    benchmark.analyze_performance()
    benchmark.generate_visualizations()
    benchmark.export_results()
    
    return benchmark


if __name__ == "__main__":
    # Run the benchmark
    print("Starting SympyFamily performance benchmark...")
    benchmark_suite = run_benchmark(max_size=1_000_000)
    print("\n🏁 Benchmark complete!")