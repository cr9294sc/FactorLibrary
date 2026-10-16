# Rust Engine Module Structure

## File Organization

### `lib.rs`
Main entry point for the Rust engine module. Contains:
- Backtest engine core (`BacktestEngine`, `BacktestConfig`)
- Strategy execution logic
- Vectorized indicators (`compute_sma`, `compute_rsi`)
- Factor backtesting functions

### `database.rs`
High-performance database and K-line synthesis module. Contains:
- K-line resampling (`resample_klines`)
- Period conversion utilities
- Datetime parsing and rounding
- OHLCV aggregation logic

## Module Usage

### From Python

```python
from engine_rust import resample_klines

# Resample K-lines using Rust (high performance)
bars_15m = resample_klines(bars_1m, "15m")
```

### From Rust

```rust
use engine_rust::database::resample_klines_rust;

let resampled = resample_klines_rust(bars, "15m")?;
```

## Performance Benefits

- **10-50x faster** than Python for large datasets (>10k bars)
- Zero-copy data structures
- Vectorized operations
- Native datetime parsing

## Adding New Modules

To add a new module:

1. Create `src/new_module.rs`
2. Add to `lib.rs`:
   ```rust
   mod new_module;
   pub use new_module::*;
   ```
3. Export functions in `#[pymodule]` if needed

