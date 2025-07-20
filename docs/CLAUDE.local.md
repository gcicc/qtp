# Local Memory - Quantitative Trading Platform Setup

## Project-Specific File Organization
- Core Modules: `/src/[module]/[feature]/[class_name].py`
- Data Models: `/src/models/[domain].py`
- Utilities: `/src/utils/[category]/[utility].py`
- Types/Schemas: `/src/types/[domain].py`
- Tests: `/tests/[module]/test_[feature].py`
- Configuration: `/config/[environment].py`

## Local Development Environment

### Python Environment Setup
```bash
# Recommended Python version
python 3.11+

# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Core dependencies
pip install -r requirements.txt
```

### Required Dependencies
```python
# Core trading libraries
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Data handling
yfinance>=0.2.0
alpha_vantage>=2.3.0
pandas-ta>=0.3.14b

# Backtesting
vectorbt>=0.25.0
zipline-reloaded>=3.0.0

# Machine Learning
tensorflow>=2.13.0
torch>=2.0.0
lightgbm>=4.0.0

# Visualization
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Web framework
fastapi>=0.100.0
streamlit>=1.25.0

# Agent framework
langchain>=0.0.200
openai>=0.27.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

### Local Configuration Files

#### `.env` file template
```bash
# API Keys (never commit to git)
ALPHA_VANTAGE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Database
DATABASE_URL=sqlite:///local_trading.db
REDIS_URL=redis://localhost:6379

# Development settings
DEBUG=True
LOG_LEVEL=DEBUG
ENVIRONMENT=development
```

#### `requirements.txt`
```
# Add project-specific versions here
pandas==2.1.0
numpy==1.25.2
# ... etc
```

## Testing Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = 
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
    -v
```

### Testing Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_strategies.py

# Run tests in watch mode
pytest-watch
```

## Local Git Configuration

### .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Environment variables
.env
.env.local
.env.*.local

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
data/raw/
data/temp/
logs/
*.log

# Model artifacts
models/saved/
*.pkl
*.joblib

# Jupyter
.ipynb_checkpoints/
```

### Branch Naming Convention
```bash
# Feature branches
feature_risk_management
feature_candlestick_patterns
feature_market_research_agent

# Bug fixes
bugfix_data_pipeline_memory_leak
bugfix_backtest_lookahead_bias

# Hot fixes
hotfix_critical_execution_bug
```

## IDE Setup Recommendations

### VS Code Extensions
- Python
- Pylance
- Python Docstring Generator
- GitLens
- Jupyter
- Thunder Client (for API testing)

### VS Code Settings (settings.json)
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "files.autoSave": "afterDelay"
}
```

## Local Development Workflow

### Daily Workflow
1. Pull latest changes from main
2. Create feature branch: `git checkout -b feature_new_functionality`
3. Write tests first (TDD approach)
4. Implement functionality
5. Run tests: `pytest`
6. Check code quality: `pylint src/`
7. Format code: `black src/`
8. Commit changes with descriptive message
9. Push to feature branch

### Code Quality Checks
```bash
# Linting
pylint src/

# Formatting
black src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/

# Import sorting
isort src/ tests/
```

## Local Database Setup

### SQLite (Development)
```python
# config/development.py
DATABASE_CONFIG = {
    'url': 'sqlite:///data/trading_dev.db',
    'echo': True  # Enable SQL logging
}
```

### Redis (Caching)
```bash
# Install Redis locally
# macOS: brew install redis
# Ubuntu: sudo apt-get install redis-server

# Start Redis
redis-server
```

## Local Data Sources

### Free Data APIs
- **Yahoo Finance**: yfinance library
- **Alpha Vantage**: Free tier available
- **Federal Reserve Economic Data (FRED)**: Economic indicators
- **Quandl**: Financial and economic data

### Data Storage Structure
```
data/
├── raw/           # Original downloaded data
├── processed/     # Cleaned and processed data
├── features/      # Engineered features
├── models/        # Trained model artifacts
└── exports/       # Analysis outputs
```

## Agent Development Environment

### Local Agent Testing
```python
# Mock agent for testing
class MockMarketResearchAgent:
    def __init__(self):
        self.responses = {
            'market_regime': 'consolidation',
            'confidence': 0.75
        }
    
    def analyze_market(self, data):
        return self.responses
```

### Agent Configuration
```python
# config/agents.py
AGENT_CONFIG = {
    'market_research': {
        'enabled': True,
        'update_frequency': '1H',
        'data_sources': ['yfinance', 'alpha_vantage']
    },
    'risk_monitor': {
        'enabled': True,
        'alert_thresholds': {
            'var_limit': 0.05,
            'correlation_threshold': 0.8
        }
    }
}
```

## Personal Notes & Reminders

### Development Priorities
1. Focus on statistical rigor over complexity
2. Always include explanations for model decisions
3. Educational value should be clear in every feature
4. Test edge cases thoroughly (missing data, extreme market conditions)

### Key Learning Areas for This Project
- Advanced time series analysis
- Market microstructure modeling
- Agent-based systems design
- Financial ML best practices
- Real-time system architecture

### Debugging Tips
- Use `pdb.set_trace()` for interactive debugging
- Log all agent decisions with timestamps
- Keep test data small but representative
- Mock external APIs during development
- Use visualization for data validation

### Performance Monitoring
```python
# Local performance tracking
import time
import functools

def time_it(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper
```

## Backup and Recovery

### Local Backup Strategy
- Git commits serve as code backup
- Database dumps before major changes
- Model artifacts versioned and backed up
- Configuration files tracked in separate branch

### Recovery Procedures
```bash
# Reset to clean state
git stash
git checkout main
git pull origin main

# Restore database from backup
cp data/backups/trading_backup.db data/trading_dev.db

# Rebuild environment
pip install -r requirements.txt
```