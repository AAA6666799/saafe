# Contributing to Saafe Fire Detection System

Thank you for your interest in contributing to the Saafe Fire Detection System! This document provides guidelines and information for contributors to ensure a smooth and productive collaboration.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Security Guidelines](#security-guidelines)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community Guidelines](#community-guidelines)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, or identity. We expect all participants to adhere to our code of conduct.

### Expected Behavior

- **Be Respectful**: Treat all community members with respect and courtesy
- **Be Collaborative**: Work together constructively and share knowledge
- **Be Professional**: Maintain professional communication in all interactions
- **Be Inclusive**: Welcome newcomers and help them get started
- **Be Constructive**: Provide helpful feedback and suggestions

### Unacceptable Behavior

- Harassment, discrimination, or offensive language
- Personal attacks or inflammatory comments
- Spam, trolling, or disruptive behavior
- Sharing private information without consent
- Any behavior that violates applicable laws

### Enforcement

Code of conduct violations should be reported to conduct@saafe.com. All reports will be reviewed and investigated promptly and fairly.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.9+** installed
- **Git** configured with your GitHub account
- **Docker** and Docker Compose (for containerized development)
- **AWS CLI** (for cloud-related contributions)
- Basic understanding of fire safety systems (helpful but not required)

### Development Environment Setup

1. **Fork the Repository**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/saafe.git
   cd saafe
   ```

2. **Set Up Virtual Environment**
   ```bash
   python3 -m venv saafe_env
   source saafe_env/bin/activate  # Linux/macOS
   # saafe_env\Scripts\activate   # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Configure Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Verify Installation**
   ```bash
   python -m pytest tests/
   streamlit run app.py
   ```

### Project Structure Understanding

Before contributing, familiarize yourself with:
- [Architecture Documentation](ARCHITECTURE.md)
- [File Manifest](FILE_MANIFEST.md)
- [Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md)

## Development Workflow

### Branch Strategy

We use a Git Flow-inspired branching model:

- **`main`**: Production-ready code
- **`develop`**: Integration branch for features
- **`feature/*`**: New features and enhancements
- **`bugfix/*`**: Bug fixes
- **`hotfix/*`**: Critical production fixes
- **`release/*`**: Release preparation

### Creating a Feature Branch

```bash
# Start from the latest develop branch
git checkout develop
git pull origin develop

# Create a new feature branch
git checkout -b feature/your-feature-name

# Make your changes and commit
git add .
git commit -m "feat: add new fire detection algorithm"

# Push to your fork
git push origin feature/your-feature-name
```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks
- **perf**: Performance improvements
- **security**: Security-related changes

#### Examples
```bash
feat(detection): add multi-sensor fusion algorithm
fix(ui): resolve dashboard refresh issue
docs(api): update authentication documentation
test(core): add unit tests for alert engine
security(auth): implement rate limiting
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line Length**: 88 characters (Black formatter default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Organized using isort

### Code Formatting

We use automated code formatting tools:

```bash
# Format code with Black
black saafe_mvp/ tests/

# Sort imports with isort
isort saafe_mvp/ tests/

# Lint with flake8
flake8 saafe_mvp/ tests/

# Type checking with mypy
mypy saafe_mvp/
```

### Code Quality Standards

#### Function and Class Documentation

```python
def detect_fire(sensor_data: Dict[str, float]) -> FireDetectionResult:
    """
    Analyze sensor data to detect fire conditions.
    
    Args:
        sensor_data: Dictionary containing sensor readings with keys:
            - temperature: Temperature in Celsius
            - humidity: Relative humidity percentage
            - smoke_level: Smoke concentration (ppm)
            - timestamp: Unix timestamp of reading
    
    Returns:
        FireDetectionResult containing:
            - is_fire: Boolean indicating fire detection
            - confidence: Confidence score (0.0 to 1.0)
            - risk_level: Risk level enum (LOW, MEDIUM, HIGH, CRITICAL)
            - contributing_factors: List of factors influencing detection
    
    Raises:
        ValueError: If sensor_data is missing required keys
        ValidationError: If sensor values are outside valid ranges
    
    Example:
        >>> sensor_data = {
        ...     'temperature': 45.2,
        ...     'humidity': 30.5,
        ...     'smoke_level': 150.0,
        ...     'timestamp': 1642694400
        ... }
        >>> result = detect_fire(sensor_data)
        >>> print(f"Fire detected: {result.is_fire}")
    """
```

#### Error Handling

```python
# Good: Specific exception handling
try:
    result = process_sensor_data(data)
except ValidationError as e:
    logger.error(f"Invalid sensor data: {e}")
    return ErrorResponse("Invalid sensor data format")
except ModelInferenceError as e:
    logger.error(f"Model inference failed: {e}")
    return ErrorResponse("Detection temporarily unavailable")

# Bad: Generic exception handling
try:
    result = process_sensor_data(data)
except Exception as e:
    return ErrorResponse("Something went wrong")
```

#### Logging Standards

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Processing sensor data batch")
logger.info("Fire detection completed successfully")
logger.warning("Sensor reading outside normal range")
logger.error("Failed to connect to notification service")
logger.critical("System shutdown due to critical error")

# Include context in log messages
logger.info(
    "Fire detected",
    extra={
        "sensor_id": sensor_id,
        "confidence": confidence,
        "location": location,
        "timestamp": timestamp
    }
)
```

## Testing Requirements

### Test Coverage

- **Minimum Coverage**: 85% overall
- **Critical Components**: 95% coverage required
- **New Features**: Must include comprehensive tests

### Test Types

#### Unit Tests
```python
import pytest
from saafe_mvp.core.fire_detection_pipeline import FireDetectionPipeline

class TestFireDetectionPipeline:
    def setup_method(self):
        self.pipeline = FireDetectionPipeline()
    
    def test_normal_conditions_no_fire(self):
        """Test that normal conditions don't trigger fire alert."""
        sensor_data = {
            'temperature': 22.0,
            'humidity': 45.0,
            'smoke_level': 10.0
        }
        result = self.pipeline.analyze(sensor_data)
        assert not result.is_fire
        assert result.risk_level == RiskLevel.LOW
    
    def test_fire_conditions_detected(self):
        """Test that fire conditions are properly detected."""
        sensor_data = {
            'temperature': 80.0,
            'humidity': 15.0,
            'smoke_level': 500.0
        }
        result = self.pipeline.analyze(sensor_data)
        assert result.is_fire
        assert result.confidence > 0.8
```

#### Integration Tests
```python
import pytest
from saafe_mvp.ui.dashboard import Dashboard
from saafe_mvp.core.fire_detection_pipeline import FireDetectionPipeline

class TestDashboardIntegration:
    def test_dashboard_displays_fire_alert(self):
        """Test that dashboard properly displays fire alerts."""
        # Setup test data
        fire_data = create_fire_scenario_data()
        
        # Process through pipeline
        pipeline = FireDetectionPipeline()
        result = pipeline.analyze(fire_data)
        
        # Verify dashboard display
        dashboard = Dashboard()
        dashboard.update_display(result)
        
        assert dashboard.alert_status == "FIRE_DETECTED"
        assert dashboard.risk_level == "CRITICAL"
```

#### Security Tests
```python
import pytest
from saafe_mvp.security.authentication import AuthenticationManager

class TestSecurityControls:
    def test_sql_injection_prevention(self):
        """Test that SQL injection attempts are blocked."""
        malicious_input = "'; DROP TABLE users; --"
        auth_manager = AuthenticationManager()
        
        with pytest.raises(SecurityError):
            auth_manager.authenticate_user(malicious_input, "password")
    
    def test_rate_limiting(self):
        """Test that rate limiting prevents brute force attacks."""
        auth_manager = AuthenticationManager()
        
        # Attempt multiple failed logins
        for _ in range(10):
            with pytest.raises(AuthenticationError):
                auth_manager.authenticate_user("user", "wrong_password")
        
        # Next attempt should be rate limited
        with pytest.raises(RateLimitError):
            auth_manager.authenticate_user("user", "correct_password")
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=saafe_mvp --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/security/

# Run performance tests
pytest tests/performance/ --benchmark-only
```

## Security Guidelines

### Security-First Development

All contributions must consider security implications:

#### Input Validation
```python
from pydantic import BaseModel, validator

class SensorReading(BaseModel):
    temperature: float
    humidity: float
    smoke_level: float
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not -50 <= v <= 200:
            raise ValueError('Temperature out of valid range')
        return v
    
    @validator('humidity')
    def validate_humidity(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Humidity out of valid range')
        return v
```

#### Secure Configuration
```python
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.encryption_key = os.environ.get('ENCRYPTION_KEY')
        if not self.encryption_key:
            raise ValueError("ENCRYPTION_KEY environment variable required")
    
    def get_secret(self, key: str) -> str:
        """Retrieve and decrypt secret value."""
        encrypted_value = os.environ.get(key)
        if not encrypted_value:
            raise ValueError(f"Secret {key} not found")
        
        fernet = Fernet(self.encryption_key)
        return fernet.decrypt(encrypted_value.encode()).decode()
```

#### Security Testing
```bash
# Run security scans
bandit -r saafe_mvp/
safety check
semgrep --config=auto saafe_mvp/

# Container security scanning
docker run --rm -v $(pwd):/app aquasec/trivy fs /app
```

### Vulnerability Reporting

Security vulnerabilities should be reported privately to security@saafe.com. Please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested mitigation (if any)

## Documentation Standards

### Documentation Requirements

All contributions must include appropriate documentation:

#### Code Documentation
- **Docstrings**: All public functions and classes
- **Type Hints**: All function parameters and return values
- **Comments**: Complex logic and business rules
- **Examples**: Usage examples for public APIs

#### User Documentation
- **Feature Documentation**: New features require user guide updates
- **API Documentation**: REST API changes need OpenAPI spec updates
- **Configuration**: New configuration options need documentation
- **Troubleshooting**: Common issues and solutions

#### Technical Documentation
- **Architecture Changes**: Update architecture documentation
- **Deployment Changes**: Update deployment guides
- **Security Changes**: Update security documentation
- **Performance Impact**: Document performance implications

### Documentation Format

We use Markdown for documentation with the following conventions:

```markdown
# Main Title (H1)

## Section Title (H2)

### Subsection Title (H3)

#### Detail Title (H4)

**Bold text** for emphasis
*Italic text* for variables
`Code snippets` for inline code

```python
# Code blocks with language specification
def example_function():
    return "Hello, World!"
```

> **Note**: Important information
> 
> **Warning**: Critical warnings
> 
> **Tip**: Helpful tips
```

## Pull Request Process

### Before Submitting

1. **Ensure Tests Pass**
   ```bash
   pytest
   flake8 saafe_mvp/
   mypy saafe_mvp/
   ```

2. **Update Documentation**
   - Update relevant documentation files
   - Add docstrings to new functions/classes
   - Update API documentation if applicable

3. **Security Review**
   - Run security scans
   - Review for potential vulnerabilities
   - Ensure secrets are not committed

### Pull Request Template

When creating a pull request, use this template:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Security fix
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Security tests added/updated
- [ ] All tests pass locally

## Documentation
- [ ] Code documentation updated
- [ ] User documentation updated
- [ ] API documentation updated
- [ ] Architecture documentation updated (if applicable)

## Security Checklist
- [ ] No secrets or credentials committed
- [ ] Input validation implemented
- [ ] Security tests added
- [ ] Vulnerability scan passed

## Screenshots (if applicable)
Add screenshots for UI changes.

## Additional Notes
Any additional information or context.
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automated tests
2. **Code Review**: At least one maintainer reviews the code
3. **Security Review**: Security-sensitive changes require security team review
4. **Documentation Review**: Documentation changes are reviewed for accuracy
5. **Final Approval**: Maintainer approves and merges the PR

### Review Criteria

Reviewers will evaluate:

- **Code Quality**: Follows coding standards and best practices
- **Functionality**: Meets requirements and works as expected
- **Security**: No security vulnerabilities introduced
- **Performance**: No significant performance degradation
- **Documentation**: Adequate documentation provided
- **Testing**: Comprehensive test coverage

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
Clear and concise description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python Version: [e.g. 3.9.7]
- Browser: [e.g. Chrome 96.0]
- Version: [e.g. 1.0.0]

**Additional Context**
Any other context about the problem.
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Description**
Clear and concise description of the feature.

**Problem Statement**
What problem does this feature solve?

**Proposed Solution**
Describe your proposed solution.

**Alternatives Considered**
Alternative solutions you've considered.

**Additional Context**
Any other context or screenshots.
```

### Issue Labels

We use the following labels:

- **Type**: `bug`, `feature`, `enhancement`, `documentation`
- **Priority**: `low`, `medium`, `high`, `critical`
- **Component**: `ui`, `core`, `models`, `services`, `security`
- **Status**: `needs-triage`, `in-progress`, `blocked`, `ready-for-review`

## Community Guidelines

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Q&A, ideas, and general discussion
- **Email**: Direct communication with maintainers
- **Documentation**: Wiki for community-contributed guides

### Getting Help

- **Documentation**: Check existing documentation first
- **Search Issues**: Look for similar issues or questions
- **Ask Questions**: Use GitHub Discussions for questions
- **Contact Maintainers**: Email for urgent or private matters

### Recognition

We recognize contributors through:

- **Contributors File**: All contributors are listed
- **Release Notes**: Significant contributions are highlighted
- **GitHub Recognition**: Stars, mentions, and thanks
- **Community Spotlight**: Featured contributors in documentation

## Development Resources

### Useful Tools

- **IDE**: VS Code with Python extension
- **Debugging**: Python debugger (pdb) or IDE debugger
- **Profiling**: cProfile for performance analysis
- **Documentation**: Sphinx for API documentation generation

### Learning Resources

- **Fire Safety**: Basic fire safety and detection principles
- **Machine Learning**: PyTorch and scikit-learn documentation
- **Web Development**: Streamlit documentation and tutorials
- **Cloud Platforms**: AWS documentation and best practices

### Code Examples

Check the `examples/` directory for:
- Custom sensor integration examples
- Notification service integration
- Model training and deployment
- Dashboard customization

## Maintainer Information

### Current Maintainers

- **Lead Maintainer**: engineering@saafe.com
- **Security Team**: security@saafe.com
- **Documentation Team**: docs@saafe.com

### Maintainer Responsibilities

- Review and merge pull requests
- Triage and respond to issues
- Maintain code quality standards
- Ensure security best practices
- Update documentation and guides

### Becoming a Maintainer

Regular contributors may be invited to become maintainers based on:
- Consistent high-quality contributions
- Understanding of project architecture
- Commitment to project goals
- Community involvement and support

---

Thank you for contributing to the Saafe Fire Detection System! Your contributions help make fire safety technology more accessible and effective.

**Questions?** Contact us at contribute@saafe.com

---

**Document Version**: 1.0  
**Last Updated**: January 15, 2025  
**Next Review**: February 15, 2025