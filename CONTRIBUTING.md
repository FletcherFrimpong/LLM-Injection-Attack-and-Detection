# Contributing to SecPrompt

Thank you for your interest in contributing to SecPrompt! This document provides guidelines and information for contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** for your changes
4. **Make your changes** following the guidelines below
5. **Test your changes** thoroughly
6. **Submit a pull request**

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Installation
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/secprompt.git
cd secprompt

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests
python test_basic.py
```

## Code Style

### Python
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and under 50 lines when possible

### Example
```python
from typing import List, Dict, Optional

def process_text(text: str, options: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Process input text according to specified options.
    
    Args:
        text: The input text to process
        options: Optional processing parameters
        
    Returns:
        List of processed text segments
        
    Raises:
        ValueError: If text is empty or invalid
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    # Processing logic here
    return processed_segments
```

## Testing

### Running Tests
```bash
# Run basic functionality tests
python test_basic.py

# Run unit tests (if pytest is installed)
pytest tests/

# Run with coverage
pytest tests/ --cov=secprompt --cov-report=html
```

### Writing Tests
- Write tests for all new functionality
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies when appropriate

### Example Test
```python
import pytest
from secprompt.simulator import PromptSimulator

def test_simulator_generates_payloads():
    """Test that simulator generates the expected number of payloads."""
    simulator = PromptSimulator()
    dataset = simulator.generate_dataset(size=5, include_mutations=False)
    
    assert len(dataset) == 5
    assert all(hasattr(payload, 'content') for payload in dataset)
```

## Pull Request Guidelines

### Before Submitting
1. **Test your changes** thoroughly
2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Check code style** with tools like `black` and `flake8`

### Pull Request Template
```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Added tests for new functionality
- [ ] All existing tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented if necessary)
```

## Security Considerations

When contributing to SecPrompt, please keep in mind:

1. **Do not include sensitive data** in commits or pull requests
2. **Report security vulnerabilities** privately to the maintainers
3. **Follow responsible disclosure** practices
4. **Test security features** thoroughly before submitting

## Areas for Contribution

### High Priority
- **Detection algorithms**: Improve ML models and rule-based detection
- **Performance optimization**: Speed up processing for large datasets
- **Integration**: Connect with popular AI frameworks
- **Documentation**: Improve guides and examples

### Medium Priority
- **Dashboard features**: Add new visualizations and analysis tools
- **Testing**: Expand test coverage and add integration tests
- **CLI improvements**: Add more command options and better error handling
- **Data generation**: Create more realistic injection payloads

### Low Priority
- **Code refactoring**: Improve code organization and maintainability
- **Documentation**: Add more examples and tutorials
- **Packaging**: Improve distribution and installation

## Communication

### Issues
- Use GitHub Issues for bug reports and feature requests
- Provide detailed information when reporting bugs
- Use appropriate labels for issues

### Discussions
- Use GitHub Discussions for general questions and ideas
- Be respectful and constructive in discussions
- Help other contributors when possible

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and considerate of others
- Use inclusive language
- Focus on constructive feedback
- Help create a positive community

## Getting Help

If you need help with contributing:

1. **Check the documentation** in the README and docstrings
2. **Search existing issues** for similar questions
3. **Ask in GitHub Discussions**
4. **Open an issue** for bugs or feature requests

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- GitHub contributors page

Thank you for contributing to SecPrompt! 🛡️ 