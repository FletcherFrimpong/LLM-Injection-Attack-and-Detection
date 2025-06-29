# GitHub Repository Setup Guide

This guide will help you set up the SecPrompt repository on GitHub.

## Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the repository details:
   - **Repository name**: `secprompt`
   - **Description**: `A comprehensive framework for detecting and defending against prompt injection attacks`
   - **Visibility**: Choose Public or Private
   - **Initialize with**: Don't initialize (we already have files)
5. Click "Create repository"

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands to connect your local repository. Run these commands:

```bash
# Add the remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/secprompt.git

# Set the main branch as upstream
git branch -M main

# Push the initial commit
git push -u origin main
```

## Step 3: Configure Repository Settings

### Repository Settings
1. Go to your repository on GitHub
2. Click "Settings" tab
3. Configure the following:

#### General Settings
- **Repository name**: `secprompt`
- **Description**: `A comprehensive framework for detecting and defending against prompt injection attacks`
- **Website**: (optional) Add your website if you have one
- **Topics**: Add relevant topics like:
  - `security`
  - `ai-safety`
  - `prompt-injection`
  - `machine-learning`
  - `python`
  - `nlp`

#### Features
- ✅ **Issues**: Enable
- ✅ **Discussions**: Enable
- ✅ **Wiki**: Enable (optional)
- ✅ **Sponsorships**: Enable (optional)

#### Pages (Optional)
- **Source**: Deploy from a branch
- **Branch**: `main`
- **Folder**: `/docs` (if you create documentation pages)

## Step 4: Set Up Repository Structure

### Create Issue Templates
Create `.github/ISSUE_TEMPLATE/` directory and add templates:

#### Bug Report Template
```markdown
---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: ['bug']
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment:**
 - OS: [e.g. macOS, Ubuntu]
 - Python version: [e.g. 3.9]
 - SecPrompt version: [e.g. 0.1.0]

**Additional context**
Add any other context about the problem here.
```

#### Feature Request Template
```markdown
---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: ['enhancement']
assignees: ''
---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions.

**Additional context**
Add any other context or screenshots about the feature request here.
```

### Create Pull Request Template
Create `.github/pull_request_template.md`:

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

## Step 5: Set Up GitHub Actions (Optional)

Create `.github/workflows/ci.yml` for continuous integration:

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python test_basic.py
    
    - name: Run linting
      run: |
        pip install flake8 black
        flake8 secprompt/
        black --check secprompt/
```

## Step 6: Create Release

### First Release
1. Go to "Releases" in your repository
2. Click "Create a new release"
3. Set tag version: `v0.1.0`
4. Title: `SecPrompt v0.1.0 - Initial Release`
5. Description:
```markdown
## 🎉 Initial Release

SecPrompt is a comprehensive framework for detecting and defending against prompt injection attacks.

### Features
- 🔍 **Detection**: Rule-based and ML-based injection detection
- 📊 **Evaluation**: Severity assessment and impact analysis
- 🛡️ **Defenses**: Input sanitization and prompt hardening
- 🧪 **Simulation**: Generate realistic injection payloads
- 🖥️ **Dashboard**: Interactive web interface
- 📝 **CLI**: Command-line interface for automation

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/secprompt.git
cd secprompt
./install.sh
```

### Quick Start
```bash
# Test functionality
python3 test_basic.py

# Generate payloads
python3 main.py generate --size 10

# Detect injection
python3 main.py detect --text "Ignore all previous instructions"
```

### Documentation
- [README.md](README.md) - Comprehensive guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [STATUS.md](STATUS.md) - Project status
```

## Step 7: Set Up Branch Protection (Recommended)

1. Go to Settings > Branches
2. Add rule for `main` branch:
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Include administrators

## Step 8: Add Repository Badges

Add these badges to your README.md:

```markdown
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](test_basic.py)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://github.com/YOUR_USERNAME/secprompt/releases)
```

## Step 9: Enable GitHub Features

### Discussions
1. Go to repository settings
2. Enable Discussions
3. Create categories:
   - **General**: General questions and discussions
   - **Ideas**: Feature requests and ideas
   - **Show and tell**: Share projects and implementations
   - **Q&A**: Questions and answers

### Wiki
1. Enable Wiki in repository settings
2. Create initial pages:
   - Home
   - Installation Guide
   - API Reference
   - Troubleshooting

## Step 10: Share Your Repository

### Social Media
- Share on Twitter, LinkedIn, Reddit
- Use hashtags: #AISafety #PromptInjection #Security #Python

### Communities
- Post in relevant Discord/Slack channels
- Share in Python and security communities
- Submit to GitHub trending repositories

### Documentation Sites
- Consider adding to PyPI for easy installation
- Create documentation site with GitHub Pages
- Add to Awesome AI Safety lists

## Next Steps

After setting up the repository:

1. **Monitor issues and pull requests**
2. **Respond to community feedback**
3. **Plan future releases**
4. **Build community around the project**
5. **Consider adding more features based on feedback**

## Repository URL

Once set up, your repository will be available at:
```
https://github.com/YOUR_USERNAME/secprompt
```

Replace `YOUR_USERNAME` with your actual GitHub username.

---

🎉 Congratulations! Your SecPrompt repository is now ready for the open source community! 🛡️ 