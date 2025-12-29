#!/bin/bash

# SAAFE Fire Detection System - GitHub Push Script
# Repository: https://github.com/AAA6666799/saafe.git

set -e  # Exit on error

echo "🔥 SAAFE Fire Detection System - GitHub Push"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo -e "${RED}Error: README.md not found. Please run this script from the project root.${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1: Checking Git status...${NC}"
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
else
    echo -e "${GREEN}✓ Git repository already exists${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2: Checking for sensitive files...${NC}"
echo "Please review these files for sensitive information before pushing:"
echo "  - aws_login_check.py"
echo "  - bucket-policy.json"
echo "  - policy.json"
echo "  - config/ directory files"
echo ""
read -p "Have you reviewed and removed any sensitive credentials? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Please review sensitive files before pushing.${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 3: Adding files to Git...${NC}"
git add .
echo -e "${GREEN}✓ Files added (respecting .gitignore)${NC}"

echo ""
echo -e "${YELLOW}Step 4: Creating commit...${NC}"
git commit -m "Initial commit: SAAFE AI Fire Detection System

- Complete fire detection and prevention system
- AI models with 95%+ accuracy
- Real-time monitoring dashboard
- AWS deployment configurations
- Comprehensive documentation
- Multi-agent system architecture
- IoT integration support
- Global deployment ready

Excludes large datasets, model files, and sensitive credentials."

echo -e "${GREEN}✓ Commit created${NC}"

echo ""
echo -e "${YELLOW}Step 5: Checking remote repository...${NC}"
if git remote | grep -q "origin"; then
    echo "Remote 'origin' already exists. Removing..."
    git remote remove origin
fi

echo "Adding remote repository..."
git remote add origin https://github.com/AAA6666799/saafe.git
echo -e "${GREEN}✓ Remote repository added${NC}"

echo ""
echo -e "${YELLOW}Step 6: Setting main branch...${NC}"
git branch -M main
echo -e "${GREEN}✓ Branch set to 'main'${NC}"

echo ""
echo -e "${YELLOW}Step 7: Pushing to GitHub...${NC}"
echo "This may take a few minutes depending on your connection..."
echo ""

# Try to push, handle authentication
if git push -u origin main; then
    echo ""
    echo -e "${GREEN}=============================================="
    echo "✓ Successfully pushed to GitHub!"
    echo "=============================================="
    echo ""
    echo "Your repository is now available at:"
    echo "https://github.com/AAA6666799/saafe"
    echo ""
    echo "Next steps:"
    echo "1. Visit your repository on GitHub"
    echo "2. Add a description and topics"
    echo "3. Enable GitHub Pages if needed"
    echo "4. Set up branch protection rules"
    echo "5. Configure GitHub Actions for CI/CD"
    echo -e "${NC}"
else
    echo ""
    echo -e "${RED}=============================================="
    echo "Push failed. Common issues:"
    echo "=============================================="
    echo ""
    echo "1. Authentication required:"
    echo "   - Use GitHub Personal Access Token"
    echo "   - Run: git config --global credential.helper store"
    echo "   - Or use SSH: git remote set-url origin git@github.com:AAA6666799/saafe.git"
    echo ""
    echo "2. Repository not empty:"
    echo "   - Use: git push -u origin main --force (if you're sure)"
    echo ""
    echo "3. Network issues:"
    echo "   - Check your internet connection"
    echo "   - Try again in a few moments"
    echo -e "${NC}"
    exit 1
fi

echo ""
echo "📊 Repository Statistics:"
git log --oneline | wc -l | xargs echo "Commits:"
git ls-files | wc -l | xargs echo "Files tracked:"
du -sh .git | cut -f1 | xargs echo "Repository size:"
echo ""
echo -e "${GREEN}🎉 Push complete!${NC}"