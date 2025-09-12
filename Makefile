# PDF Estimator - Simplified Makefile for Job Execution
# Clean and focused automation for structural analysis jobs

# Project configuration
PROJECT_NAME := pdf-estimator
VERSION := 2.0.0

# Docker configuration
DOCKER_COMPOSE := $(shell command -v docker-compose >/dev/null 2>&1 && echo "docker-compose" || echo "docker compose")

# Colors for output
RESET := \033[0m
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
BOLD := \033[1m

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# HELP
# =============================================================================

.PHONY: help
help: ## 📋 Show available commands
	@echo -e "$(BOLD)$(BLUE)PDF Estimator v$(VERSION) - Job Execution$(RESET)"
	@echo -e "$(BLUE)============================================$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' | \
		sort
	@echo ""
	@echo -e "$(YELLOW)💡 Quick Start:$(RESET)"
	@echo -e "  1. Place your PDF in input/file.pdf"
	@echo -e "  2. Run 'make job' for complete structural analysis"
	@echo -e "  3. Check results in output/ directory"

# =============================================================================
# CORE JOB EXECUTION
# =============================================================================

.PHONY: job
job: check-config ## 🏗️ Execute comprehensive structural analysis (MAIN COMMAND)
	@echo -e "$(BLUE)🏗️ Starting comprehensive structural analysis...$(RESET)"
	@echo -e "$(YELLOW)⏱️  This will take 5-10 minutes for complete analysis$(RESET)"
	$(DOCKER_COMPOSE) run --rm pdf-estimator python -m src.cli analyze /app/input/file.pdf --analysis-type comprehensive

.PHONY: job-quick
job-quick: check-config ## ⚡ Quick PDF analysis (faster, less detailed)
	@echo -e "$(BLUE)⚡ Quick PDF analysis...$(RESET)"
	$(DOCKER_COMPOSE) run --rm pdf-estimator python -m src.cli analyze /app/input/file.pdf --analysis-type general

# job-yaml removed - use job or job-quick for standard analysis

.PHONY: chat
chat: check-config ## 💬 Interactive chat with PDF
	@echo -e "$(BLUE)💬 Starting interactive chat...$(RESET)"
	$(DOCKER_COMPOSE) run --rm pdf-estimator python -m src.cli chat /app/input/file.pdf

# =============================================================================
# PROJECT MANAGEMENT
# =============================================================================

.PHONY: setup
setup: check-system create-dirs setup-env ## 🚀 Initial project setup
	@echo -e "$(GREEN)✅ Project setup completed!$(RESET)"
	@echo -e "$(YELLOW)Next steps:$(RESET)"
	@echo -e "  1. Place your PDF in input/file.pdf"
	@echo -e "  2. Run 'make job' for structural analysis"

.PHONY: check-system
check-system: ## 🔍 Check system requirements
	@echo -e "$(BLUE)🔍 Checking system requirements...$(RESET)"
	@command -v docker >/dev/null 2>&1 || (echo -e "$(RED)❌ Docker not found$(RESET)" && exit 1)
	@command -v $(firstword $(DOCKER_COMPOSE)) >/dev/null 2>&1 || (echo -e "$(RED)❌ Docker Compose not found$(RESET)" && exit 1)
	@docker info >/dev/null 2>&1 || (echo -e "$(RED)❌ Docker daemon not running$(RESET)" && exit 1)
	@echo -e "$(GREEN)✅ System requirements satisfied$(RESET)"

.PHONY: create-dirs
create-dirs: ## 📁 Create project directories
	@echo -e "$(BLUE)📁 Creating project directories...$(RESET)"
	@mkdir -p input output logs temp
	@echo -e "$(GREEN)✅ Directories created$(RESET)"

.PHONY: setup-env
setup-env: ## 🔧 Setup environment files
	@echo -e "$(BLUE)🔧 Setting up environment...$(RESET)"
	@if [ ! -f .env ]; then \
		echo "GEMINI_API_KEY=your_api_key_here" > .env; \
		echo "CONTAINER=false" >> .env; \
		echo "LOG_LEVEL=INFO" >> .env; \
		echo -e "$(GREEN)✅ .env created$(RESET)"; \
		echo -e "$(YELLOW)⚠️  Please edit .env and set your GEMINI_API_KEY$(RESET)"; \
	else \
		echo -e "$(GREEN)✅ .env already exists$(RESET)"; \
	fi

.PHONY: check-config
check-config: ## ✅ Check configuration
	@if [ ! -f .env ]; then \
		echo -e "$(RED)❌ .env file not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@if grep -q "your_api_key_here" .env 2>/dev/null; then \
		echo -e "$(YELLOW)⚠️  Please configure your GEMINI_API_KEY in .env$(RESET)"; \
	fi

# =============================================================================
# DOCKER OPERATIONS
# =============================================================================

.PHONY: build
build: ## 🔨 Build Docker images
	@echo -e "$(BLUE)🔨 Building Docker images...$(RESET)"
	$(DOCKER_COMPOSE) build --no-cache
	@echo -e "$(GREEN)✅ Docker images built$(RESET)"

.PHONY: shell
shell: ## 🐚 Open shell in container
	@echo -e "$(BLUE)🐚 Opening container shell...$(RESET)"
	$(DOCKER_COMPOSE) run --rm pdf-estimator bash

# =============================================================================
# UTILITIES
# =============================================================================

.PHONY: results
results: ## 📊 Show recent analysis results
	@echo -e "$(BLUE)📊 Recent Analysis Results:$(RESET)"
	@echo -e "$(BLUE)========================$(RESET)"
	@ls -la output/ 2>/dev/null || echo "No results found"

.PHONY: clean
clean: ## 🧹 Clean temporary files
	@echo -e "$(YELLOW)🧹 Cleaning temporary files...$(RESET)"
	@rm -rf temp/* logs/*.log
	@$(DOCKER_COMPOSE) down --remove-orphans 2>/dev/null || true
	@echo -e "$(GREEN)✅ Cleanup completed$(RESET)"

.PHONY: clean-all
clean-all: clean ## 🗑️ Deep clean (includes results)
	@echo -e "$(RED)🗑️ Deep cleaning (this will remove all results!)$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf output/* temp/* logs/*; \
		docker system prune -f 2>/dev/null || true; \
		echo -e "$(GREEN)✅ Deep clean completed$(RESET)"; \
	else \
		echo -e "$(YELLOW)❌ Deep clean cancelled$(RESET)"; \
	fi

.PHONY: status
status: ## 📈 Show project status
	@echo -e "$(BOLD)$(CYAN)Project Status$(RESET)"
	@echo -e "$(CYAN)==============$(RESET)"
	@echo -e "PDFs in input:     $(shell ls input/*.pdf 2>/dev/null | wc -l)"
	@echo -e "Results in output: $(shell ls output/*.json 2>/dev/null | wc -l)"
	@echo -e "Config file:       $(shell [ -f config.toml ] && echo "✅" || echo "❌")"
	@echo -e "Environment file:  $(shell [ -f .env ] && echo "✅" || echo "❌")"
	@echo -e "Docker running:    $(shell docker info >/dev/null 2>&1 && echo "✅" || echo "❌")"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Prevent deletion of intermediate files
.PRECIOUS: .env config.toml

# Declare phony targets
.PHONY: help job job-quick chat setup check-system create-dirs setup-env check-config \
        build shell results clean clean-all status