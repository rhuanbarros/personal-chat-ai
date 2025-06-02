FROM python:3.12-bookworm

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        curl \
        build-essential \
        && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --upgrade pip \
    && pip install uv

# Install additional Python development tools
RUN uv tool install ruff \
    && uv tool install mypy \
    && uv tool install bandit \
    && uv tool install black \
    && uv tool install isort

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to non-root user
USER $USERNAME

# Set the working directory
WORKDIR /workspace

# Copy uv project files and install dependencies
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen

# Default command
CMD ["sleep", "infinity"] 