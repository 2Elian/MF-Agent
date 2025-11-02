# MF-Agent: Financial Knowledge Base Retrieval Agent Based on Graph Structure Topology

MF-Agent is a financial knowledge base retrieval tool whose core structure is built upon a graph topology.

**Why choose MF-Agent?**

- Powerful document parsing capabilities: The document parsing prompt can be adjusted according to your actual business needs. Only minor manual adjustments are required later to build a high-quality, clean document block.

- Multi-strategy level NER: Based on the MRV-EKR module, it can extract preliminary entity information, relationship information, and keyword information from the business line. Then, the NER model is fine-tuned to obtain high-quality entity, relationship, and keyword information.

- Graph-based QA generation: We appreciate the ideas provided by the GraphGen framework. MF-Agent, based on the LCC-KG architecture, automatically generates high-quality supervised data for subsequent tasks (e.g., embedding training data, NER training data, intent recognition model training data, etc.).

- Multi-strategy retrievers: The retriever adapts to different types of queries from the business line.

![MF-Agent Framework](resource/images/framework.png)

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Make, curl
- An API key for at least one supported LLM provider

<details>
<summary><b>Docker Setup Instructions</b> (click to expand)</summary>

#### Installing Docker

**macOS:**
```bash
# Install Docker Desktop from https://www.docker.com/products/docker-desktop/
# Or using Homebrew:
brew install --cask docker
```

**Linux (Ubuntu/Debian):**
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

#### Verifying Docker Installation
```bash
docker --version
docker compose version
```
</details>

### 30-Second Setup

```bash
git clone https://github.com/2Elian/MF-Agent.git
cd MF-Agent

# One-stop setup: creates .env, generates protobuf files
make setup

# Add your LLM API key to .env
echo "OPENAI_API_KEY=your-key-here" >> .env

# Start all services and verify
make dev
make smoke
```
