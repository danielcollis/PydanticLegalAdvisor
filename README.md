# Legal Case Analyzer

## Overview
The Legal Case Analyzer is an AI-powered application that simulates a legal proceeding using multiple large language models to analyze case details and predict outcomes. The system employs a multi-agent approach where different AI models play specific roles: defense attorney, prosecutor, and judge/jury.

## Features
- Multi-agent legal analysis with specialized AI roles
- Support for various case types (criminal, civil, family, intellectual property)
- Comprehensive argument generation from both defense and prosecution perspectives
- Structured verdict output with confidence levels and sentencing recommendations
- User-friendly Streamlit interface with customizable case parameters

## Technical Architecture
The application uses:
- Streamlit for the web interface
- Multiple LLM models through the Pydantic-AI framework:
  - Deepseek (via Ollama) for defense arguments
  - GPT-4o Mini for prosecution arguments
  - GPT-4o for impartial verdict determination
- Structured outputs using Pydantic models
- Environment variable configuration for API keys

## Installation

### Prerequisites
- Python 3.8+
- An OpenAI API key
- Ollama (for local model inference)

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/legal-case-analyzer.git
   cd legal-case-analyzer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   OLLAMA_BASE_URL=http://localhost:11434/v1
   ```

## Usage
1. Start the Streamlit application:
   ```
   streamlit run legalAnalyzer2.py
   ```

2. Enter your case details in the sidebar:
   - Select a case type
   - Enter defendant and plaintiff information
   - Choose jurisdiction
   - Provide case facts and background
   - Set evidence strength

3. Click "Analyze Case" to generate:
   - Defense arguments
   - Prosecution arguments
   - Final verdict with reasoning and sentencing recommendations

## Customization
- Models can be configured in the LLM Settings expander
- Case types and parameters can be modified in the code
- Prompts for each agent can be adjusted for different legal contexts

## Limitations
- This tool is for educational and demonstration purposes only
- It is not a substitute for professional legal advice
- Results depend on the quality of input data and underlying models
- Models may not have full knowledge of specific jurisdictional laws

## Future Enhancements
- Support for more specialized legal domains
- Addition of case precedent lookup
- Document upload for case facts
- Jurisdiction-specific legal frameworks
- Collaborative mode for multiple users

## License
[MIT License]

## Disclaimer
This tool provides automated legal analysis for educational purposes only. It is not a substitute for professional legal advice. The predictions and analyses generated should not be used for making actual legal decisions.
