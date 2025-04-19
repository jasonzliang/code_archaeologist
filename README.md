# 🏺 Code Archaeologist 🏺

Code Archaeologist is an advanced Git repository analysis tool that combines the power of AI (GPT-4) with data visualization to provide deep insights into your codebase's evolution, team dynamics, and development patterns.

## Features

### 1. Repository Analysis
- Real-time analysis of Git repositories (local or remote)
- Option to analyze all commits or limit to K most recent commits
- Automatic code complexity calculation
- Technical debt tracking
- Breaking changes detection
- Security impact assessment

### 2. Interactive Dashboard
- **Overview Tab**
  - Key repository metrics
  - Commit activity heatmap
  - Development velocity trends
  - Active contributor statistics

- **Code Insights Tab**
  - Technical debt metrics
  - Code complexity evolution
  - File impact analysis
  - Breaking changes tracking

- **Team Analysis Tab**
  - Collaboration network visualization
  - Knowledge distribution maps
  - Developer impact levels
  - File ownership patterns

- **Interactive Q&A Tab**
  - Natural language queries about your repository
  - AI-powered analysis with visualizations
  - Example questions provided for guidance

- **Custom Analysis Tab**
  - Customizable visualizations
  - Flexible data filtering
  - Multiple chart types
  - Export capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/code_archaeologist.git
cd code_archaeologist
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

The following installation steps are optional and enable code generation using LLMs:

4. Create a "code_generation.yaml" file with provided template. Install [AutoGen](https://github.com/jasonzliang/ag2) and [MetaGPT](https://github.com/jasonzliang/MetaGPT). Make sure to update agent_model, builder_model, and metagpt_path to reflect your local configuration.

5. Create an AutoGen config file located at ~/.autogen/OAI_CONFIG_LIST (see [AutoGen documentation](https://microsoft.github.io/autogen/0.2/docs/topics/llm_configuration) for details).

## Dependencies

- Python 3.8+
- OpenAI API key (for commit analysis)
- Required Python packages:
  - streamlit
  - pydriller
  - openai
  - plotly
  - networkx
  - pandas
  - numpy
  - matplotlib
  - wordcloud

## Usage

1. Start the Streamlit app:
```bash
streamlit run code_archaeologist.py
```

2. Enter your repository path (local or remote Git URL)

3. Optional: Set commit limit for analysis

4. Explore the different tabs:
   - Navigate through Overview, Code Insights, Team Analysis
   - Ask questions in the Interactive Q&A tab
   - Create custom visualizations in the Custom Analysis tab

## Features in Detail

### AI-Powered Analysis
The tool uses GPT-4 to analyze:
- Architectural patterns and decisions
- Technical debt introduction/resolution
- Breaking changes and API modifications
- Security implications
- Code quality trends
- Team dynamics
- Development velocity
- Knowledge distribution

### Visualization Types
- Timeline charts
- Bar charts
- Pie charts
- Scatter plots
- Network graphs
- Heatmaps
- Sunburst diagrams
- Treemaps
- Radar charts
- Sankey diagrams
- Word clouds
- Bubble charts
- Violin plots
- Funnel charts

### Cache Management
- Results are cached for performance
- Clear cache button available for fresh analysis
- Automatic cache invalidation on parameter changes

## Example Questions for Q&A

- "Show me the complexity trend for critical files"
- "What's the knowledge distribution across the team?"
- "Analyze the correlation between technical debt and breaking changes"
- "Which files have the most frequent security-related commits?"
- "Generate a development velocity report for Q1"

## Customization

### Visualization Controls
- Select visualization type
- Choose data columns for axes
- Apply color coding
- Set data filters
- Configure aggregations
- Export results

### Data Filters
- Date range selection
- Author filtering
- Impact level filtering
- File type filtering

## Performance Considerations

- Use commit limits for large repositories
- Clear cache when needed
- Consider rate limits of OpenAI API
- Be mindful of memory usage with large datasets

## Troubleshooting

1. If analysis fails:
   - Check repository path
   - Verify OpenAI API key
   - Clear cache and retry
   - Check error messages in console

2. If visualizations don't load:
   - Verify data availability
   - Check selected columns match data types
   - Reduce data size if browser becomes slow

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT License - feel free to use this tool for any purpose, commercial or non-commercial.

## Acknowledgments

- Built with Streamlit
- Powered by OpenAI's GPT-4
- Uses PyDriller for Git analysis
- Visualization by Plotly

## Support

For questions and support, please open an issue in the GitHub repository.
