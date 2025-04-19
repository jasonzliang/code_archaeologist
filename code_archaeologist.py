# Standard library imports
import asyncio
import calendar
import io
import json
import os
import re
import sys
import traceback
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# Third-party library imports
import matplotlib.pyplot as plt
import nest_asyncio
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from openai import OpenAI
from plotly.subplots import make_subplots
from pydriller import Repository, ModifiedFile
from wordcloud import WordCloud

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except:
    client = None

MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are an expert software architect and data analyst specialized in repository analysis.
Analyze git commits considering:
1. Architectural patterns and decisions
2. Technical debt introduction and resolution
3. Breaking changes and API modifications
4. Security implications and vulnerabilities
5. Code quality metrics and trends
6. Team dynamics and collaboration patterns
7. Development velocity and efficiency
8. Knowledge distribution and documentation quality"""

COMMIT_ANALYSIS_PROMPT = """Analyze this commit in detail:
Commit Message: {message}
Files Changed: {files}
Lines Added: {lines_added}
Lines Removed: {lines_removed}
File Types Changed: {file_types}
Complex Files: {complex_files}

Provide a JSON response with the following structure:
```json
{{
    "impact_level": 1-5 (1 minimal, 5 major),
    "category": "architectural|bugfix|feature|refactor|security|documentation|technical_debt|performance|test|ci_cd",
    "breaking_change": true|false,
    "summary": "Detailed analysis of the commit's impact",
    "technical_debt_impact": -2 to 2,
    "security_impact": -2 to 2,
    "knowledge_sharing": 0-5,
    "complexity_change": -2 to 2,
    "tags": ["list", "of", "relevant", "tags"],
    "suggested_reviewers": ["based", "on", "file", "history"]
}}
```"""

QA_SYSTEM_PROMPT = """You are an advanced repository analysis AI with expertise in software development patterns and metrics.
Generate insights and visualizations based on the available data and questions.

Available visualization types:
1. line - Time-based trends
2. bar - Comparisons and rankings
3. pie - Distributions
4. scatter - Correlations and patterns
5. network - Collaboration and dependencies
6. heatmap - Time-based patterns or correlations
7. sunburst - Hierarchical data
8. treemap - Hierarchical comparisons
9. radar - Multi-dimensional metrics
10. sankey - Flow analysis
11. wordcloud - Text analysis
12. bubble - Multi-variable relationships
13. violin - Distribution patterns
14. funnel - Process analysis

When responding, provide:
1. A detailed analysis
2. Relevant metrics
3. Visual suggestions (visualization_type from available ones above)
4. Actionable insights

Provide a JSON response with the following structure:
```json
{
    "answer": "Detailed analysis with insights",
    "metrics": {
        "key_metric1": value1,
        "key_metric2": value2
    },
    "visualization": {
        "type": "visualization_type",
        "data": {
            "x": "column_or_data",
            "y": "column_or_data",
            "color": "optional_column",
            "size": "optional_column"
        },
        "layout": {
            "title": "Visualization title",
            "type": "linear|log|date|etc",
        }
    },
    "recommendations": [
        "actionable insight 1",
        "actionable insight 2"
    ]
}
```"""

CODE_GEN_PROMPT = """# Here is some python code and a request to modify the python code:

## Python code
{python_code}

## Request
{request}

# Your task
Complete the request to the best of your ability and output the modified python code.
"""

def parse_json(rsp):
    pattern = r"```json(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text

class CodeAnalyzer(object):
    def __init__(self, repo_path: str, commit_limit: int = 0):
        self.repo_path = repo_path
        self.commit_limit = commit_limit
        self.file_complexity_cache = {}

    def analyze_repository(self):
        """Analyze repository commits and build collaboration data

        Returns:
            Tuple[List[Dict], nx.Graph]: Tuple containing commits data and collaboration graph
        """
        commits_data = []
        collaboration_graph = nx.Graph()
        file_author_history = defaultdict(list)

        try:
            # Get repository object
            repo = Repository(self.repo_path)

            # Remove merge commits (more than 1 parent)
            all_commits = list(repo.traverse_commits())
            all_commits = [commit for commit in all_commits if len(commit.parents) <= 1]

            # If commit limit is set, get only the most recent commits
            if self.commit_limit > 0:
                # Convert generator to list and slice
                commits_to_analyze = all_commits[-self.commit_limit:] if len(all_commits) > self.commit_limit else all_commits
            else:
                commits_to_analyze = all_commits

            # Create a progress display
            total_commits = len(commits_to_analyze)

            # Display the total number of commits to analyze
            st.write(f"**Total commits to analyze:** {total_commits}")

            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Analyze each commit in the repository
            for i, commit in enumerate(commits_to_analyze):
                try:
                    # Update progress bar and text with each iteration
                    progress_value = (i + 1) / total_commits
                    progress_bar.progress(progress_value)
                    status_text.write(f"**Processing commit {i+1} of {total_commits}:** {commit.hash[:8]}")

                    # Extract author information
                    author = commit.author.name if commit.author.name else commit.author.email

                    # Analyze modified files
                    modified_files = []
                    complex_files = []
                    file_types = []
                    lines_added = 0
                    lines_removed = 0

                    for modified_file in commit.modified_files:
                        if not modified_file.filename or modified_file.source_code is None:
                            continue

                        file_analysis = self.analyze_file_changes(modified_file)
                        modified_files.append(file_analysis['filename'])
                        lines_added += file_analysis['lines_added']
                        lines_removed += file_analysis['lines_removed']
                        file_types.append(file_analysis['file_type'])

                        if file_analysis['complexity_change'] > 1.0:
                            complex_files.append(file_analysis['filename'])

                        # Update file author history
                        file_author_history[modified_file.filename].append(author)

                    # Skip commits with no valid file changes
                    if not modified_files:
                        continue

                    # Prepare commit analysis prompt
                    commit_context = {
                        "message": commit.msg,
                        "files": modified_files,
                        "lines_added": lines_added,
                        "lines_removed": lines_removed,
                        "file_types": list(set(file_types)),
                        "complex_files": complex_files
                    }

                    content = COMMIT_ANALYSIS_PROMPT.format(**commit_context)

                    # Get AI analysis of commit
                    if client is not None:
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": content}
                            ],
                            temperature=0.01
                        )
                    else: # Fallback method using autogen builder model
                        agent_list, builder = init_builder()
                        response = builder._builder_model_create(
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": content}
                            ]
                        )

                    json_str = parse_json(response.choices[0].message.content)
                    analysis = json.loads(json_str)
                    # Build commit data
                    commit_data = {
                        "hash": commit.hash,
                        "author": author,
                        "date": commit.author_date,
                        "message": commit.msg,
                        "files_changed": modified_files,
                        "lines_added": lines_added,
                        "lines_removed": lines_removed,
                        "file_types": list(set(file_types)),
                        "complex_files": complex_files,
                        **analysis
                    }
                    commits_data.append(commit_data)

                    # Update collaboration graph
                    for filename in modified_files:
                        recent_authors = list(set(file_author_history[filename][-5:]))
                        for author1 in recent_authors:
                            for author2 in recent_authors:
                                if author1 < author2:
                                    if not collaboration_graph.has_edge(author1, author2):
                                        collaboration_graph.add_edge(author1, author2, weight=1)
                                    else:
                                        collaboration_graph[author1][author2]['weight'] += 1

                except Exception as e:
                    st.error(f"Error analyzing commit {commit.hash}: {str(e)}")
                    traceback.print_exc()
                    continue

            # Update status when complete
            status_text.write(f"✅ **Repository analysis complete:** {total_commits} commits processed")

        except Exception as e:
            st.error(f"Error analyzing repository: {str(e)}")
            traceback.print_exc()

        return commits_data, collaboration_graph


    def calculate_file_complexity(self, file_content: str) -> float:
        """Calculate code complexity using various metrics"""
        # Basic complexity metrics
        lines = file_content.split('\n')
        complexity = 0
        
        # Control flow complexity
        control_patterns = ['if', 'for', 'while', 'catch', 'switch', 'case']
        complexity += sum(line.count(pattern) for line in lines for pattern in control_patterns)
        
        # Nesting complexity
        indent_levels = [len(line) - len(line.lstrip()) for line in lines]
        complexity += sum(level > 0 for level in indent_levels) * 0.5
        
        # Function complexity
        function_patterns = ['def ', 'function ', 'class ']
        complexity += sum(line.count(pattern) for line in lines for pattern in function_patterns) * 2
        
        return complexity

    def analyze_file_changes(self, modified_file: ModifiedFile) -> Dict[str, Any]:
        """Analyze changes in a single file"""
        if modified_file.filename in self.file_complexity_cache:
            old_complexity = self.file_complexity_cache[modified_file.filename]
        else:
            old_complexity = 0

        new_complexity = self.calculate_file_complexity(modified_file.source_code)
        self.file_complexity_cache[modified_file.filename] = new_complexity

        return {
            'filename': modified_file.filename,
            'complexity_change': new_complexity - old_complexity,
            'lines_added': modified_file.added_lines,
            'lines_removed': modified_file.deleted_lines,  # Fixed: removed_lines -> deleted_lines
            'file_type': modified_file.filename.split('.')[-1] if '.' in modified_file.filename else 'unknown'
        }

def generate_visualization(viz_spec: Dict[str, Any], df: pd.DataFrame) -> go.Figure:
    """Generate basic Plotly visualizations based on specification

    Args:
        viz_spec (Dict[str, Any]): Visualization specification containing type, data, and layout
        df (pd.DataFrame): DataFrame containing the data to visualize

    Returns:
        go.Figure: Plotly figure object
    """
    if viz_spec["type"] == "line":
        fig = px.line(
            df,
            x=viz_spec["data"]["x"],
            y=viz_spec["data"]["y"],
            color=viz_spec["data"].get("color"),
            title=viz_spec["layout"].get("title", "Timeline Analysis")
        )

    elif viz_spec["type"] == "bar":
        fig = px.bar(
            df,
            x=viz_spec["data"]["x"],
            y=viz_spec["data"]["y"],
            color=viz_spec["data"].get("color"),
            title=viz_spec["layout"].get("title", "Bar Chart Analysis")
        )

    elif viz_spec["type"] == "pie":
        fig = px.pie(
            df,
            values=viz_spec["data"]["values"],
            names=viz_spec["data"]["names"],
            title=viz_spec["layout"].get("title", "Distribution Analysis")
        )

    elif viz_spec["type"] == "scatter":
        fig = px.scatter(
            df,
            x=viz_spec["data"]["x"],
            y=viz_spec["data"]["y"],
            color=viz_spec["data"].get("color"),
            size=viz_spec["data"].get("size"),
            title=viz_spec["layout"].get("title", "Correlation Analysis")
        )

    elif viz_spec["type"] == "network":
        # Create network layout
        G = nx.Graph()
        edges = zip(df[viz_spec["data"]["source"]], df[viz_spec["data"]["target"]])
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)

        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines"
        )

        # Create node traces
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=list(G.nodes()),
            marker=dict(size=10)
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=viz_spec["layout"].get("title", "Network Analysis"),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

    elif viz_spec["type"] == "treemap":
        fig = px.treemap(
            df,
            path=viz_spec["data"]["path"],
            values=viz_spec["data"].get("values"),
            color=viz_spec["data"].get("color"),
            title=viz_spec["layout"].get("title", "Hierarchical Analysis")
        )

    else:
        raise ValueError(f"Unsupported visualization type: {viz_spec['type']}")

    # Apply common layout settings
    if "layout" in viz_spec:
        fig.update_layout(
            template=viz_spec["layout"].get("template", "plotly_white"),
            showlegend=viz_spec["layout"].get("showlegend", True),
            height=viz_spec["layout"].get("height", 600),
            width=viz_spec["layout"].get("width", None)
        )

        # Apply axis type if specified
        if viz_spec["layout"].get("type"):
            axis_type = viz_spec["layout"]["type"]
            fig.update_xaxes(type=axis_type)
            fig.update_yaxes(type=axis_type)

        # Apply color scheme if specified
        # if viz_spec["layout"].get("color_scheme"):
        #     fig.update_traces(marker_colorscale=viz_spec["layout"]["color_scheme"])

    return fig

def generate_advanced_visualization(viz_spec: Dict[str, Any], df: pd.DataFrame) -> go.Figure:
    """Generate advanced Plotly visualizations based on specification"""
    if viz_spec["type"] == "heatmap":
        fig = go.Figure(data=go.Heatmap(
            z=df[viz_spec["data"]["z"]],
            x=df[viz_spec["data"]["x"]],
            y=df[viz_spec["data"]["y"]],
            colorscale=viz_spec["layout"].get("color_scheme", "Viridis")
        ))
    
    elif viz_spec["type"] == "sunburst":
        fig = px.sunburst(
            df,
            path=viz_spec["data"]["path"],
            values=viz_spec["data"].get("values"),
            color=viz_spec["data"].get("color")
        )
    
    elif viz_spec["type"] == "radar":
        fig = go.Figure(data=go.Scatterpolar(
            r=viz_spec["data"]["r"],
            theta=viz_spec["data"]["theta"],
            fill='toself'
        ))
    
    elif viz_spec["type"] == "sankey":
        fig = go.Figure(data=[go.Sankey(
            node=viz_spec["data"]["node"],
            link=viz_spec["data"]["link"]
        )])
    
    elif viz_spec["type"] == "bubble":
        fig = px.scatter(
            df,
            x=viz_spec["data"]["x"],
            y=viz_spec["data"]["y"],
            size=viz_spec["data"]["size"],
            color=viz_spec["data"].get("color"),
            hover_name=viz_spec["data"].get("hover_name")
        )
    
    elif viz_spec["type"] == "violin":
        fig = go.Figure(data=go.Violin(
            y=df[viz_spec["data"]["y"]],
            x=df[viz_spec["data"].get("x")],
            box_visible=True,
            meanline_visible=True
        ))
    
    elif viz_spec["type"] == "funnel":
        fig = go.Figure(data=[go.Funnel(
            y=viz_spec["data"]["y"],
            x=viz_spec["data"]["x"]
        )])
    
    elif viz_spec["type"] == "wordcloud":
        # Generate wordcloud using matplotlib
        wordcloud = WordCloud(background_color='white').generate_from_frequencies(
            viz_spec["data"]["frequencies"]
        )
        
        # Convert matplotlib figure to plotly
        img_buf = io.BytesIO()
        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Convert to plotly figure
        fig = px.imshow(plt.imread(io.BytesIO(img_buf.getvalue())))
        
    else:
        # Fall back to basic visualizations
        return generate_visualization(viz_spec, df)
    
    # Apply layout settings
    if "layout" in viz_spec:
        fig.update_layout(
            title=viz_spec["layout"].get("title", ""),
            template=viz_spec["layout"].get("template", "plotly_white"),
            showlegend=viz_spec["layout"].get("showlegend", True)
        )
    
    return fig

def process_question(question: str, df: pd.DataFrame, collaboration_graph: nx.Graph) -> Tuple[str, Dict[str, Any], go.Figure]:
    """Enhanced question processing with advanced analytics"""
    # Prepare rich context about the repository
    if len(collaboration_graph) > 0:
        avg_degree = sum(dict(collaboration_graph.degree()).values()) / len(collaboration_graph)
    else:
        avg_degree = 0

    commit_data = df.to_json(orient='records')
    stats = {
        "basic_stats": {
            "total_commits": len(df),
            "date_range": f"{df['date'].min()} to {df['date'].max()}",
            "unique_authors": len(df['author'].unique()),
            "total_files_changed": len(df['files_changed'])
        },
        "technical_metrics": {
            "avg_impact_level": float(df['impact_level'].mean()),
            "total_breaking_changes": len(df[df['breaking_change']]),
            "net_technical_debt": float(df['technical_debt_impact'].sum())
        },
        "available_columns": list(df.columns),
        "collaboration_metrics": {
            "network_density": float(nx.density(collaboration_graph)),
            "avg_degree": avg_degree
        }
    }
    print("Commit statistics:"); print(stats)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": f"Commit data:\n{json.dumps(commit_data, indent=4)}\n\nCommit statistics:\n{json.dumps(stats, indent=4)}\n\nQuestion: {question}"}
        ],
        temperature=0.3
    )
    
    try:
        result = json.loads(parse_json(response.choices[0].message.content))
        try:
            fig = generate_advanced_visualization(result["visualization"], df)
        except Exception as e:
            traceback.print_exc()
            fig = None
        return result["answer"], result["metrics"], fig
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        traceback.print_exc()
        return "I couldn't process that question properly. Please try rephrasing it.", {}, None

@st.cache_data(persist=True)
def get_commit_info(repo_path, commit_limit):
    analyzer = CodeAnalyzer(repo_path, commit_limit)
    commits_data, collaboration_graph = analyzer.analyze_repository()
    return commits_data, collaboration_graph

def get_abs_path(path, check=True):
    abs_path = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
    if check: assert os.path.exists(abs_path)
    return abs_path

@st.cache_resource
def init_builder():
    cfg_file = get_abs_path('code_generation.yaml')
    with open(cfg_file, 'r') as f: cfg = yaml.safe_load(f)

    assert 'metagpt_path' in cfg and 'team_file' in cfg
    metagpt_path = get_abs_path(cfg['metagpt_path'])
    sys.path.insert(0, metagpt_path)

    team_file = get_abs_path(cfg['team_file'])
    with open(team_file, 'r') as f: team_role = json.load(f)

    from autogen_team import init_builder, BUILDER_LLM_CONFIG, CHAT_LLM_CONFIG

    if 'agent_model' in cfg:
        BUILDER_LLM_CONFIG['agent_model'] = cfg['agent_model']
        CHAT_LLM_CONFIG['model'] = cfg['agent_model']
    if 'builder_model' in cfg:
        BUILDER_LLM_CONFIG['builder_model'] = cfg['builder_model']
    if 'max_round' in cfg:
        CHAT_LLM_CONFIG['max_round'] = int(cfg['max_round'])

    agent_list, _, builder, _, executor = init_builder(
        building_task=None,
        use_builder_dict=True,
        builder_dict=team_role,
        builder_llm_config=BUILDER_LLM_CONFIG,
        clear_cache=True,
        debug_mode=False)
    return agent_list, builder

@st.cache_data(persist=True)
def code_generation(request, code_file):
    try:
        code_file = get_abs_path(code_file)
        with open(code_file, 'r') as f: python_code = f.read()
        prompt = CODE_GEN_PROMPT.format(python_code=python_code,
            request=request)

        from util import extract_code_from_chat
        from autogen_team import start_task, BUILDER_LLM_CONFIG, CHAT_LLM_CONFIG

        agent_list, builder = init_builder()
        chat_result, groupchat_messages = start_task(
                execution_task=prompt,
                agent_list=agent_list,
                chat_llm_config=CHAT_LLM_CONFIG,
                builder=builder,
                builder_llm_config=BUILDER_LLM_CONFIG,
                code_library=None,
                imports=None,
                log_file=None,
                use_captain_agent=False)
        output = extract_code_from_chat(chat_result)
        return output

    except Exception as e:
        st.error(f"An error occurred while trying code generation: {str(e)}")
        st.write("Please check that the request, code file, and metagpt.yaml are correct.")
        traceback.print_exc()

def get_python_files(repo_path):
    """Get a list of Python files in the repository"""
    python_files = []
    try:
        for root, dirs, files in os.walk(repo_path):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
            # Add Python files to the list
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    # Make path relative to repo_path
                    rel_path = os.path.relpath(file_path, repo_path)
                    python_files.append(rel_path)
        return sorted(python_files)
    except Exception as e:
        st.error(f"Error finding Python files: {str(e)}")
        traceback.print_exc()
        return []

def main():
    st.title("🏺Code Archaeologist🏺")
    st.write(f"**Deep dive into your Git repository's evolution with LLM powered analytics!**")
    
    repo_path = st.text_input("Enter repository path (local or remote)")

    col1, col2 = st.columns((3, 1))
    with col1:
        commit_limit = st.number_input(
            "Number of most recent commits to analyze (0 for all)",
            min_value=0,
            value=0,
            help="Limit analysis to K most recent commits. Enter 0 to analyze all commits.")
    with col2:
        if st.button("🔄 Clear Analysis Cache", help="Clear cached repository analysis results"):
            get_commit_info.clear()
            st.success("Cache cleared successfully!")
            st.rerun()

    if repo_path:
        try:
            repo_path = get_abs_path(repo_path)
        except:
            st.error(f"Invalid repository path: {repo_path}")
            return

        try:
            with st.spinner("Analyzing repository..."):
                commits_data, collaboration_graph = get_commit_info(repo_path, commit_limit)

                if not commits_data:
                    st.error("No commit data was found in the repository. Please check the repository path and try again.")
                    return

                df = pd.DataFrame(commits_data)

                # Verify required columns exist
                required_columns = ['author', 'date', 'hash', 'files_changed']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Missing required columns in data: {', '.join(missing_columns)}")
                    return

                # Create tabs for different analysis views
                tabs = st.tabs([
                    "Overview",
                    "Code Insights",
                    "Team Analysis",
                    "Interactive Q&A",
                    # "Custom Analysis",
                    "Code Improvement"
                ])

                with tabs[0]:
                    st.subheader("Repository Overview")

                    # Key metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Commits", len(df))
                    with col2:
                        try:
                            contributors = len(df['author'].unique())
                            st.metric("Contributors", contributors)
                        except KeyError:
                            st.metric("Contributors", "N/A")
                    with col3:
                        try:
                            total_files = df['files_changed'].explode().nunique()
                            st.metric("Files Changed", total_files)
                        except KeyError:
                            st.metric("Files Changed", "N/A")
                    with col4:
                        try:
                            active_days = len(df['date'].dt.date.unique())
                            st.metric("Active Days", active_days)
                        except (KeyError, AttributeError):
                            st.metric("Active Days", "N/A")

                    # Commit activity heatmap
                    try:
                        st.subheader("Commit Activity Pattern")
                        activity_df = df.copy()
                        # Ensure date is in datetime format
                        if not pd.api.types.is_datetime64_any_dtype(activity_df['date']):
                            activity_df['date'] = pd.to_datetime(activity_df['date'], utc=True)

                        activity_df['hour'] = activity_df['date'].dt.hour
                        activity_df['day'] = activity_df['date'].dt.day_name()

                        activity_pivot = pd.pivot_table(
                            activity_df,
                            values='hash',
                            index='day',
                            columns='hour',
                            aggfunc='count',
                            fill_value=0
                        )
                        # Reorder days
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        activity_pivot = activity_pivot.reindex(day_order)

                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=activity_pivot.values,
                            x=activity_pivot.columns,
                            y=activity_pivot.index,
                            colorscale='Viridis'
                        ))

                        fig_heatmap.update_layout(
                            title='Commit Activity by Day and Hour',
                            xaxis_title='Hour of Day',
                            yaxis_title='Day of Week'
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not generate commit activity heatmap: {str(e)}")
                        traceback.print_exc()

                    # Commit trends over time
                    try:
                        st.subheader("Development Velocity")
                        # Ensure date is in datetime format
                        if not pd.api.types.is_datetime64_any_dtype(df['date']):
                            df['date'] = pd.to_datetime(df['date'], utc=True)

                        # Calculate week and year directly from datetime
                        df['week'] = df['date'].dt.isocalendar().week
                        df['year'] = df['date'].dt.year
                        weekly_commits = df.groupby(['year', 'week']).size().reset_index(name='commits')

                        # Create proper datetime for x-axis
                        weekly_commits['date'] = weekly_commits.apply(
                            lambda x: pd.to_datetime(f"{x['year']}-{x['week']:02d}-1", format="%Y-%W-%w"),
                            axis=1
                        )

                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(
                            x=weekly_commits['date'],
                            y=weekly_commits['commits'],
                            mode='lines+markers',
                            name='Weekly Commits'
                        ))
                        fig_trend.add_trace(go.Scatter(
                            x=weekly_commits['date'],
                            y=weekly_commits['commits'].rolling(4).mean(),
                            mode='lines',
                            name='4-week Moving Average',
                            line=dict(dash='dash')
                        ))
                        fig_trend.update_layout(
                            title='Commit Velocity Over Time',
                            xaxis_title='Date',
                            yaxis_title='Number of Commits'
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not generate development velocity chart: {str(e)}")
                        traceback.print_exc()

                with tabs[1]:
                    st.subheader("Code Quality Insights")

                    try:
                        # Technical Debt Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            tech_debt = df.get('technical_debt_impact', pd.Series([0])).sum()
                            recent_debt = df.get('technical_debt_impact', pd.Series([0])).tail(10).sum()
                            st.metric(
                                "Technical Debt Score",
                                f"{tech_debt:.1f}",
                                delta=f"{recent_debt:.1f} (Last 10 commits)"
                            )
                        with col2:
                            breaking_changes = df.get('breaking_change', pd.Series([False])).sum()
                            st.metric(
                                "Breaking Changes",
                                breaking_changes,
                                f"{(breaking_changes/len(df)*100):.1f}% of commits"
                            )
                        with col3:
                            high_complexity = df[df.get('complexity_change', pd.Series([0])) > 1].shape[0]
                            st.metric(
                                "High Complexity Changes",
                                high_complexity,
                                f"{(high_complexity/len(df)*100):.1f}% of commits"
                            )

                        # Code Complexity Trend
                        st.subheader("Code Complexity Evolution")
                        df['cumulative_complexity'] = df.get('complexity_change', pd.Series([0])).cumsum()

                        fig_complexity = go.Figure()
                        fig_complexity.add_trace(go.Scatter(
                            x=df['date'],
                            y=df['cumulative_complexity'],
                            mode='lines',
                            name='Cumulative Complexity'
                        ))
                        fig_complexity.update_layout(
                            title='Code Complexity Over Time',
                            xaxis_title='Date',
                            yaxis_title='Cumulative Complexity Score'
                        )
                        st.plotly_chart(fig_complexity, use_container_width=True)

                        # File Impact Analysis
                        st.subheader("File Impact Analysis")
                        if 'files_changed' in df.columns and 'impact_level' in df.columns:
                            file_impact = df.explode('files_changed').groupby('files_changed').agg({
                                'impact_level': 'mean',
                                'technical_debt_impact': 'sum',
                                'hash': 'count'
                            }).reset_index()

                            file_impact = file_impact.sort_values('impact_level', ascending=False).head(10)

                            fig_impact = go.Figure()
                            fig_impact.add_trace(go.Bar(
                                x=file_impact['files_changed'],
                                y=file_impact['impact_level'],
                                name='Average Impact',
                                marker_color='blue'
                            ))
                            fig_impact.add_trace(go.Bar(
                                x=file_impact['files_changed'],
                                y=file_impact['technical_debt_impact'],
                                name='Technical Debt Impact',
                                marker_color='red'
                            ))
                            fig_impact.update_layout(
                                title='Top 10 Files by Impact',
                                xaxis_title='File',
                                yaxis_title='Score',
                                barmode='group'
                            )
                            st.plotly_chart(fig_impact, use_container_width=True)
                        else:
                            st.warning("File impact analysis requires files_changed and impact_level data")
                    except Exception as e:
                        st.error(f"Error in Code Quality Insights tab: {str(e)}")
                        traceback.print_exc()

                with tabs[2]:
                    st.subheader("Team Collaboration Patterns")

                    try:
                        # Author Statistics
                        author_stats = df.groupby('author').agg({
                            'hash': 'count',
                            'impact_level': 'mean',
                            'breaking_change': 'sum',
                            'technical_debt_impact': 'sum'
                        }).reset_index()

                        col1, col2 = st.columns(2)

                        with col1:
                            # Contribution Distribution
                            fig_contrib = px.pie(
                                author_stats,
                                values='hash',
                                names='author',
                                title='Contribution Distribution'
                            )
                            st.plotly_chart(fig_contrib, use_container_width=True)

                        with col2:
                            # Developer Impact
                            fig_impact = go.Figure()
                            fig_impact.add_trace(go.Bar(
                                x=author_stats['author'],
                                y=author_stats['impact_level'],
                                name='Average Impact'
                            ))
                            fig_impact.update_layout(
                                title='Developer Impact Levels',
                                xaxis_title='Developer',
                                yaxis_title='Average Impact'
                            )
                            st.plotly_chart(fig_impact, use_container_width=True)

                        # Collaboration Network
                        if not collaboration_graph.number_of_nodes() == 0:
                            st.subheader("Team Collaboration Network")

                            # Calculate node sizes based on number of commits
                            node_sizes = {
                                author: len(df[df['author'] == author]) * 5
                                for author in df['author'].unique()
                            }

                            # Create network layout
                            pos = nx.spring_layout(collaboration_graph)

                            # Create edges
                            edge_x = []
                            edge_y = []
                            for edge in collaboration_graph.edges():
                                x0, y0 = pos[edge[0]]
                                x1, y1 = pos[edge[1]]
                                edge_x.extend([x0, x1, None])
                                edge_y.extend([y0, y1, None])

                            # Create nodes
                            node_x = []
                            node_y = []
                            node_text = []
                            node_size = []

                            for node in collaboration_graph.nodes():
                                x, y = pos[node]
                                node_x.append(x)
                                node_y.append(y)
                                node_text.append(node)
                                node_size.append(node_sizes[node])

                            # Create the network visualization
                            fig_network = go.Figure()

                            # Add edges
                            fig_network.add_trace(go.Scatter(
                                x=edge_x,
                                y=edge_y,
                                mode='lines',
                                line=dict(width=0.5, color='#888'),
                                hoverinfo='none'
                            ))

                            # Add nodes
                            fig_network.add_trace(go.Scatter(
                                x=node_x,
                                y=node_y,
                                mode='markers+text',
                                marker=dict(
                                    size=node_size,
                                    line_width=2
                                ),
                                text=node_text,
                                textposition='bottom center'
                            ))

                            fig_network.update_layout(
                                title='Developer Collaboration Network',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                            )

                            st.plotly_chart(fig_network, use_container_width=True)
                        else:
                            st.warning("No collaboration data available to generate network visualization")
                            traceback.print_exc()

                        # Knowledge Distribution
                        st.subheader("Knowledge Distribution")
                        if 'files_changed' in df.columns:
                            file_ownership = df.explode('files_changed').groupby(
                                ['files_changed', 'author']
                            ).size().reset_index(name='changes')

                            # Calculate primary owner for each file
                            file_owners = file_ownership.sort_values(
                                'changes', ascending=False
                            ).groupby('files_changed').head(1)

                            # Create treemap of file ownership
                            fig_ownership = px.treemap(
                                file_owners,
                                path=['author', 'files_changed'],
                                values='changes',
                                title='File Ownership Distribution'
                            )
                            st.plotly_chart(fig_ownership, use_container_width=True)
                        else:
                            st.warning("File ownership analysis requires files_changed data")
                    except Exception as e:
                        st.error(f"Error in Team Collaboration tab: {str(e)}")
                        traceback.print_exc()

                with tabs[3]:
                    st.subheader("Ask Questions About Your Repository")
                    st.write("""Example questions:
- Show me the complexity trend for critical files.
- What's the knowledge distribution across the team?
- Analyze the correlation between technical debt and breaking changes.
- Which files have the most frequent security-related commits?
- Generate a development velocity report for Q1""")

                    question = st.text_input("Enter your question:")
                    if question:
                        with st.spinner("Analyzing..."):
                            try:
                                answer, metrics, fig = process_question(question, df, collaboration_graph)

                                # Display metrics in an organized way
                                if metrics:
                                    cols = st.columns(len(metrics))
                                    for col, (metric_name, value) in zip(cols, metrics.items()):
                                        col.metric(metric_name, str(value))

                                # Display the answer and visualization
                                st.write(answer)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error processing question: {str(e)}")
                                traceback.print_exc()

                # Disabled because not very useful
                # with tabs[4]:
                #     st.subheader("Custom Analysis")

                #     try:
                #         # Create two columns for controls and output
                #         control_col, viz_col = st.columns([1, 2])

                #         with control_col:
                #             st.subheader("Visualization Controls")

                #             # Select visualization type
                #             viz_type = st.selectbox(
                #                 "Select Visualization Type",
                #                 [
                #                     "Time Series",
                #                     "Bar Chart",
                #                     "Scatter Plot",
                #                     "Heatmap",
                #                     "Pie Chart",
                #                     "Box Plot",
                #                     "Violin Plot",
                #                     "Bubble Chart"
                #                 ]
                #             )

                #             # Select data columns based on visualization type
                #             if viz_type in ["Time Series", "Scatter Plot", "Bubble Chart"]:
                #                 x_col = st.selectbox(
                #                     "X-Axis Data",
                #                     df.select_dtypes(include=['datetime64', 'number']).columns
                #                 )
                #             else:
                #                 x_col = st.selectbox(
                #                     "X-Axis Data",
                #                     df.columns
                #                 )

                #             if viz_type != "Pie Chart":
                #                 y_col = st.selectbox(
                #                     "Y-Axis Data",
                #                     df.select_dtypes(include=['number']).columns
                #                 )

                #             # Optional parameters based on chart type
                #             color_col = st.selectbox(
                #                 "Color By (Optional)",
                #                 ["None"] + list(df.columns)
                #             )

                #             if viz_type == "Bubble Chart":
                #                 size_col = st.selectbox(
                #                     "Size By",
                #                     df.select_dtypes(include=['number']).columns
                #                 )

                #             # Data filtering options
                #             st.subheader("Data Filters")

                #             # Date range filter
                #             if 'date' in df.columns:
                #                 date_range = st.date_input(
                #                     "Date Range",
                #                     value=(
                #                         df['date'].min().date(),
                #                         df['date'].max().date()
                #                     ),
                #                     key="date_filter"
                #                 )

                #             # Author filter
                #             if 'author' in df.columns:
                #                 selected_authors = st.multiselect(
                #                     "Filter by Authors",
                #                     options=list(df['author'].unique()),
                #                     default=[]
                #                 )

                #             # Impact level filter
                #             if 'impact_level' in df.columns:
                #                 impact_range = st.slider(
                #                     "Impact Level Range",
                #                     min_value=int(df['impact_level'].min()),
                #                     max_value=int(df['impact_level'].max()),
                #                     value=(
                #                         int(df['impact_level'].min()),
                #                         int(df['impact_level'].max())
                #                     )
                #                 )

                #             # Aggregation options
                #             st.subheader("Aggregation")
                #             agg_func = st.selectbox(
                #                 "Aggregation Function",
                #                 ["None", "Count", "Sum", "Average", "Max", "Min"]
                #             )

                #             if agg_func != "None":
                #                 group_by = st.multiselect(
                #                     "Group By",
                #                     options=[col for col in df.columns if col != y_col],
                #                     default=[]
                #                 )

                #         with viz_col:
                #             st.subheader("Custom Visualization")

                #             # Apply filters
                #             filtered_df = df.copy()

                #             if 'date' in df.columns:
                #                 # Ensure date column is datetime type
                #                 try:
                #                     if not pd.api.types.is_datetime64_any_dtype(filtered_df['date']):
                #                         filtered_df['date'] = pd.to_datetime(filtered_df['date'], utc=True)

                #                     if date_range is not None:
                #                         # Convert date range to UTC timezone-aware datetime
                #                         date_start = pd.to_datetime(date_range[0]).tz_localize('UTC')
                #                         date_end = pd.to_datetime(date_range[1]).tz_localize('UTC')

                #                         filtered_df = filtered_df[
                #                             (filtered_df['date'] >= date_start) &
                #                             (filtered_df['date'] <= date_end)
                #                         ]
                #                 except Exception as e:
                #                     st.error(f"Error processing date filter: {str(e)}")
                #                     st.write("Please ensure the date column contains valid datetime values.")
                #                     traceback.print_exc()

                #             if selected_authors:
                #                 filtered_df = filtered_df[filtered_df['author'].isin(selected_authors)]

                #             if 'impact_level' in df.columns:
                #                 filtered_df = filtered_df[
                #                     (filtered_df['impact_level'] >= impact_range[0]) &
                #                     (filtered_df['impact_level'] <= impact_range[1])
                #                 ]

                #             # Apply aggregation if selected
                #             if agg_func != "None" and group_by:
                #                 agg_map = {
                #                     "Count": "count",
                #                     "Sum": "sum",
                #                     "Average": "mean",
                #                     "Max": "max",
                #                     "Min": "min"
                #                 }
                #                 filtered_df = filtered_df.groupby(group_by)[y_col].agg(
                #                     agg_map[agg_func]
                #                 ).reset_index()

                #             # Create visualization based on selection
                #             try:
                #                 if viz_type == "Time Series":
                #                     fig = px.line(
                #                         filtered_df,
                #                         x=x_col,
                #                         y=y_col,
                #                         color=None if color_col == "None" else color_col,
                #                         title=f"{y_col} over {x_col}"
                #                     )

                #                 elif viz_type == "Bar Chart":
                #                     fig = px.bar(
                #                         filtered_df,
                #                         x=x_col,
                #                         y=y_col,
                #                         color=None if color_col == "None" else color_col,
                #                         title=f"{y_col} by {x_col}"
                #                     )

                #                 elif viz_type == "Scatter Plot":
                #                     fig = px.scatter(
                #                         filtered_df,
                #                         x=x_col,
                #                         y=y_col,
                #                         color=None if color_col == "None" else color_col,
                #                         title=f"{y_col} vs {x_col}"
                #                     )

                #                 elif viz_type == "Heatmap":
                #                     pivot_data = filtered_df.pivot_table(
                #                         values=y_col,
                #                         index=x_col,
                #                         columns=color_col if color_col != "None" else None,
                #                         aggfunc='count'
                #                     )
                #                     fig = px.imshow(
                #                         pivot_data,
                #                         title=f"Heatmap of {y_col}"
                #                     )

                #                 elif viz_type == "Pie Chart":
                #                     fig = px.pie(
                #                         filtered_df,
                #                         values=y_col,
                #                         names=x_col,
                #                         title=f"Distribution of {x_col}"
                #                     )

                #                 elif viz_type == "Box Plot":
                #                     fig = px.box(
                #                         filtered_df,
                #                         x=x_col,
                #                         y=y_col,
                #                         color=None if color_col == "None" else color_col,
                #                         title=f"Distribution of {y_col} by {x_col}"
                #                     )

                #                 elif viz_type == "Violin Plot":
                #                     fig = px.violin(
                #                         filtered_df,
                #                         x=x_col,
                #                         y=y_col,
                #                         color=None if color_col == "None" else color_col,
                #                         title=f"Distribution of {y_col} by {x_col}"
                #                     )

                #                 elif viz_type == "Bubble Chart":
                #                     fig = px.scatter(
                #                         filtered_df,
                #                         x=x_col,
                #                         y=y_col,
                #                         size=size_col,
                #                         color=None if color_col == "None" else color_col,
                #                         title=f"{y_col} vs {x_col} (size: {size_col})"
                #                     )

                #                 # Update layout for all charts
                #                 fig.update_layout(
                #                     height=600,
                #                     template="plotly_white"
                #                 )

                #                 # Display the figure
                #                 st.plotly_chart(fig, use_container_width=True)

                #                 # Display data table
                #                 with st.expander("View Data"):
                #                     st.dataframe(filtered_df)

                #                 # Add export options
                #                 if st.button("Export Data"):
                #                     csv = filtered_df.to_csv(index=False)
                #                     st.download_button(
                #                         label="Download CSV",
                #                         data=csv,
                #                         file_name="custom_analysis.csv",
                #                         mime="text/csv"
                #                     )

                #             except Exception as e:
                #                 st.error(f"Error creating visualization: {str(e)}")
                #                 st.write("Please try different parameters or data columns.")
                #                 traceback.print_exc()
                #     except Exception as e:
                #         st.error(f"Error in Custom Analysis tab: {str(e)}")
                #         traceback.print_exc()

                with tabs[4]:
                    st.subheader("Code Generation & Improvement")
                    st.write("Select a Python file and specify how you'd like to improve or modify it.")

                    # Get list of Python files in the repository
                    python_files = get_python_files(repo_path)

                    if not python_files:
                        st.warning("No Python files found in this repository.")
                    else:
                        selected_file = st.selectbox(
                            "Select Python file to improve",
                            options=python_files
                        )

                        if selected_file:
                            # Create full path to file
                            full_path = os.path.join(repo_path, selected_file)

                            # Show file preview
                            try:
                                with open(full_path, 'r') as f:
                                    file_content = f.read()
                                with st.expander("Preview file content"):
                                    st.code(file_content, language="python")
                            except Exception as e:
                                st.error(f"Error reading file: {str(e)}")

                            # Request for code improvement
                            improvement_request = st.text_area(
                                "What changes or improvements would you like to make?",
                                placeholder="For example: Add docstrings, optimize performance, add type hints, etc."
                            )

                            if st.button("Generate Improved Code"):
                                if improvement_request:
                                    with st.spinner("Generating improved code..."):
                                        try:
                                            improved_code = code_generation(improvement_request, full_path)

                                            if improved_code:
                                                st.success("Code successfully improved!")
                                                st.subheader("Improved Code")
                                                st.code(improved_code, language="python")

                                                # Option to download the improved code
                                                st.download_button(
                                                    label="Download Improved Code",
                                                    data=improved_code,
                                                    file_name=selected_file,
                                                    mime="text/plain"
                                                )

                                                # Option to save changes directly to file
                                                if st.button("Save Changes to File"):
                                                    try:
                                                        with open(full_path, 'w') as f:
                                                            f.write(improved_code)
                                                        st.success(f"Changes saved to {selected_file}")
                                                    except Exception as e:
                                                        st.error(f"Error saving changes: {str(e)}")
                                            else:
                                                st.error("Could not generate improved code. Check the logs for details.")
                                        except Exception as e:
                                            st.error(f"Error in code generation: {str(e)}")
                                            traceback.print_exc()
                                else:
                                    st.warning("Please specify what improvements or changes you'd like to make.")

        except Exception as e:
            st.error(f"An error occurred while analyzing the repository: {str(e)}")
            st.write("Please check the repository path and ensure you have the necessary permissions.")
            traceback.print_exc()

def test_code_gen():
    request = "Modify all print statements to use logging.info instead"
    code_file = "test_file.py"
    print(code_generation(request, code_file))

if __name__ == "__main__":
    main()
