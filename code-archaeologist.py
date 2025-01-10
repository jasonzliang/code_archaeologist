import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from pydriller import Repository, ModifiedFile
from openai import OpenAI
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Any, Tuple
import calendar
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import traceback

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
}}"""

QA_SYSTEM_PROMPT = """You are an advanced repository analysis AI with expertise in software development patterns and metrics.
Generate insights and visualizations based on the available data and questions.

Available visualization types:
1. timeline - Time-based trends
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
3. Visual suggestions
4. Actionable insights

Response format:
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
            "color_scheme": "color scheme name"
        }
    },
    "recommendations": [
        "actionable insight 1",
        "actionable insight 2"
    ]
}"""

def parse_json(rsp):
    pattern = r"```json(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text

class CodeAnalyzer:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.file_complexity_cache = {}

    def analyze_repository(self):
        """Analyze repository commits and build collaboration data

        Returns:
            Tuple[List[Dict], nx.Graph]: Tuple containing commits data and collaboration graph
        """
        commits_data = []
        collaboration_graph = nx.Graph()
        file_author_history = defaultdict(list)

        # Analyze each commit in the repository
        for commit in Repository(self.repo_path).traverse_commits():
            # Skip merge commits
            if len(commit.parents) > 1:
                continue

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

                # Update file author history with consistent author identifier
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

            try:
                content = COMMIT_ANALYSIS_PROMPT.format(**commit_context)

                # Get AI analysis of commit
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content}
                    ],
                    temperature=0.3
                )
                json_str = parse_json(response.choices[0].message.content)
                analysis = json.loads(json_str)
                # Build commit data with consistent author field
                commit_data = {
                    "hash": commit.hash,
                    "author": author,  # Using consistent author identifier
                    "date": commit.author_date,
                    "message": commit.msg,
                    "files_changed": modified_files,
                    "lines_added": lines_added,
                    "lines_removed": lines_removed,
                    "file_types": list(set(file_types)),
                    "complex_files": complex_files,
                    **analysis  # Include AI analysis results
                }
                print(commit_data)
                commits_data.append(commit_data)

                # Update collaboration graph with consistent author identifiers
                for filename in modified_files:
                    recent_authors = list(set(file_author_history[filename][-5:]))  # Last 5 authors
                    for author1 in recent_authors:
                        for author2 in recent_authors:
                            if author1 < author2:  # Avoid duplicate edges
                                if not collaboration_graph.has_edge(author1, author2):
                                    collaboration_graph.add_edge(author1, author2, weight=1)
                                else:
                                    collaboration_graph[author1][author2]['weight'] += 1

            except Exception as e:
                print(f"Error analyzing commit {commit.hash}: {str(e)}")
                traceback.print_exc()
                continue

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
        return super().generate_visualization(viz_spec, df)
    
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
    context = {
        "basic_stats": {
            "total_commits": len(df),
            "date_range": f"{df['date'].min()} to {df['date'].max()}",
            "unique_authors": len(df['author'].unique()),
            "total_files_changed": df['files_changed'].sum()
        },
        "technical_metrics": {
            "avg_impact_level": df['impact_level'].mean(),
            "total_breaking_changes": len(df[df['breaking_change']]),
            "net_technical_debt": df['technical_debt_impact'].sum()
        },
        "available_columns": list(df.columns),
        "collaboration_metrics": {
            "network_density": nx.density(collaboration_graph),
            "avg_degree": sum(dict(collaboration_graph.degree()).values()) / len(collaboration_graph)
        }
    }
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{json.dumps(context, indent=2)}\n\nQuestion: {question}"}
        ],
        temperature=0.3
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        fig = generate_advanced_visualization(result["visualization"], df)
        return result["answer"], result["metrics"], fig
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return "I couldn't process that question properly. Please try rephrasing it.", {}, None

def main():
    st.title("🏺 Advanced Code Archaeologist")
    st.write("Deep dive into your repository's evolution with AI-powered analytics")
    
    repo_path = st.text_input("Enter repository path (local or remote)")
    
    if repo_path:
        try:
            with st.spinner("Analyzing repository..."):
                analyzer = CodeAnalyzer(repo_path)
                commits_data, collaboration_graph = analyzer.analyze_repository()

                # Check if we got any data
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
                    "Custom Analysis"
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

                    # Commit trends over time
                    try:
                        st.subheader("Development Velocity")
                        df['week'] = df['date'].dt.isocalendar().week
                        df['year'] = df['date'].dt.year
                        weekly_commits = df.groupby(['year', 'week']).size().reset_index(name='commits')
                        weekly_commits['date'] = pd.to_datetime(
                            weekly_commits['year'].astype(str) + '-W' +
                            weekly_commits['week'].astype(str) + '-1'
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

                with tabs[3]:
                    st.subheader("Ask Questions About Your Repository")
                    st.write("""Example questions:
                    - Show me the complexity trend for critical files
                    - What's the knowledge distribution across the team?
                    - Analyze the correlation between technical debt and breaking changes
                    - Which files have the most frequent security-related commits?
                    - Generate a development velocity report for Q1
                    """)

                    question = st.text_input("Enter your question:")
                    if question:
                        with st.spinner("Analyzing..."):
                            try:
                                answer, metrics, fig = process_question(question, df, collaboration_graph)

                                # Display metrics in an organized way
                                if metrics:
                                    cols = st.columns(len(metrics))
                                    for col, (metric_name, value) in zip(cols, metrics.items()):
                                        col.metric(metric_name, value)

                                # Display the answer and visualization
                                st.write(answer)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error processing question: {str(e)}")

                with tabs[4]:
                    st.subheader("Custom Analysis")

                    try:
                        # Create two columns for controls and output
                        control_col, viz_col = st.columns([1, 2])

                        with control_col:
                            st.subheader("Visualization Controls")

                            # Select visualization type
                            viz_type = st.selectbox(
                                "Select Visualization Type",
                                [
                                    "Time Series",
                                    "Bar Chart",
                                    "Scatter Plot",
                                    "Heatmap",
                                    "Pie Chart",
                                    "Box Plot",
                                    "Violin Plot",
                                    "Bubble Chart"
                                ]
                            )

                            # Select data columns based on visualization type
                            if viz_type in ["Time Series", "Scatter Plot", "Bubble Chart"]:
                                x_col = st.selectbox(
                                    "X-Axis Data",
                                    df.select_dtypes(include=['datetime64', 'number']).columns
                                )
                            else:
                                x_col = st.selectbox(
                                    "X-Axis Data",
                                    df.columns
                                )

                            if viz_type != "Pie Chart":
                                y_col = st.selectbox(
                                    "Y-Axis Data",
                                    df.select_dtypes(include=['number']).columns
                                )

                            # Optional parameters based on chart type
                            color_col = st.selectbox(
                                "Color By (Optional)",
                                ["None"] + list(df.columns)
                            )

                            if viz_type == "Bubble Chart":
                                size_col = st.selectbox(
                                    "Size By",
                                    df.select_dtypes(include=['number']).columns
                                )

                            # Data filtering options
                            st.subheader("Data Filters")

                            # Date range filter
                            if 'date' in df.columns:
                                date_range = st.date_input(
                                    "Date Range",
                                    value=(
                                        df['date'].min().date(),
                                        df['date'].max().date()
                                    ),
                                    key="date_filter"
                                )

                            # Author filter
                            if 'author' in df.columns:
                                selected_authors = st.multiselect(
                                    "Filter by Authors",
                                    options=list(df['author'].unique()),
                                    default=[]
                                )

                            # Impact level filter
                            if 'impact_level' in df.columns:
                                impact_range = st.slider(
                                    "Impact Level Range",
                                    min_value=int(df['impact_level'].min()),
                                    max_value=int(df['impact_level'].max()),
                                    value=(
                                        int(df['impact_level'].min()),
                                        int(df['impact_level'].max())
                                    )
                                )

                            # Aggregation options
                            st.subheader("Aggregation")
                            agg_func = st.selectbox(
                                "Aggregation Function",
                                ["None", "Count", "Sum", "Average", "Max", "Min"]
                            )

                            if agg_func != "None":
                                group_by = st.multiselect(
                                    "Group By",
                                    options=[col for col in df.columns if col != y_col],
                                    default=[]
                                )

                        with viz_col:
                            st.subheader("Custom Visualization")

                            # Apply filters
                            filtered_df = df.copy()

                            if 'date' in df.columns and date_range is not None:
                                filtered_df = filtered_df[
                                    (filtered_df['date'].dt.date >= date_range[0]) &
                                    (filtered_df['date'].dt.date <= date_range[1])
                                ]

                            if selected_authors:
                                filtered_df = filtered_df[filtered_df['author'].isin(selected_authors)]

                            if 'impact_level' in df.columns:
                                filtered_df = filtered_df[
                                    (filtered_df['impact_level'] >= impact_range[0]) &
                                    (filtered_df['impact_level'] <= impact_range[1])
                                ]

                            # Apply aggregation if selected
                            if agg_func != "None" and group_by:
                                agg_map = {
                                    "Count": "count",
                                    "Sum": "sum",
                                    "Average": "mean",
                                    "Max": "max",
                                    "Min": "min"
                                }
                                filtered_df = filtered_df.groupby(group_by)[y_col].agg(
                                    agg_map[agg_func]
                                ).reset_index()

                            # Create visualization based on selection
                            try:
                                if viz_type == "Time Series":
                                    fig = px.line(
                                        filtered_df,
                                        x=x_col,
                                        y=y_col,
                                        color=None if color_col == "None" else color_col,
                                        title=f"{y_col} over {x_col}"
                                    )

                                elif viz_type == "Bar Chart":
                                    fig = px.bar(
                                        filtered_df,
                                        x=x_col,
                                        y=y_col,
                                        color=None if color_col == "None" else color_col,
                                        title=f"{y_col} by {x_col}"
                                    )

                                elif viz_type == "Scatter Plot":
                                    fig = px.scatter(
                                        filtered_df,
                                        x=x_col,
                                        y=y_col,
                                        color=None if color_col == "None" else color_col,
                                        title=f"{y_col} vs {x_col}"
                                    )

                                elif viz_type == "Heatmap":
                                    pivot_data = filtered_df.pivot_table(
                                        values=y_col,
                                        index=x_col,
                                        columns=color_col if color_col != "None" else None,
                                        aggfunc='count'
                                    )
                                    fig = px.imshow(
                                        pivot_data,
                                        title=f"Heatmap of {y_col}"
                                    )

                                elif viz_type == "Pie Chart":
                                    fig = px.pie(
                                        filtered_df,
                                        values=y_col,
                                        names=x_col,
                                        title=f"Distribution of {x_col}"
                                    )

                                elif viz_type == "Box Plot":
                                    fig = px.box(
                                        filtered_df,
                                        x=x_col,
                                        y=y_col,
                                        color=None if color_col == "None" else color_col,
                                        title=f"Distribution of {y_col} by {x_col}"
                                    )

                                elif viz_type == "Violin Plot":
                                    fig = px.violin(
                                        filtered_df,
                                        x=x_col,
                                        y=y_col,
                                        color=None if color_col == "None" else color_col,
                                        title=f"Distribution of {y_col} by {x_col}"
                                    )

                                elif viz_type == "Bubble Chart":
                                    fig = px.scatter(
                                        filtered_df,
                                        x=x_col,
                                        y=y_col,
                                        size=size_col,
                                        color=None if color_col == "None" else color_col,
                                        title=f"{y_col} vs {x_col} (size: {size_col})"
                                    )

                                # Update layout for all charts
                                fig.update_layout(
                                    height=600,
                                    template="plotly_white"
                                )

                                # Display the figure
                                st.plotly_chart(fig, use_container_width=True)

                                # Display data table
                                with st.expander("View Data"):
                                    st.dataframe(filtered_df)

                                # Add export options
                                if st.button("Export Data"):
                                    csv = filtered_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv,
                                        file_name="custom_analysis.csv",
                                        mime="text/csv"
                                    )

                            except Exception as e:
                                st.error(f"Error creating visualization: {str(e)}")
                                st.write("Please try different parameters or data columns.")
                    except Exception as e:
                        st.error(f"Error in Custom Analysis tab: {str(e)}")

        except Exception as e:
            st.error(f"An error occurred while analyzing the repository: {str(e)}")
            st.write("Please check the repository path and ensure you have the necessary permissions.")


if __name__ == "__main__":
    main()
