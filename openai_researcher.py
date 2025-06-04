import asyncio
import streamlit as st
from typing import Dict, Any, List
from agents import Agent, Runner
from agents import set_default_openai_key
from agents.tool import function_tool, WebSearchTool
from crawl4ai import AsyncWebCrawler
import json

# Set page configuration
st.set_page_config(
    page_title="OpenAI Deep Research Agent",
    page_icon="📘",
    layout="wide"
)

# Initialize session state for API key if not exists
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

# Sidebar for API keys
with st.sidebar:
    st.title("API Configuration")
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        value=st.session_state.openai_api_key,
        type="password"
    )
    
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        set_default_openai_key(openai_api_key)

# Main content
st.title("📘 OpenAI Deep Research Agent")
st.markdown(
    "This OpenAI Agent performs deep research on any topic"
)

# Research topic input
research_topic = st.text_input(
    "Enter your research topic:", 
    placeholder="e.g., Latest developments in AI"
)

# Web crawling tool using Crawl4AI
@function_tool
async def crawl_urls(urls: List[str], max_urls: int = 10) -> Dict[str, Any]:
    """
    Crawl multiple URLs and extract their content using Crawl4AI.
    
    Args:
        urls: List of URLs to crawl
        max_urls: Maximum number of URLs to crawl (default: 10)
    """
    try:
        # Limit the number of URLs to crawl
        urls_to_crawl = urls[:max_urls]
        
        crawled_content = []
        
        async with AsyncWebCrawler(verbose=True) as crawler:
            for i, url in enumerate(urls_to_crawl):
                try:
                    st.write(
                        f"🔍 Crawling URL {i+1}/{len(urls_to_crawl)}: {url}"
                    )
                    
                    result = await crawler.arun(
                        url=url,
                        word_count_threshold=100,  # Only extract substantial content
                        bypass_cache=True
                    )
                    
                    if result.success:
                        content = result.cleaned_html or result.markdown
                        crawled_content.append({
                            "url": url,
                            "title": result.metadata.get("title", ""),
                            "content": content,
                            "word_count": len((content or "").split()),
                            "success": True
                        })
                        title = result.metadata.get('title', url)
                        st.write(f"✅ Successfully crawled: {title}")
                    else:
                        crawled_content.append({
                            "url": url,
                            "error": "Failed to crawl",
                            "success": False
                        })
                        st.write(f"❌ Failed to crawl: {url}")
                        
                except Exception as e:
                    crawled_content.append({
                        "url": url,
                        "error": str(e),
                        "success": False
                    })
                    st.write(f"❌ Error crawling {url}: {str(e)}")
        
        successful_crawls = [item for item in crawled_content if item.get("success")]
        
        return {
            "success": True,
            "total_urls_attempted": len(urls_to_crawl),
            "successful_crawls": len(successful_crawls),
            "crawled_content": crawled_content,
            "summary": (
                f"Successfully crawled {len(successful_crawls)} out of "
                f"{len(urls_to_crawl)} URLs"
            )
        }
        
    except Exception as e:
        st.error(f"Crawling error: {str(e)}")
        return {"error": str(e), "success": False}

# URL extraction helper tool
@function_tool  
async def extract_urls_from_search(search_results: str, max_urls: int = 10) -> List[str]:
    """
    Extract URLs from web search results.
    
    Args:
        search_results: Raw search results from WebSearchTool
        max_urls: Maximum number of URLs to extract
    """
    try:
        # Parse the search results to extract URLs
        # The WebSearchTool typically returns results in a structured format
        urls = []
        
        # Try to parse as JSON first
        try:
            if isinstance(search_results, str):
                results_data = json.loads(search_results)
            else:
                results_data = search_results
                
            # Extract URLs from different possible structures
            if isinstance(results_data, list):
                for result in results_data[:max_urls]:
                    if isinstance(result, dict):
                        url = (result.get('url') or result.get('link') or 
                               result.get('href'))
                        if url:
                            urls.append(url)
            elif isinstance(results_data, dict):
                # Handle single result or results with metadata
                results_list = results_data.get('results', [results_data])
                for result in results_list[:max_urls]:
                    if isinstance(result, dict):
                        url = (result.get('url') or result.get('link') or 
                               result.get('href'))
                        if url:
                            urls.append(url)
                            
        except (json.JSONDecodeError, TypeError):
            # If not JSON, try to extract URLs using simple string parsing
            # This is a fallback for plain text search results
            lines = str(search_results).split('\n')
            for line in lines:
                if 'http' in line:
                    # Simple URL extraction - can be improved
                    words = line.split()
                    for word in words:
                        if word.startswith(('http://', 'https://')):
                            urls.append(word.strip('.,;:()[]{}'))
                            if len(urls) >= max_urls:
                                break
                if len(urls) >= max_urls:
                    break
        
        # Remove duplicates while preserving order
        unique_urls = []
        seen = set()
        for url in urls:
            if url not in seen:
                unique_urls.append(url)
                seen.add(url)
        
        return unique_urls[:max_urls]
        
    except Exception as e:
        st.error(f"URL extraction error: {str(e)}")
        return []

# Updated research agent with new tools
research_agent = Agent(
    name="research_agent",
    instructions="""You are a research assistant that performs comprehensive, multi-angle research on ANY topic or request.

    Your adaptive workflow:
    1. ANALYZE THE REQUEST: First understand the nature of the research request:
       - Is it about a product/service (shopping, comparison, reviews)?
       - Is it about a market/industry (business, technology, trends)?
       - Is it about a concept/topic (academic, educational, explanatory)?
       - Is it about current events (news, developments, updates)?
       - Is it about a process/how-to (instructions, guides, best practices)?
       - Is it about a person/organization (biography, background, achievements)?
       - Or something else entirely?
       
    2. DETERMINE RESEARCH ANGLES: Based on the request type, break it into 4-6 relevant angles:
       
       For PRODUCTS/SHOPPING: features, pricing, reviews, alternatives, pros/cons, best use cases
       For MARKETS/INDUSTRIES: size/trends, key players, technologies, challenges, opportunities, future outlook
       For CONCEPTS/TOPICS: definition/overview, key aspects, applications, debates, recent developments, implications
       For CURRENT EVENTS: background, key facts, different perspectives, impact, timeline, future implications
       For PROCESSES/HOW-TO: overview, step-by-step methods, tools/requirements, best practices, common mistakes, tips
       For PEOPLE/ORGANIZATIONS: background, achievements, current activities, impact, controversies, future plans
       
    3. RESEARCH EACH ANGLE: For each relevant research angle:
       - Use web_search with specific, targeted queries tailored to the angle
       - Extract URLs from search results using extract_urls_from_search
       - Use crawl_urls to get detailed content from the most relevant sources
       - Ensure you gather substantial, relevant information for each angle
       
    4. SYNTHESIZE COMPREHENSIVE REPORT: 
       - Create a well-structured report using markdown format with clear sections for each research angle
       - Include specific data, facts, quotes, and examples from sources
       - Use in-place citations in the format [Source Title](URL) throughout the text
       - Tailor the depth and style to match the request type
       - Ensure the report is detailed, substantive, and directly addresses the original request
       
    5. QUALITY STANDARDS:
       - Each section should be substantive with specific details and evidence
       - Include relevant data, statistics, quotes, or examples when available
       - Provide concrete information and actionable insights
       - Use appropriate tone (analytical for business, informative for education, practical for how-to, etc.)
       - Always cite sources properly with in-place citations
    
    Always adapt your approach based on the specific nature of the research request.
    """,
    tools=[WebSearchTool(), extract_urls_from_search, crawl_urls]
)

# Keep the same elaboration agent
elaboration_agent = Agent(
    name="elaboration_agent",
    instructions="""You are an expert content enhancer specializing in research elaboration.

    When given a research report:
    1. Analyze the structure and content of the report
    2. Enhance the report by:
       - Including relevant examples, case studies, and real-world applications
       - Expanding on key points with additional context and nuance
       - Adding descriptions of visual elements (charts, diagrams, infographics)
       - Incorporating latest trends and future predictions
       - Suggesting practical implications for different stakeholders
       - Adding proper in-place citations in the format [Source Title](URL)
       - Maintaining consistent citation format throughout the document
    3. Maintain academic rigor and factual accuracy
    4. Preserve the original structure and title - DO NOT change the report title
    5. Ensure all additions are relevant and valuable to the topic
    6. Always cite sources properly using in-place citations when adding new information
    7. Use consistent in-place citation format: [Source Title](URL) or [Author/Organization](URL)
    8. Do not add "Enhanced Research Report:" or similar prefixes to the title
    9. Generate the final report in markdown format with proper headers, subheaders, bullet points, etc.
    10. CRITICAL: Do NOT add AI meta-commentary at the end such as:
        - "Next Steps" sections
        - "Conclusion" with AI process mentions
        - Requests for feedback ("Please let me know...")
        - Mentions of data collection phases or AI processes
        - Phrases like "if you need more information" or "let me know if there are specific aspects"
        - Any requests for further input or interaction
    11. End the report with substantive content, not meta-commentary
    12. The enhanced report should be a complete, standalone professional document
    """
)

async def run_research_process(topic: str):
    """Run the complete research process."""
    # Step 1: Initial Research
    with st.spinner("Conducting comprehensive research..."):
        research_result = await Runner.run(research_agent, topic)
        initial_report = research_result.final_output
    
    # Display initial report in an expander
    with st.expander("View Initial Research Report"):
        st.markdown(initial_report)
    
    # Step 2: Enhance the report
    with st.spinner("Enhancing the report with additional information..."):
        elaboration_input = f"""
        RESEARCH TOPIC: {topic}
        
        INITIAL RESEARCH REPORT:
        {initial_report}
        
        Please enhance this research report with additional information, examples, case studies, 
        and deeper insights while maintaining its academic rigor and factual accuracy.
        """
        
        elaboration_result = await Runner.run(elaboration_agent, elaboration_input)
        enhanced_report = elaboration_result.final_output
    
    return enhanced_report

# Main research process
button_disabled = not (openai_api_key and research_topic)
if st.button("Start Research", disabled=button_disabled):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not research_topic:
        st.warning("Please enter a research topic.")
    else:
        try:
            # Create placeholder for the final report
            report_placeholder = st.empty()
            
            # Run the research process
            enhanced_report = asyncio.run(run_research_process(research_topic))
            
            # Display the enhanced report
            report_placeholder.markdown("## Enhanced Research Report")
            report_placeholder.markdown(enhanced_report)
            
            # Add download button
            filename = f"{research_topic.replace(' ', '_')}_report.md"
            st.download_button(
                "Download Report",
                enhanced_report,
                file_name=filename,
                mime="text/markdown"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Powered by OpenAI Agents SDK and Crawl4AI") 