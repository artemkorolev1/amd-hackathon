#!/usr/bin/env python3.12
"""Search for academic papers on self-improving agents, Kaizen AI, supervisor-based task distribution, etc."""

import urllib.request
import urllib.parse
import json
import time
import sys
import socket
import ssl

def search_arxiv(query, max_results=5):
    """Search arXiv API"""
    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0 (linux)"})
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data = resp.read().decode('utf-8')
        return data
    except Exception as e:
        print(f"  arXiv error: {e}", file=sys.stderr)
        return None

def search_semantic_scholar(query, limit=5):
    """Search Semantic Scholar API"""
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(query)}&limit={limit}&fields=title,authors,year,externalIds,abstract,url"
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchAgent/1.0 (linux)"})
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return data
    except Exception as e:
        print(f"  Semantic Scholar error: {e}", file=sys.stderr)
        return None

def parse_arxiv_xml(xml_data):
    """Simple parse of arXiv XML to extract entries"""
    results = []
    entries = xml_data.split('<entry>')[1:] if '<entry>' in xml_data else []
    for entry in entries:
        title = ""
        if '<title>' in entry:
            title = entry.split('<title>')[1].split('</title>')[0].strip()
            title = title.replace('\n', ' ').replace('  ', ' ')
        authors = []
        author_parts = entry.split('<author>')
        for ap in author_parts[1:]:
            if '<name>' in ap:
                authors.append(ap.split('<name>')[1].split('</name>')[0].strip())
        link = ""
        if '<id>' in entry:
            link = entry.split('<id>')[1].split('</id>')[0].strip()
        summary = ""
        if '<summary>' in entry:
            summary = entry.split('<summary>')[1].split('</summary>')[0].strip()
            summary = summary.replace('\n', ' ').replace('  ', ' ')
        if title:
            results.append({
                'title': title,
                'authors': authors,
                'url': link,
                'abstract': summary,
                'source': 'arXiv'
            })
    return results

# Queries organized by category
queries = [
    ("Self-improving AI agents", "self-improving+AI+agents"),
    ("Recursive self-improvement AI", "recursive+self-improvement+artificial+intelligence"),
    ("Kaizen continuous improvement AI", "kaizen+AI+continuous+improvement"),
    ("Supervisor-based multi-agent task distribution", "supervisor+multi-agent+task+distribution"),
    ("Multi-agent systems hierarchical control", "hierarchical+multi-agent+supervisor+control"),
    ("Circuit breaker patterns AI", "circuit+breaker+pattern+AI+system"),
    ("AI safety circuit breaker", "circuit+breaker+AI+safety"),
    ("Graceful degradation AI systems", "graceful+degradation+AI+autonomous+systems"),
    ("Fallback strategies autonomous AI", "fallback+strategy+autonomous+AI+system"),
    ("Agent safety production reliability", "agent+safety+reliability+production+AI"),
    ("Safe AI agent deployment", "safe+deployment+AI+agents+production"),
]

print("=" * 80)
print("ACADEMIC PAPER SEARCH RESULTS")
print("=" * 80)

all_papers = []

# First try arXiv for the key topics
arxiv_key_queries = [
    ("Self-improving / recursive AI agents", "self-improving+agents"),
    ("Supervisor-based multi-agent systems", "supervisor+multi-agent+reinforcement+learning"),
    ("Circuit breakers for AI safety", "circuit+breaker+AI+safety"),
    ("Graceful degradation autonomous systems", "graceful+degradation+autonomous"),
    ("Agent safety production deployment", "agent+safety+deployment+reinforcement+learning"),
]

for topic_name, query in arxiv_key_queries:
    print(f"\n--- arXiv: {topic_name} ---")
    xml_data = search_arxiv(query)
    if xml_data:
        papers = parse_arxiv_xml(xml_data)
        for p in papers:
            p['category'] = topic_name
            all_papers.append(p)
            print(f"  [{len(all_papers)}] {p['title'][:80]}")
            print(f"      URL: {p['url']}")
    time.sleep(1.0)

# Then Semantic Scholar for all topics
for topic_name, query in queries:
    print(f"\n--- Semantic Scholar: {topic_name} ---")
    ss_results = search_semantic_scholar(topic_name, limit=5)
    if ss_results and 'data' in ss_results:
        for p in ss_results['data']:
            title = p.get('title', 'Unknown')
            year = p.get('year', 'Unknown')
            authors_list = [a.get('name', 'Unknown') for a in p.get('authors', [])]
            ext_ids = p.get('externalIds', {})
            url = ""
            if ext_ids.get('ArXiv'):
                url = f"https://arxiv.org/abs/{ext_ids['ArXiv']}"
            elif ext_ids.get('CorpusId'):
                url = f"https://www.semanticscholar.org/paper/{ext_ids['CorpusId']}"
            elif p.get('url'):
                url = p['url']
            abstract = (p.get('abstract', '') or '')[:300]
            
            paper_entry = {
                'title': title,
                'authors': ', '.join(authors_list[:5]),
                'year': year,
                'url': url,
                'abstract': abstract,
                'category': topic_name,
                'source': 'Semantic Scholar'
            }
            all_papers.append(paper_entry)
            print(f"  [{len(all_papers)}] {title[:80]}")
            print(f"      Authors: {', '.join(authors_list[:3])}")
            print(f"      URL: {url}")
    time.sleep(1.0)

# Deduplicate by title
seen = set()
unique_papers = []
for p in all_papers:
    t = p['title'].lower().strip()
    if t not in seen:
        seen.add(t)
        unique_papers.append(p)

print("\n\n" + "=" * 80)
print(f"TOTAL UNIQUE PAPERS FOUND: {len(unique_papers)}")
print("=" * 80)

for i, p in enumerate(unique_papers):
    print(f"\n[{i+1}] {p['title']}")
    print(f"    Authors: {p.get('authors', 'N/A')}")
    print(f"    Year: {p.get('year', 'N/A')}")
    print(f"    URL: {p.get('url', 'N/A')}")
    print(f"    Category: {p.get('category', 'N/A')}")
    print(f"    Abstract: {p.get('abstract', '')[:200]}...")

print("\n\n--- JSON OUTPUT ---")
print(json.dumps(unique_papers, indent=2))
