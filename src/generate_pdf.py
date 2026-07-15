import os
import subprocess
import sys
import markdown

def main() -> None:
    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    report_md_path = os.path.join(project_root, "docs", "PROJECT_REPORT.md")
    report_html_path = os.path.join(project_root, "docs", "PROJECT_REPORT.html")
    report_pdf_path = os.path.join(project_root, "docs", "PROJECT_REPORT.pdf")
    
    print(f"Reading Markdown report: {report_md_path}")
    if not os.path.exists(report_md_path):
        print(f"Error: {report_md_path} not found.")
        sys.exit(1)
        
    with open(report_md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    # Convert markdown to html (enabling tables extension)
    html_content = markdown.markdown(md_content, extensions=["tables", "fenced_code"])
    
    # Styled HTML Template with clean, professional typography and print formatting
    styled_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ESC Project Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #2d3748;
        line-height: 1.6;
        margin: 50px auto;
        max-width: 850px;
        padding: 0 30px;
    }}
    h1, h2, h3, h4 {{
        color: #1a365d;
        font-weight: 700;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }}
    h1 {{
        font-size: 2.2em;
        border-bottom: 3px solid #1a365d;
        padding-bottom: 10px;
        margin-top: 0;
    }}
    h2 {{
        font-size: 1.6em;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 8px;
        margin-top: 40px;
        page-break-before: always; /* Force section to start on a new page */
    }}
    h3 {{
        font-size: 1.3em;
        color: #2b6cb0;
    }}
    p {{
        margin-bottom: 1.25em;
        text-align: justify;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.95em;
    }}
    th, td {{
        padding: 12px 15px;
        border: 1px solid #cbd5e0;
        text-align: left;
    }}
    th {{
        background-color: #ebf8ff;
        color: #2b6cb0;
        font-weight: 700;
    }}
    tr:nth-child(even) {{
        background-color: #f7fafc;
    }}
    pre, code {{
        background-color: #f7fafc;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.9em;
        border-radius: 4px;
    }}
    pre {{
        padding: 15px;
        overflow-x: auto;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4299e1;
        margin-bottom: 20px;
    }}
    code {{
        padding: 2px 6px;
        color: #c53030;
        background-color: #fff5f5;
        border: 1px solid #fed7d7;
    }}
    pre code {{
        padding: 0;
        color: inherit;
        background-color: transparent;
        border: none;
    }}
    blockquote, .note, .tip {{
        padding: 15px 20px;
        margin: 25px 0;
        border-left: 4px solid #3182ce;
        background-color: #ebf8ff;
        color: #2b6cb0;
        border-radius: 0 4px 4px 0;
    }}
    blockquote p {{
        margin-bottom: 0;
    }}
    img {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: 30px auto;
        border: 1px solid #cbd5e0;
        border-radius: 6px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    /* Mermaid diagram handling */
    pre.mermaid {{
        background-color: white;
        border: none;
        text-align: center;
    }}
    hr {{
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 40px 0;
    }}
    @media print {{
        body {{
            margin: 20px;
            max-width: 100%;
            font-size: 11pt;
        }}
        h1, h2, h3 {{
            page-break-inside: avoid;
        }}
        table, pre, blockquote {{
            page-break-inside: avoid;
        }}
    }}
</style>
</head>
<body>
    {html_content}
</body>
</html>
"""
    
    # Save styled HTML
    print(f"Saving HTML format: {report_html_path}")
    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(styled_html)
        
    # Convert HTML to PDF using Chrome Headless CLI
    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    print("Launching Google Chrome in headless mode to render and print PDF...")
    
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={report_pdf_path}",
        "--no-sandbox",
        report_html_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n🎉 Success! PDF Report generated successfully at:\n{report_pdf_path}")
        
        # Clean up temporary HTML file
        if os.path.exists(report_html_path):
            os.remove(report_html_path)
            print("Removed temporary HTML rendering file.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to print PDF using Google Chrome. process exited with: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during PDF generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
