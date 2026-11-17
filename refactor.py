import os, re

dash_path = r'c:\Users\Raghav\Desktop\raga\parallel-chess-programming-engine\templates\dashboard.html'
with open(dash_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Extract header (up to </head>) and nav (up to <div class="container">)
header_match = re.search(r'(.*?<div class="container">)', html, re.DOTALL)
header = header_match.group(1)

# Modify header for analytics title
header = header.replace('<title>Dashboard - Chess AI Platform</title>', '<title>Analytics - Chess AI Platform</title>')

# Extract analytics section
analytics_match = re.search(r'(<div id="analytics-section".*?</div>\s*</div>\s*{% endif %})', html, re.DOTALL)
if not analytics_match:
    print("Could not find analytics section")
    exit(1)
analytics_section = analytics_match.group(1)

# Extract Chart JS
chart_js_match = re.search(r'({% if stats.role == \'admin\' %}\s*Chart\.defaults\.color.*?new Chart\(gamesCtx.*?\}\);\s*{% endif %})', html, re.DOTALL)
chart_js = chart_js_match.group(1) if chart_js_match else ''

analytics_html = header + f'''
        <div class="dashboard-title">
            <h1><i class="fas fa-chart-line"></i> Platform Analytics</h1>
            <p>Detailed statistics and training metrics</p>
            <a href="{{{{ url_for('dashboard') }}}}" class="btn btn-primary" style="margin-top: 20px;"><i class="fas fa-arrow-left"></i> Back to Dashboard</a>
        </div>

        {analytics_section.replace('style="border-top: 1px solid rgba(255, 255, 255, 0.2); padding-top: 40px;"', '')}
    </div>

    <script>
        {chart_js}
    </script>
</body>
</html>
'''

with open(r'c:\Users\Raghav\Desktop\raga\parallel-chess-programming-engine\templates\analytics.html', 'w', encoding='utf-8') as f:
    f.write(analytics_html)

# Now remove analytics section and chart JS from dashboard
new_dash = html.replace(analytics_section, '')
if chart_js:
    new_dash = new_dash.replace(chart_js, '')

# Change showAnalytics function
new_dash = new_dash.replace('const analyticsSection = document.getElementById(\'analytics-section\');\n            if (analyticsSection) {\n                analyticsSection.scrollIntoView({ behavior: \'smooth\' });\n            }', 'window.location.href = \'{{ url_for("analytics") }}\';')

with open(dash_path, 'w', encoding='utf-8') as f:
    f.write(new_dash)

print('Analytics page created and dashboard updated!')
