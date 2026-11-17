with open(r'c:\Users\Raghav\Desktop\raga\parallel-chess-programming-engine\templates\dashboard.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

header_lines = lines[0:28]
analytics_html_lines = lines[177:263]
chart_js_lines = lines[330:418]

analytics_content = "".join(header_lines) + '''
    <div class="container">
        <div class="dashboard-title">
            <h1><i class="fas fa-chart-line"></i> Platform Analytics</h1>
            <p>Detailed statistics and training metrics</p>
            <a href="{{ url_for('dashboard') }}" class="btn btn-primary" style="margin-top: 20px; display: inline-block;"><i class="fas fa-arrow-left"></i> Back to Dashboard</a>
        </div>
''' + "".join(analytics_html_lines).replace('style="border-top: 1px solid rgba(255, 255, 255, 0.2); padding-top: 40px;"', '') + '''
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
''' + "".join(chart_js_lines) + '''
    </script>
</body>
</html>
'''

with open(r'c:\Users\Raghav\Desktop\raga\parallel-chess-programming-engine\templates\analytics.html', 'w', encoding='utf-8') as f:
    f.write(analytics_content)

# Modify dashboard
new_dash_lines = lines[:177] + lines[263:]
new_dash = "".join(new_dash_lines)

# Remove chart js
new_dash = new_dash.replace("".join(chart_js_lines), "")

# Update showAnalytics
new_dash = new_dash.replace(
    "const analyticsSection = document.getElementById('analytics-section');\n            if (analyticsSection) {\n                analyticsSection.scrollIntoView({ behavior: 'smooth' });\n            }",
    "window.location.href = '{{ url_for(\"analytics\") }}';"
)

with open(r'c:\Users\Raghav\Desktop\raga\parallel-chess-programming-engine\templates\dashboard.html', 'w', encoding='utf-8') as f:
    f.write(new_dash)

print("Success")
