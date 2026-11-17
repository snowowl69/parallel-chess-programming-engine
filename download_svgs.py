import urllib.request
import os

pieces = {
    'wk': 'https://upload.wikimedia.org/wikipedia/commons/4/42/Chess_klt45.svg',
    'wq': 'https://upload.wikimedia.org/wikipedia/commons/1/15/Chess_qlt45.svg',
    'wr': 'https://upload.wikimedia.org/wikipedia/commons/7/72/Chess_rlt45.svg',
    'wb': 'https://upload.wikimedia.org/wikipedia/commons/b/b1/Chess_blt45.svg',
    'wn': 'https://upload.wikimedia.org/wikipedia/commons/7/70/Chess_nlt45.svg',
    'wp': 'https://upload.wikimedia.org/wikipedia/commons/4/45/Chess_plt45.svg',
    'bk': 'https://upload.wikimedia.org/wikipedia/commons/f/f0/Chess_kdt45.svg',
    'bq': 'https://upload.wikimedia.org/wikipedia/commons/4/47/Chess_qdt45.svg',
    'br': 'https://upload.wikimedia.org/wikipedia/commons/f/ff/Chess_rdt45.svg',
    'bb': 'https://upload.wikimedia.org/wikipedia/commons/9/98/Chess_bdt45.svg',
    'bn': 'https://upload.wikimedia.org/wikipedia/commons/e/ef/Chess_ndt45.svg',
    'bp': 'https://upload.wikimedia.org/wikipedia/commons/c/c7/Chess_pdt45.svg'
}

js_content = 'const svgPieces = {\n'

for name, url in pieces.items():
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            svg_data = response.read().decode('utf-8')
            # Clean up the SVG slightly for embedding in JS
            svg_data = svg_data.replace('\n', '').replace('\r', '').replace('"', "'")
            js_content += f'    "{name}": "{svg_data}",\n'
    except Exception as e:
        print(f'Failed {name}: {e}')

js_content += '};\n'

js_content += '''
const getPieceSvg = (char) => {
    if (!char) return '';
    const map = {
        'K': 'wk', 'Q': 'wq', 'R': 'wr', 'B': 'wb', 'N': 'wn', 'P': 'wp',
        'k': 'bk', 'q': 'bq', 'r': 'br', 'b': 'bb', 'n': 'bn', 'p': 'bp'
    };
    return svgPieces[map[char]] || '';
};
'''

os.makedirs('static/js', exist_ok=True)
with open('static/js/pieces.js', 'w', encoding='utf-8') as f:
    f.write(js_content)

print('Generated pieces.js successfully!')
