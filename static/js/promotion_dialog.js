
// Chess Promotion Dialog
class PromotionDialog {
    constructor() {
        this.isOpen = false;
        this.callback = null;
    }

    show(color, callback) {
        this.callback = callback;
        this.isOpen = true;
        
        const isWhite = color === 'w' || color === 'white';
        const symbols = isWhite ? 
            {q: '♕', r: '♖', b: '♗', n: '♘'} : 
            {q: '♛', r: '♜', b: '♝', n: '♞'};
        
        const dialog = document.createElement('div');
        dialog.className = 'promotion-dialog';
        dialog.innerHTML = `
            <div class="promotion-content">
                <h3>Choose Promotion Piece</h3>
                <div class="promotion-pieces">
                    <div class="promotion-piece" data-piece="q">${symbols.q}</div>
                    <div class="promotion-piece" data-piece="r">${symbols.r}</div>
                    <div class="promotion-piece" data-piece="b">${symbols.b}</div>
                    <div class="promotion-piece" data-piece="n">${symbols.n}</div>
                </div>
            </div>
        `;
        
        // Style the dialog
        dialog.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
        `;
        
        dialog.querySelector('.promotion-content').style.cssText = `
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        `;
        
        dialog.querySelector('.promotion-pieces').style.cssText = `
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 15px;
        `;
        
        dialog.querySelectorAll('.promotion-piece').forEach(piece => {
            piece.style.cssText = `
                font-size: 3em;
                padding: 10px;
                cursor: pointer;
                border-radius: 5px;
                transition: background 0.2s;
            `;
            
            piece.addEventListener('mouseover', () => {
                piece.style.background = '#f0f0f0';
            });
            
            piece.addEventListener('mouseout', () => {
                piece.style.background = '';
            });
            
            piece.addEventListener('click', () => {
                this.selectPiece(piece.dataset.piece);
                document.body.removeChild(dialog);
                this.isOpen = false;
            });
        });
        
        document.body.appendChild(dialog);
    }
    
    selectPiece(piece) {
        if (this.callback) {
            this.callback(piece);
        }
    }
}

const promotionDialog = new PromotionDialog();
