import os
import json
import threading
import subprocess
import time
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'chess_ai_platform_secret_key_2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app, cors_allowed_origins="*")

training_state = {
    'is_training': False,
    'process': None,
    'start_time': None,
    'current_iteration': 0,
    'total_games': 0,
    'current_loss': 0.0,
    'best_model_elo': 1500,
    'training_thread': None
}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def init_user_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'user' CHECK(role IN ('admin', 'user')),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        games_played INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        draws INTEGER DEFAULT 0,
        training_sessions INTEGER DEFAULT 0,
        favorite_mode TEXT DEFAULT 'playing',
        rating INTEGER DEFAULT 1200,
        is_active BOOLEAN DEFAULT 1
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS game_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        game_type TEXT NOT NULL,
        moves_pgn TEXT NOT NULL,
        result TEXT NOT NULL,
        duration_seconds INTEGER,
        user_color TEXT,
        ai_difficulty TEXT DEFAULT 'medium',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        used_for_training BOOLEAN DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS training_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        admin_id INTEGER,
        model_name TEXT NOT NULL,
        iterations_completed INTEGER DEFAULT 0,
        games_trained_on INTEGER DEFAULT 0,
        training_start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        training_end_time TIMESTAMP,
        status TEXT DEFAULT 'active' CHECK(status IN ('active', 'completed', 'paused', 'failed')),
        model_path TEXT,
        checkpoint_path TEXT,
        current_elo INTEGER DEFAULT 1500,
        notes TEXT,
        FOREIGN KEY (admin_id) REFERENCES users (id)
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS model_checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        training_session_id INTEGER,
        iteration_number INTEGER,
        model_path TEXT NOT NULL,
        evaluation_score REAL,
        games_count INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_best_model BOOLEAN DEFAULT 0,
        FOREIGN KEY (training_session_id) REFERENCES training_sessions (id)
    )''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get current user stats for home page
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT username, role, games_played, wins, losses, draws, rating FROM users WHERE id = ?', 
                   (session['user_id'],))
    user_data = cursor.fetchone()
    
    if not user_data:
        session.clear()
        conn.close()
        return redirect(url_for('login'))
    
    stats = {
        'username': user_data[0],
        'role': user_data[1],
        'games_played': user_data[2],
        'wins': user_data[3],
        'losses': user_data[4],
        'draws': user_data[5],
        'rating': user_data[6]
    }
    
    # Get platform statistics for admin
    if stats['role'] == 'admin':
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
        total_users = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(games_played) FROM users')
        platform_games = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(rating) FROM users WHERE games_played > 0')
        avg_rating = cursor.fetchone()[0] or 1200
        
        stats['total_users'] = total_users
        stats['platform_games'] = platform_games
        stats['avg_rating'] = round(avg_rating, 1)
        
        # Get training stats
        try:
            from training_monitor import TrainingDatabase
            db = TrainingDatabase('training_history.db')
            summary = db.get_summary_stats()
            stats['training_iterations'] = summary.get('total_iterations', 0)
            stats['training_games'] = summary.get('total_games', 0)
        except:
            stats['training_iterations'] = 0
            stats['training_games'] = 0
    
    conn.close()
    return render_template('index_dashboard.html', stats=stats, training_status=training_state)

@app.route('/analytics')
def analytics():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') != 'admin':
        flash('Admin access required', 'error')
        return redirect(url_for('dashboard'))
    
    # Get current user stats
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT username, role, games_played, wins, losses, draws, rating FROM users WHERE id = ?', 
                   (session['user_id'],))
    user_data = cursor.fetchone()
    conn.close()
    
    stats = {
        'username': user_data[0],
        'role': user_data[1],
        'games_played': user_data[2],
        'wins': user_data[3],
        'losses': user_data[4],
        'draws': user_data[5],
        'rating': user_data[6]
    }
    
    # Get analytics data
    analytics_data = {'metrics': [], 'training': {}, 'users': {}}
    try:
        analytics_data = get_analytics()
    except Exception as e:
        print(f"Error getting analytics: {e}")
        
    return render_template('analytics.html', stats=stats, analytics=analytics_data)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get current user stats
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT username, role, games_played, wins, losses, draws, rating FROM users WHERE id = ?', 
                   (session['user_id'],))
    user_data = cursor.fetchone()
    
    if not user_data:
        session.clear()
        conn.close()
        return redirect(url_for('login'))
    
    stats = {
        'username': user_data[0],
        'role': user_data[1],
        'games_played': user_data[2],
        'wins': user_data[3],
        'losses': user_data[4],
        'draws': user_data[5],
        'rating': user_data[6]
    }
    
    # Get analytics data
    try:
        from training_monitor import TrainingDatabase
        db = TrainingDatabase('training_history.db')
        summary = db.get_summary_stats()
        metrics = db.get_all_metrics()[-10:]  # Last 10 iterations
    except Exception as e:
        print(f"Error loading training data: {e}")
        summary = {
            'total_iterations': 0,
            'total_games': 0,
            'total_positions': 0,
            'avg_loss': 0,
            'best_loss': 0,
            'total_time_hours': 0
        }
        metrics = []
    
    # Get platform statistics
    cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
    total_users = cursor.fetchone()[0]
    
    cursor.execute('SELECT SUM(games_played) FROM users')
    total_games = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT AVG(rating) FROM users WHERE games_played > 0')
    avg_rating = cursor.fetchone()[0] or 1200
    conn.close()
    
    analytics_data = {
        'training': summary,
        'users': {
            'total': total_users,
            'total_games': total_games,
            'avg_rating': round(avg_rating, 1)
        },
        'metrics': metrics
    }
    
    return render_template('dashboard.html', analytics=analytics_data, stats=stats, training_status=training_state)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, password_hash, role FROM users WHERE username = ? AND is_active = 1', (username,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            session['role'] = user[2]
            
            cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                         (datetime.now(), user[0]))
            conn.commit()
            conn.close()
            
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        
        conn.close()
        flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return render_template('register.html')
        
        password_hash = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM users')
            user_count = cursor.fetchone()[0]
            role = 'admin' if user_count == 0 else 'user'
            
            cursor.execute('''INSERT INTO users (username, email, password_hash, role) 
                            VALUES (?, ?, ?, ?)''', (username, email, password_hash, role))
            conn.commit()
            conn.close()
            
            flash(f'Registration successful! You are registered as {role}.', 'success')
            return redirect(url_for('login'))
        
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
        except Exception as e:
            flash(f'Registration error: {str(e)}', 'error')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/admin/users')
def manage_users():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT id, username, email, password_hash, role, games_played, wins, losses, draws, 
                      rating, created_at, last_login, is_active FROM users ORDER BY created_at DESC''')
    users = cursor.fetchall()
    conn.close()
    
    users_list = []
    for user in users:
        users_list.append({
            'id': user[0],
            'username': user[1],
            'email': user[2],
            'password_hash': user[3],
            'role': user[4],
            'games_played': user[5],
            'wins': user[6],
            'losses': user[7],
            'draws': user[8],
            'rating': user[9],
            'created_at': user[10],
            'last_login': user[11],
            'is_active': user[12]
        })
    
    return render_template('manage_users.html', users=users_list)

@app.route('/admin/users/add', methods=['POST'])
def add_user():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Admin access required'}), 403
    
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user')
    
    if not username or not email or not password:
        return jsonify({'success': False, 'error': 'All fields required'})
    
    try:
        password_hash = generate_password_hash(password)
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO users (username, email, password_hash, role) 
                        VALUES (?, ?, ?, ?)''', (username, email, password_hash, role))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'User added successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'error': 'Username or email already exists'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
def delete_user(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Admin access required'}), 403
    
    # Prevent admin from deleting themselves
    if user_id == session['user_id']:
        return jsonify({'success': False, 'error': 'Cannot delete your own account'})
    
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET is_active = 0 WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'User deactivated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/users/<int:user_id>/activate', methods=['POST'])
def activate_user(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Admin access required'}), 403
    
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET is_active = 1 WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'User activated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/users/<int:user_id>/stats')
def get_admin_user_stats(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Admin access required'}), 403
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT username, email, role, games_played, wins, losses, draws, 
                      rating, created_at, last_login FROM users WHERE id = ?''', (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return jsonify({'success': False, 'error': 'User not found'})
    
    return jsonify({
        'success': True,
        'user': {
            'username': user[0],
            'email': user[1],
            'role': user[2],
            'games_played': user[3],
            'wins': user[4],
            'losses': user[5],
            'draws': user[6],
            'rating': user[7],
            'created_at': user[8],
            'last_login': user[9],
            'win_rate': round((user[4] / user[3] * 100) if user[3] > 0 else 0, 1)
        }
    })

@app.route('/admin/users/<int:user_id>/view')
def view_admin_user_stats(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Admin access required', 'error')
        return redirect(url_for('dashboard'))
        
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT username, email, role, games_played, wins, losses, draws, 
                      rating, created_at, last_login, is_active FROM users WHERE id = ?''', (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('manage_users'))
        
    user_data = {
        'id': user_id,
        'username': user[0],
        'email': user[1],
        'role': user[2],
        'games_played': user[3],
        'wins': user[4],
        'losses': user[5],
        'draws': user[6],
        'rating': user[7],
        'created_at': user[8],
        'last_login': user[9],
        'is_active': user[10],
        'win_rate': round((user[4] / user[3] * 100) if user[3] > 0 else 0, 1)
    }
    
    return render_template('user_stats.html', user=user_data)

@app.route('/admin/users/<int:user_id>/reset-password', methods=['POST'])
def reset_user_password(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Admin access required'}), 403
    
    data = request.json
    new_password = data.get('password')
    
    if not new_password:
        return jsonify({'success': False, 'error': 'Password required'})
    
    try:
        password_hash = generate_password_hash(new_password)
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (password_hash, user_id))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Password reset successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'is_training': training_state['is_training'],
        'current_iteration': training_state['current_iteration'],
        'current_loss': training_state['current_loss'],
        'best_model_elo': training_state['best_model_elo']
    })

@app.route('/game')
def game():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('game.html')

@app.route('/training')
def training():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') != 'admin':
        flash('Admin access required', 'error')
        return redirect(url_for('index'))
    return render_template('training.html')

@app.route('/playing')
def playing():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('playing.html')

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('history.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Admin access required'}), 403
    return jsonify({'success': True, 'message': 'Dataset uploaded successfully'})

@app.route('/api/game/move', methods=['POST'])
def make_move():
    data = request.json
    return jsonify({'status': 'ok'})

def update_user_stats(user_id, result_str, user_color):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT games_played, wins, losses, draws, rating FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return
        
        games_played, wins, losses, draws, rating = row
        games_played += 1
        
        user_won = False
        is_draw = False
        
        if result_str == '1/2-1/2':
            draws += 1
            is_draw = True
        elif (result_str == '1-0' and user_color == 'white') or (result_str == '0-1' and user_color == 'black'):
            wins += 1
            user_won = True
        else:
            losses += 1
            
        if user_won:
            rating += 15
        elif is_draw:
            pass
        else:
            rating = max(100, rating - 15)
            
        cursor.execute('''UPDATE users SET games_played = ?, wins = ?, losses = ?, draws = ?, rating = ?
                          WHERE id = ?''', (games_played, wins, losses, draws, rating, user_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating stats: {e}")

@app.route('/api/stats', methods=['GET'])
def get_user_stats():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT username, games_played, wins, losses, draws, rating, role 
                      FROM users WHERE id = ?''', (session['user_id'],))
    user_data = cursor.fetchone()
    conn.close()
    
    if not user_data:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'username': user_data[0],
        'games_played': user_data[1],
        'wins': user_data[2],
        'losses': user_data[3],
        'draws': user_data[4],
        'rating': user_data[5],
        'role': user_data[6]
    })

@app.route('/api/training/start', methods=['POST'])
def start_training():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Admin access required'}), 403
    
    if training_state['is_training']:
        return jsonify({'success': False, 'error': 'Training already in progress'}), 400
    
    training_state['is_training'] = True
    training_state['start_time'] = time.time()
    training_state['current_iteration'] = 0
    
    # Get parameters from UI, falling back to defaults if not provided
    iterations = request.form.get('iterations', 3, type=int)
    games = request.form.get('games_per_iteration', 5, type=int)
    mcts = request.form.get('mcts_simulations', 50, type=int)
    
    # Launch the actual real training process in the background with UI parameters
    try:
        cmd = [
            'python', 'intelligent_training_pipeline.py',
            '--iterations', str(iterations),
            '--games', str(games),
            '--mcts', str(mcts)
        ]
        training_state['process'] = subprocess.Popen(cmd)
    except Exception as e:
        print(f"Failed to launch training process: {e}")
        
    flash('Training started successfully', 'success')
    return jsonify({'success': True, 'status': 'started', 'message': 'Training initiated'})

@app.route('/start_training', methods=['POST'])
def start_training_legacy():
    return start_training()

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'success': False, 'error': 'Admin access required'}), 403
    
    training_state['is_training'] = False
    if training_state['process']:
        training_state['process'].terminate()
        training_state['process'] = None
    
    flash('Training stopped', 'info')
    return jsonify({'success': True, 'status': 'stopped', 'message': 'Training terminated'})

@app.route('/stop_training', methods=['POST'])
def stop_training_legacy():
    return stop_training()

@app.route('/training_status', methods=['GET'])
def training_status():
    return jsonify({
        'is_training': training_state['is_training'],
        'current_iteration': training_state['current_iteration'],
        'total_games': training_state['total_games'],
        'current_loss': training_state['current_loss'],
        'best_model_elo': training_state['best_model_elo'],
        'start_time': training_state['start_time']
    })

@app.route('/api/game_history', methods=['GET'])
def game_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT game_type, result, created_at, duration_seconds, ai_difficulty 
                      FROM game_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 20''', 
                   (session['user_id'],))
    games = cursor.fetchall()
    conn.close()
    
    return jsonify({
        'games': [{
            'game_type': g[0],
            'result': g[1],
            'created_at': g[2],
            'duration': g[3],
            'difficulty': g[4]
        } for g in games]
    })

@app.route('/api/new_game', methods=['POST'])
def new_game():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    game_type = data.get('game_type', 'vs_ai')
    difficulty = data.get('difficulty', 'medium')
    user_color = data.get('color', 'white')
    
    game_id = f"game_{int(time.time())}_{session['user_id']}"
    
    import chess
    from chess_ai import get_best_move
    
    board = chess.Board()
    moves = []
    
    if user_color == 'black':
        # Map difficulty strings to levels (easy: 1, medium: 2, hard: 3)
        difficulty_level = 2
        if difficulty == 'easy': difficulty_level = 1
        elif difficulty == 'hard': difficulty_level = 3
        
        # AI makes the first move
        ai_move = get_best_move(board, difficulty_level)
        board.push(ai_move)
        moves.append(ai_move.uci())
    
    return jsonify({
        'success': True,
        'game_state': {
            'game_id': game_id,
            'fen': board.fen(),
            'moves': moves,
            'user_color': user_color,
            'difficulty': difficulty,
            'turn': 'black' if user_color == 'black' else 'white'
        }
    })

@app.route('/api/resign', methods=['POST'])
def resign_game():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    user_color = data.get('user_color', 'white')
    result = '0-1' if user_color == 'white' else '1-0'
    update_user_stats(session['user_id'], result, user_color)
    return jsonify({'success': True, 'message': 'Game resigned'})

@app.route('/api/draw', methods=['POST'])
def draw_game():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    user_color = data.get('user_color', 'white')
    update_user_stats(session['user_id'], '1/2-1/2', user_color)
    return jsonify({'success': True, 'message': 'Game drawn'})

@app.route('/api/make_move', methods=['POST'])
def make_move_api():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    import chess
    
    data = request.json
    fen = data.get('fen', data.get('current_fen'))
    move = data.get('move')
    game_id = data.get('game_id')
    
    if not fen:
        return jsonify({'success': False, 'error': 'No board position provided'})
    
    if not move:
        return jsonify({'success': False, 'error': 'No move provided'})
    
    try:
        board = chess.Board(fen)
        
        try:
            chess_move = chess.Move.from_uci(move)
        except:
            return jsonify({'success': False, 'error': f'Invalid move format: {move}'})
        
        # Handle pawn promotion - auto-promote to queen if not specified
        if chess_move not in board.legal_moves:
            # Check if it's a pawn promotion move without promotion piece
            piece = board.piece_at(chess_move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                to_rank = chess.square_rank(chess_move.to_square)
                # If pawn reaches last rank, auto-promote to queen
                if (piece.color == chess.WHITE and to_rank == 7) or (piece.color == chess.BLACK and to_rank == 0):
                    chess_move = chess.Move(chess_move.from_square, chess_move.to_square, promotion=chess.QUEEN)
        
        if chess_move not in board.legal_moves:
            return jsonify({'success': False, 'error': 'Illegal move'})
        
        board.push(chess_move)
        moves_list = [move]
        
        ai_move = None
        if not board.is_game_over():
            legal_moves = list(board.legal_moves)
            if legal_moves:
                from chess_ai import get_best_move
                diff_str = data.get('difficulty', 'medium')
                diff_val = 2
                if diff_str == 'easy': diff_val = 1
                elif diff_str == 'hard': diff_val = 3
                
                ai_move = get_best_move(board, difficulty=diff_val)
                if isinstance(ai_move, str):
                    ai_move = chess.Move.from_uci(ai_move)
                board.push(ai_move)
                moves_list.append(ai_move.uci())
        
        is_over = board.is_game_over()
        result = board.result() if is_over else None
        if is_over:
            update_user_stats(session['user_id'], result, data.get('user_color', 'white'))

        return jsonify({
            'success': True,
            'fen': board.fen(),
            'moves': moves_list,
            'ai_move': ai_move.uci() if ai_move else None,
            'game_over': is_over,
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_legal_moves', methods=['POST'])
def get_legal_moves():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    import chess
    
    data = request.json
    fen = data.get('fen')
    from_square = data.get('from_square', data.get('square'))
    
    if not fen:
        return jsonify({'success': False, 'error': 'No board position provided'})
    
    if not from_square:
        return jsonify({'success': False, 'error': 'No square specified'})
    
    try:
        board = chess.Board(fen)
        from_sq = chess.parse_square(from_square)
        
        legal_moves = []
        for move in board.legal_moves:
            if move.from_square == from_sq:
                legal_moves.append(move.uci())
        
        return jsonify({
            'success': True,
            'moves': legal_moves
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/check_promotion', methods=['POST'])
def check_promotion():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    import chess
    
    data = request.json
    fen = data.get('fen')
    move = data.get('move')
    
    try:
        board = chess.Board(fen)
        chess_move = chess.Move.from_uci(move)
        
        piece = board.piece_at(chess_move.from_square)
        is_promotion = (piece and piece.piece_type == chess.PAWN and
                       ((board.turn == chess.WHITE and chess.square_rank(chess_move.to_square) == 7) or
                        (board.turn == chess.BLACK and chess.square_rank(chess_move.to_square) == 0)))
        
        return jsonify({
            'success': True,
            'needs_promotion': is_promotion
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status_update', training_state)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    init_user_db()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
