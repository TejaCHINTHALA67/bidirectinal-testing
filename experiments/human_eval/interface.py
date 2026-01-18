"""
Human Evaluation Interface for IEEE-Grade RAG Experiments

Flask web application for collecting human annotations of RAG responses.

Requirements:
- Side-by-side comparison of 3 systems (Standard, Naive, Bidirectional)
- 5-point Likert scales: Factuality, Relevance, Completeness, Fluency, Safety
- Inter-annotator agreement (Cohen's kappa)
- Export to CSV for analysis
"""

import os
import sqlite3
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import hashlib
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # Generate secure secret key
CORS(app)

# Configuration
DATABASE = 'experiments/human_eval/annotations.db'
RESULTS_DIR = 'results'
EXPORT_DIR = 'experiments/human_eval/exports'
MIN_ANNOTATIONS_PER_QUERY = 3  # Number of annotators needed per query
MIN_ANNOTATIONS_FOR_ANALYSIS = 50  # Minimum annotations before computing kappa

# Create directories
Path(DATABASE).parent.mkdir(parents=True, exist_ok=True)
Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)

# Initialize database
def init_db():
    """Initialize SQLite database for annotations."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    # Annotators table
    c.execute('''
        CREATE TABLE IF NOT EXISTS annotators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            annotator_id TEXT UNIQUE NOT NULL,
            name TEXT,
            email TEXT,
            background TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Annotations table
    c.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            annotator_id TEXT NOT NULL,
            query_id TEXT NOT NULL,
            system_name TEXT NOT NULL,
            query_text TEXT NOT NULL,
            response_text TEXT NOT NULL,
            factuality INTEGER NOT NULL CHECK(factuality BETWEEN 1 AND 5),
            relevance INTEGER NOT NULL CHECK(relevance BETWEEN 1 AND 5),
            completeness INTEGER NOT NULL CHECK(completeness BETWEEN 1 AND 5),
            fluency INTEGER NOT NULL CHECK(fluency BETWEEN 1 AND 5),
            safety INTEGER NOT NULL CHECK(safety BETWEEN 1 AND 5),
            comments TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (annotator_id) REFERENCES annotators (annotator_id),
            UNIQUE(annotator_id, query_id, system_name)
        )
    ''')
    
    # Query tracking table
    c.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            query_id TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            dataset TEXT,
            num_annotations INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Load results from experiment directory
def load_results_from_experiments(dataset: str = 'stackoverflow', seed: int = 42, num_samples: int = 100):
    """
    Load query-response pairs from experiment results.
    
    Expected structure:
    results/
        {dataset}/
            {system}/
                {seed}/
                    test_results.json
    
    Returns:
        List of Dict with query_id, query_text, and responses from each system
    """
    results_dir = Path(RESULTS_DIR) / dataset
    queries = []
    
    systems = ['standard_rag', 'naive_writeback', 'bidirectional_rag']
    
    # Load test results from each system
    system_results = {}
    for system in systems:
        results_file = results_dir / system / str(seed) / 'test_results.json'
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                system_results[system] = json.load(f)
        else:
            print(f"[WARNING] Results not found: {results_file}")
            return []
    
    # Match queries across systems
    if not system_results:
        return []
    
    # Use first system's queries as reference
    first_system = list(system_results.keys())[0]
    reference_results = system_results[first_system]
    
    # Sample up to num_samples queries
    sample_size = min(num_samples, len(reference_results))
    sampled_results = reference_results[:sample_size]
    
    for idx, ref_result in enumerate(sampled_results):
        query_id = f"{dataset}_{seed}_{idx}"
        query_text = ref_result.get('ground_truth', ref_result.get('query', ''))
        
        # Get response from each system
        responses = {}
        for system in systems:
            if system in system_results and idx < len(system_results[system]):
                result = system_results[system][idx]
                responses[system] = result.get('response', 'No response available')
            else:
                responses[system] = 'No response available'
        
        queries.append({
            'query_id': query_id,
            'query_text': query_text,
            'responses': responses,
            'dataset': dataset,
            'seed': seed
        })
        
        # Store query in database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('''
            INSERT OR IGNORE INTO queries (query_id, query_text, dataset)
            VALUES (?, ?, ?)
        ''', (query_id, query_text, dataset))
        conn.commit()
        conn.close()
    
    return queries

# Routes
@app.route('/')
def index():
    """Main page - annotation interface."""
    if 'annotator_id' not in session:
        return redirect(url_for('login'))
    
    # Load a query for annotation
    query = get_next_query_for_annotator(session['annotator_id'])
    
    if not query:
        return render_template('complete.html', annotator_id=session['annotator_id'])
    
    return render_template('annotate.html', query=query)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login/registration page."""
    if request.method == 'POST':
        annotator_id = request.form.get('annotator_id', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        background = request.form.get('background', '').strip()
        
        if not annotator_id:
            return render_template('login.html', error='Annotator ID required')
        
        # Register annotator
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('''
            INSERT OR IGNORE INTO annotators (annotator_id, name, email, background)
            VALUES (?, ?, ?, ?)
        ''', (annotator_id, name, email, background))
        conn.commit()
        conn.close()
        
        session['annotator_id'] = annotator_id
        return redirect(url_for('index'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout and clear session."""
    session.clear()
    return redirect(url_for('login'))

@app.route('/submit_annotation', methods=['POST'])
def submit_annotation():
    """Submit annotation for a query-system pair."""
    if 'annotator_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    annotator_id = session['annotator_id']
    query_id = data.get('query_id')
    system_name = data.get('system_name')
    query_text = data.get('query_text')
    response_text = data.get('response_text')
    
    ratings = {
        'factuality': int(data.get('factuality', 3)),
        'relevance': int(data.get('relevance', 3)),
        'completeness': int(data.get('completeness', 3)),
        'fluency': int(data.get('fluency', 3)),
        'safety': int(data.get('safety', 3))
    }
    
    comments = data.get('comments', '').strip()
    
    # Validate ratings
    for key, value in ratings.items():
        if not (1 <= value <= 5):
            return jsonify({'error': f'Invalid {key} rating'}), 400
    
    # Save annotation
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    try:
        c.execute('''
            INSERT OR REPLACE INTO annotations 
            (annotator_id, query_id, system_name, query_text, response_text,
             factuality, relevance, completeness, fluency, safety, comments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (annotator_id, query_id, system_name, query_text, response_text,
              ratings['factuality'], ratings['relevance'], ratings['completeness'],
              ratings['fluency'], ratings['safety'], comments))
        
        # Update query annotation count
        c.execute('''
            UPDATE queries SET num_annotations = (
                SELECT COUNT(DISTINCT annotator_id) 
                FROM annotations 
                WHERE query_id = ?
            ) WHERE query_id = ?
        ''', (query_id, query_id))
        
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/get_next_query')
def get_next_query():
    """Get next query for annotation."""
    if 'annotator_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    query = get_next_query_for_annotator(session['annotator_id'])
    
    if not query:
        return jsonify({'complete': True})
    
    return jsonify(query)

@app.route('/stats')
def stats():
    """Show annotation statistics."""
    if 'annotator_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    # Get annotator stats
    c.execute('''
        SELECT COUNT(*) FROM annotations WHERE annotator_id = ?
    ''', (session['annotator_id'],))
    annotator_count = c.fetchone()[0]
    
    # Get total stats
    c.execute('SELECT COUNT(*) FROM annotations')
    total_count = c.fetchone()[0]
    
    # Get per-system counts
    c.execute('''
        SELECT system_name, COUNT(*) 
        FROM annotations 
        GROUP BY system_name
    ''')
    system_counts = dict(c.fetchall())
    
    # Get kappa if enough annotations
    kappa = None
    if total_count >= MIN_ANNOTATIONS_FOR_ANALYSIS:
        try:
            kappa = compute_kappa()
        except:
            pass
    
    conn.close()
    
    return render_template('stats.html',
                         annotator_count=annotator_count,
                         total_count=total_count,
                         system_counts=system_counts,
                         kappa=kappa)

@app.route('/export')
def export():
    """Export annotations to CSV."""
    if 'annotator_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    c.execute('''
        SELECT 
            annotator_id,
            query_id,
            system_name,
            query_text,
            response_text,
            factuality,
            relevance,
            completeness,
            fluency,
            safety,
            comments,
            created_at
        FROM annotations
        ORDER BY query_id, system_name, annotator_id
    ''')
    
    rows = c.fetchall()
    conn.close()
    
    # Export to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_file = Path(EXPORT_DIR) / f'annotations_{timestamp}.csv'
    
    with open(export_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'annotator_id', 'query_id', 'system_name', 'query_text', 'response_text',
            'factuality', 'relevance', 'completeness', 'fluency', 'safety',
            'comments', 'created_at'
        ])
        writer.writerows(rows)
    
    return jsonify({
        'success': True,
        'file': str(export_file),
        'count': len(rows)
    })

# Helper functions
def get_next_query_for_annotator(annotator_id: str) -> Optional[Dict]:
    """Get next query-system pair that needs annotation."""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    # Get queries that need more annotations
    c.execute('''
        SELECT q.query_id, q.query_text
        FROM queries q
        WHERE q.num_annotations < ?
        ORDER BY q.query_id
        LIMIT 1
    ''', (MIN_ANNOTATIONS_PER_QUERY,))
    
    query_row = c.fetchone()
    if not query_row:
        conn.close()
        return None
    
    query_id, query_text = query_row
    
    # Get systems for this query
    c.execute('''
        SELECT DISTINCT system_name, response_text
        FROM annotations
        WHERE query_id = ?
        LIMIT 1
    ''', (query_id,))
    
    # Try to get from original results if not in DB yet
    # For now, use placeholder - will be loaded from results files
    systems = ['standard_rag', 'naive_writeback', 'bidirectional_rag']
    
    # Get which systems this annotator has already rated
    c.execute('''
        SELECT system_name FROM annotations
        WHERE annotator_id = ? AND query_id = ?
    ''', (annotator_id, query_id))
    rated_systems = {row[0] for row in c.fetchall()}
    
    # Find next unrated system
    next_system = None
    for system in systems:
        if system not in rated_systems:
            next_system = system
            break
    
    conn.close()
    
    if not next_system:
        return None
    
    # Load response from results file (simplified - should load from actual results)
    response_text = "Loading from results..."  # Placeholder
    
    return {
        'query_id': query_id,
        'query_text': query_text,
        'system_name': next_system,
        'response_text': response_text
    }

def compute_kappa() -> float:
    """
    Compute Cohen's kappa for inter-annotator agreement.
    
    Kappa interpretation:
    < 0: No agreement
    0.00-0.20: Slight agreement
    0.21-0.40: Fair agreement
    0.41-0.60: Moderate agreement
    0.61-0.80: Substantial agreement
    0.81-1.00: Almost perfect agreement
    """
    try:
        from sklearn.metrics import cohen_kappa_score
        import numpy as np
    except ImportError:
        return None
    
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    # Get annotations grouped by query-system pairs
    c.execute('''
        SELECT query_id, system_name, annotator_id,
               factuality, relevance, completeness, fluency, safety
        FROM annotations
        ORDER BY query_id, system_name, annotator_id
    ''')
    
    rows = c.fetchall()
    conn.close()
    
    if len(rows) < 2:
        return None
    
    # Group by query-system pairs
    groups = {}
    for row in rows:
        key = (row[0], row[1])  # (query_id, system_name)
        if key not in groups:
            groups[key] = []
        groups[key].append(row[2:])  # annotator_id and ratings
    
    # Compute kappa for each criterion
    kappas = []
    for criterion_idx in range(4):  # 0: factuality, 1: relevance, 2: completeness, 3: fluency, 4: safety
        ratings_list = []
        for group in groups.values():
            if len(group) >= 2:
                # Get ratings for this criterion
                ratings = [item[criterion_idx + 1] for item in group]  # +1 to skip annotator_id
                if len(set(ratings)) > 1:  # Only if there's variation
                    ratings_list.append(ratings)
        
        if len(ratings_list) >= 2:
            # Compute pairwise kappa
            pairwise_kappas = []
            for ratings in ratings_list:
                if len(ratings) >= 2:
                    # Compute kappa between pairs
                    for i in range(len(ratings)):
                        for j in range(i + 1, len(ratings)):
                            kappa = cohen_kappa_score([ratings[i]], [ratings[j]])
                            if not np.isnan(kappa):
                                pairwise_kappas.append(kappa)
            
            if pairwise_kappas:
                avg_kappa = np.mean(pairwise_kappas)
                kappas.append(avg_kappa)
    
    return np.mean(kappas) if kappas else None

# Initialize database on startup
init_db()

if __name__ == '__main__':
    print("=" * 70)
    print("HUMAN EVALUATION INTERFACE")
    print("=" * 70)
    print(f"Database: {DATABASE}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Export directory: {EXPORT_DIR}")
    print("\nStarting Flask server on http://localhost:5000")
    print("\nTo access:")
    print("  1. Open browser: http://localhost:5000")
    print("  2. Register as annotator")
    print("  3. Start annotating queries")
    print("\nPress Ctrl+C to stop")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

