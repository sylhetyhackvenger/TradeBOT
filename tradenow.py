#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import base64
from datetime import datetime
import numpy as np
import cv2
from collections import deque, Counter
from flask import Flask, request, jsonify, render_template_string

# ---------- Pattern detection functions (same as before) ----------
def body_length(c):
    return abs(c['close'] - c['open'])

def upper_shadow(c):
    return c['high'] - max(c['open'], c['close'])

def lower_shadow(c):
    return min(c['open'], c['close']) - c['low']

def is_bullish(c):
    return c['close'] > c['open']

def is_bearish(c):
    return c['close'] < c['open']

def is_doji(c, tol=0.1):
    return body_length(c) <= tol * (c['high'] - c['low'])

def is_spinning_top(c, ratio=0.4):
    return body_length(c) < ratio * (c['high'] - c['low'])

def is_hammer(c):
    b = body_length(c); ls = lower_shadow(c); us = upper_shadow(c)
    return (ls > 2*b) and (us < 0.3*b) and (c['close'] < c['open'] + 0.5*b)

def is_inverted_hammer(c):
    b = body_length(c); us = upper_shadow(c); ls = lower_shadow(c)
    return (us > 2*b) and (ls < 0.3*b) and (c['close'] < c['open'] + 0.5*b)

def is_marubozu(c, tol=0.1):
    total = c['high'] - c['low']
    return (upper_shadow(c) < tol*total) and (lower_shadow(c) < tol*total)

def is_dragonfly_doji(c):
    return is_doji(c) and (lower_shadow(c) > 2*body_length(c))

def is_gravestone_doji(c):
    return is_doji(c) and (upper_shadow(c) > 2*body_length(c))

def is_bullish_engulfing(p, c):
    return is_bearish(p) and is_bullish(c) and (c['open'] < p['close']) and (c['close'] > p['open'])

def is_bearish_engulfing(p, c):
    return is_bullish(p) and is_bearish(c) and (c['open'] > p['close']) and (c['close'] < p['open'])

def is_bullish_harami(p, c):
    return is_bearish(p) and is_bullish(c) and (c['open'] > p['close']) and (c['close'] < p['open'])

def is_bearish_harami(p, c):
    return is_bullish(p) and is_bearish(c) and (c['open'] < p['close']) and (c['close'] > p['open'])

def is_piercing(p, c):
    mid = (p['open'] + p['close']) / 2
    return is_bearish(p) and is_bullish(c) and (c['close'] > mid) and (c['open'] < p['close'])

def is_dark_cloud_cover(p, c):
    mid = (p['open'] + p['close']) / 2
    return is_bullish(p) and is_bearish(c) and (c['close'] < mid) and (c['open'] > p['close'])

def is_tweezer_bottom(c1, c2, tol=0.01):
    return abs(c1['low'] - c2['low']) <= tol * (c1['high'] - c1['low'])

def is_tweezer_top(c1, c2, tol=0.01):
    return abs(c1['high'] - c2['high']) <= tol * (c1['high'] - c1['low'])

def is_morning_star(c1, c2, c3):
    if not is_bearish(c1) or not is_bullish(c3): return False
    if body_length(c2) > 0.5*body_length(c1): return False
    return (c2['high'] < c1['close']) and (c3['open'] > c2['high']) and (c3['close'] > (c1['open']+c1['close'])/2)

def is_evening_star(c1, c2, c3):
    if not is_bullish(c1) or not is_bearish(c3): return False
    if body_length(c2) > 0.5*body_length(c1): return False
    return (c2['high'] < c1['close']) and (c3['open'] > c2['high']) and (c3['close'] < (c1['open']+c1['close'])/2)

def is_three_white_soldiers(c1, c2, c3):
    return (is_bullish(c1) and is_bullish(c2) and is_bullish(c3) and
            (c2['close'] > c1['close']) and (c3['close'] > c2['close']) and
            (c2['open'] < c2['close']) and (c3['open'] < c3['close']) and
            (c2['open'] > c1['open']) and (c3['open'] > c2['open']))

def is_three_black_crows(c1, c2, c3):
    return (is_bearish(c1) and is_bearish(c2) and is_bearish(c3) and
            (c2['close'] < c1['close']) and (c3['close'] < c2['close']) and
            (c2['open'] < c2['close']) and (c3['open'] < c3['close']) and
            (c2['open'] < c1['open']) and (c3['open'] < c2['open']))

def is_three_inside_up(c1, c2, c3):
    return is_bearish(c1) and is_bullish_harami(c1, c2) and is_bullish(c3) and (c3['close'] > c2['high'])

def is_three_inside_down(c1, c2, c3):
    return is_bullish(c1) and is_bearish_harami(c1, c2) and is_bearish(c3) and (c3['close'] < c2['low'])

def is_three_outside_up(c1, c2, c3):
    return is_bearish(c1) and is_bullish_engulfing(c1, c2) and is_bullish(c3) and (c3['close'] > c2['high'])

def is_three_outside_down(c1, c2, c3):
    return is_bullish(c1) and is_bearish_engulfing(c1, c2) and is_bearish(c3) and (c3['close'] < c2['low'])

def is_rising_three_method(c1, c2, c3, c4, c5):
    if not is_bullish(c1) or not is_bullish(c5) or c5['close'] < c1['high']: return False
    for c in [c2,c3,c4]:
        if is_bullish(c): return False
        if c['high'] > c1['high'] or c['low'] < c1['low']: return False
    return True

def is_falling_three_method(c1, c2, c3, c4, c5):
    if not is_bearish(c1) or not is_bearish(c5) or c5['close'] > c1['low']: return False
    for c in [c2,c3,c4]:
        if is_bearish(c): return False
        if c['high'] > c1['high'] or c['low'] < c1['low']: return False
    return True

def is_gap_up(p, c): return c['low'] > p['high']
def is_gap_down(p, c): return c['high'] < p['low']

def detect_single(c):
    pat = []
    if is_doji(c): pat.append("Doji")
    if is_spinning_top(c): pat.append("Spinning Top")
    if is_hammer(c): pat.append("Hammer")
    if is_inverted_hammer(c): pat.append("Inverted Hammer")
    if is_marubozu(c):
        pat.append("Bullish Marubozu" if is_bullish(c) else "Bearish Marubozu")
    if is_dragonfly_doji(c): pat.append("Dragonfly Doji")
    if is_gravestone_doji(c): pat.append("Gravestone Doji")
    return pat

def detect_two(p, c):
    pat = []
    if is_bullish_engulfing(p, c): pat.append("Bullish Engulfing")
    if is_bearish_engulfing(p, c): pat.append("Bearish Engulfing")
    if is_bullish_harami(p, c): pat.append("Bullish Harami")
    if is_bearish_harami(p, c): pat.append("Bearish Harami")
    if is_piercing(p, c): pat.append("Piercing Pattern")
    if is_dark_cloud_cover(p, c): pat.append("Dark Cloud Cover")
    if is_tweezer_bottom(p, c): pat.append("Tweezer Bottom")
    if is_tweezer_top(p, c): pat.append("Tweezer Top")
    if is_gap_up(p, c): pat.append("Gap Up")
    if is_gap_down(p, c): pat.append("Gap Down")
    return pat

def detect_three(c1,c2,c3):
    pat = []
    if is_morning_star(c1,c2,c3): pat.append("Morning Star")
    if is_evening_star(c1,c2,c3): pat.append("Evening Star")
    if is_three_white_soldiers(c1,c2,c3): pat.append("Three White Soldiers")
    if is_three_black_crows(c1,c2,c3): pat.append("Three Black Crows")
    if is_three_inside_up(c1,c2,c3): pat.append("Three Inside Up")
    if is_three_inside_down(c1,c2,c3): pat.append("Three Inside Down")
    if is_three_outside_up(c1,c2,c3): pat.append("Three Outside Up")
    if is_three_outside_down(c1,c2,c3): pat.append("Three Outside Down")
    return pat

def detect_five(c1,c2,c3,c4,c5):
    pat = []
    if is_rising_three_method(c1,c2,c3,c4,c5): pat.append("Rising Three Method")
    if is_falling_three_method(c1,c2,c3,c4,c5): pat.append("Falling Three Method")
    return pat

def detect_all(ohlc_list):
    patterns = []
    n = len(ohlc_list)
    if n == 0: return patterns
    patterns.extend(detect_single(ohlc_list[-1]))
    if n >= 2:
        patterns.extend(detect_two(ohlc_list[-2], ohlc_list[-1]))
    if n >= 3:
        patterns.extend(detect_three(ohlc_list[-3], ohlc_list[-2], ohlc_list[-1]))
    if n >= 5:
        patterns.extend(detect_five(ohlc_list[-5], ohlc_list[-4], ohlc_list[-3], ohlc_list[-2], ohlc_list[-1]))
    return list(set(patterns))

# ---------- Pattern memory (to predict next pattern) ----------
class PatternMemory:
    def __init__(self, max_hist=2000):
        self.history = deque(maxlen=max_hist)
        self.sequence = deque(maxlen=5)

    def add(self, pat_list):
        combined = ",".join(pat_list) if pat_list else "None"
        self.history.append(combined)
        self.sequence.append(combined)

    def predict_next(self, last_n=4):
        if len(self.history) < last_n + 1:
            return None
        seq = tuple(list(self.sequence)[-last_n:])
        next_pats = []
        for i in range(len(self.history) - last_n - 1):
            if tuple(self.history[i:i+last_n]) == seq:
                next_pats.append(self.history[i+last_n])
        if not next_pats:
            return None
        freq = Counter(next_pats)
        return freq.most_common(1)[0][0]

# ---------- Flask app ----------
app = Flask(__name__)
memory = PatternMemory()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Candlestick Pattern Predictor</title>
    <style>
        body { font-family: Arial; max-width: 900px; margin: auto; padding: 20px; background: #0d1117; color: #c9d1d9; }
        h1 { text-align: center; color: #58a6ff; }
        .card { background: #161b22; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: center; border-bottom: 1px solid #30363d; }
        input { width: 100px; padding: 6px; background: #0d1117; color: #fff; border: 1px solid #30363d; border-radius: 4px; }
        button { padding: 10px 20px; background: #238636; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
        button:hover { background: #2ea043; }
        #result { margin-top: 15px; padding: 15px; background: #0d1117; border-left: 4px solid #58a6ff; white-space: pre-wrap; }
        .signal-buy { border-left-color: #3fb950; }
        .signal-sell { border-left-color: #f85149; }
        .signal-hold { border-left-color: #d29922; }
        .info { color: #8b949e; font-size: 0.9em; }
        .upload-area { border: 2px dashed #30363d; padding: 20px; text-align: center; border-radius: 8px; }
    </style>
</head>
<body>
    <h1>📊 Candlestick Pattern Predictor</h1>
    <div class="card">
        <h3>Enter the OHLC of the last 4 candles</h3>
        <p class="info">Candle 1 = oldest, Candle 4 = most recent (just closed).</p>
        <form id="ohlcForm">
            <table>
                <tr><th>Candle</th><th>Open</th><th>High</th><th>Low</th><th>Close</th></tr>
                <tr><td>1</td><td><input name="o1" placeholder="14.83902" required></td><td><input name="h1" placeholder="14.84500" required></td><td><input name="l1" placeholder="14.83500" required></td><td><input name="c1" placeholder="14.84000" required></td></tr>
                <tr><td>2</td><td><input name="o2" placeholder="14.84000" required></td><td><input name="h2" placeholder="14.84600" required></td><td><input name="l2" placeholder="14.83600" required></td><td><input name="c2" placeholder="14.84200" required></td></tr>
                <tr><td>3</td><td><input name="o3" placeholder="14.84200" required></td><td><input name="h3" placeholder="14.84800" required></td><td><input name="l3" placeholder="14.83800" required></td><td><input name="c3" placeholder="14.84400" required></td></tr>
                <tr><td>4</td><td><input name="o4" placeholder="14.84400" required></td><td><input name="h4" placeholder="14.84900" required></td><td><input name="l4" placeholder="14.84000" required></td><td><input name="c4" placeholder="14.84600" required></td></tr>
            </table>
            <br>
            <button type="button" onclick="analyze()">🔮 Predict Next Candle</button>
        </form>
    </div>
    <div class="card">
        <h3>📸 Upload chart screenshot (optional, for reference)</h3>
        <div class="upload-area">
            <input type="file" id="imageUpload" accept="image/*">
            <br><br>
            <img id="preview" style="max-width:100%; display:none;">
        </div>
    </div>
    <div id="result">Enter 4 candles and click "Predict Next Candle".</div>

    <script>
        function analyze() {
            const o = [document.getElementsByName('o1')[0].value, document.getElementsByName('o2')[0].value,
                       document.getElementsByName('o3')[0].value, document.getElementsByName('o4')[0].value];
            const h = [document.getElementsByName('h1')[0].value, document.getElementsByName('h2')[0].value,
                       document.getElementsByName('h3')[0].value, document.getElementsByName('h4')[0].value];
            const l = [document.getElementsByName('l1')[0].value, document.getElementsByName('l2')[0].value,
                       document.getElementsByName('l3')[0].value, document.getElementsByName('l4')[0].value];
            const c = [document.getElementsByName('c1')[0].value, document.getElementsByName('c2')[0].value,
                       document.getElementsByName('c3')[0].value, document.getElementsByName('c4')[0].value];
            if (o.some(v => v==='') || h.some(v => v==='') || l.some(v => v==='') || c.some(v => v==='')) {
                alert('Please fill all fields.');
                return;
            }
            const candles = [];
            for (let i=0; i<4; i++) {
                candles.push({open: parseFloat(o[i]), high: parseFloat(h[i]), low: parseFloat(l[i]), close: parseFloat(c[i])});
            }
            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({candles: candles})
            })
            .then(r => r.json())
            .then(data => {
                const res = document.getElementById('result');
                let html = `<strong>Analysis:</strong><br>`;
                html += `Detected patterns: ${data.patterns || 'None'}<br>`;
                html += `<strong>Prediction:</strong> ${data.direction} (${data.confidence}%)<br>`;
                html += `Expected size: ${data.size}<br>`;
                html += `Likely pattern: ${data.pattern_name || 'N/A'}<br>`;
                res.innerHTML = html;
                res.className = data.direction.toLowerCase().includes('buy') ? 'signal-buy' :
                                data.direction.toLowerCase().includes('sell') ? 'signal-sell' : 'signal-hold';
            });
        }

        // Image upload preview (just for visual)
        document.getElementById('imageUpload').addEventListener('change', function(e) {
            const reader = new FileReader();
            reader.onload = function(ev) {
                const img = document.getElementById('preview');
                img.src = ev.target.result;
                img.style.display = 'block';
            };
            reader.readAsDataURL(e.target.files[0]);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'candles' not in data:
        return jsonify({'error': 'Missing candles'}), 400
    candles = data['candles']
    if len(candles) < 4:
        return jsonify({'error': 'Need at least 4 candles'}), 400

    # Convert to list of dicts with float values
    ohlc_list = [{'open': float(c['open']), 'high': float(c['high']), 'low': float(c['low']), 'close': float(c['close'])} for c in candles]

    # Detect patterns in the sequence
    patterns = detect_all(ohlc_list)
    patterns_str = ', '.join(patterns) if patterns else 'None'

    # Add to memory for future predictions
    memory.add(patterns)

    # Predict next pattern from memory
    predicted_pattern = memory.predict_next()
    if predicted_pattern:
        # Determine direction from pattern name
        pred_l = predicted_pattern.lower()
        if any(w in pred_l for w in ['hammer', 'bullish', 'engulfing', 'morning', 'piercing', 'marubozu', 'tweezer bottom', 'three white', 'rising']):
            direction = 'BUY'
            confidence = 70
        elif any(w in pred_l for w in ['shooting', 'bearish', 'hanging', 'dark cloud', 'evening', 'three black', 'falling']):
            direction = 'SELL'
            confidence = 70
        else:
            direction = 'HOLD'
            confidence = 50
        pattern_name = predicted_pattern
    else:
        # Fallback: use the last candle's direction
        last = ohlc_list[-1]
        if is_bullish(last):
            direction = 'BUY'
            confidence = 55
        elif is_bearish(last):
            direction = 'SELL'
            confidence = 55
        else:
            direction = 'HOLD'
            confidence = 50
        pattern_name = 'No clear pattern'

    # Estimate size based on average body length of previous candles
    avg_body = np.mean([abs(c['close'] - c['open']) for c in ohlc_list])
    last_body = abs(ohlc_list[-1]['close'] - ohlc_list[-1]['open'])
    if last_body > 1.5 * avg_body:
        size = 'Long'
    elif last_body < 0.5 * avg_body:
        size = 'Short'
    else:
        size = 'Medium'

    return jsonify({
        'patterns': patterns_str,
        'direction': direction,
        'confidence': confidence,
        'size': size,
        'pattern_name': pattern_name
    })

if __name__ == '__main__':
    print("🌐 Starting server at http://localhost:5000")
    print("📌 Enter OHLC of the last 4 candles and click Predict.")
    print("   Optionally upload a chart screenshot for reference.")
    app.run(host='0.0.0.0', port=5000, debug=False)
 
