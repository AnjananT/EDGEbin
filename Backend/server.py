from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*')

stats = {'electronic': 25, 'organic': 20, 'recycle': 35, 'trash': 15}

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify(stats)

@app.route('/update_stats', methods=['POST'])
def update_stats():
    data = request.get_json()
    trash_class = data.get('trash_class')

    if trash_class in stats:
        stats[trash_class]+=1
        print(f'Statistics updated for {trash_class}: {stats}')

        socketio.emit('stats_update', stats)
    
    return jsonify(stats)

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    global stats
    data = request.get_json()
    trash_class = data.get('trash_class')

    if trash_class in stats:
        stats[trash_class] = 0
        print('Statistics for {trash_class} set to zero')
        socketio.emit('stats_update', stats)
        
    return jsonify(stats)

if __name__ == '__main__':
    socketio.run(app, debug=True)

