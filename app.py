
import os
import cv2
from flask import Flask, render_template_string, Response
import gymnasium as gym
import gymnasium.wrappers as gym_wrap
from DQN_model import Agent, SkipFrame

app = Flask(__name__)

# Chemin fixe vers TON modèle
MODEL_PATH = "./training/saved_models/DQN_1600ep_FAST_final_254790.pt"

def generate_stream():
    # Créer l'environnement
    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = SkipFrame(env, skip=4)
    env = gym_wrap.GrayscaleObservation(env)
    env = gym_wrap.ResizeObservation(env, shape=(84, 84))
    env = gym_wrap.FrameStackObservation(env, stack_size=4)
    
    # Charger l'agent avec TON modèle
    state, _ = env.reset()
    agent = Agent(
        state.shape,
        env.action_space.n,
        load_state='eval',
        load_model=os.path.basename(MODEL_PATH)
    )
    agent.epsilon = 0  # Mode évaluation pur
    print(f"Modèle chargé : {MODEL_PATH}")

    episode_reward = 0

    try:
        while True:
            action = agent.take_action(state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

            frame = env.render()
            if frame is not None:
                # Afficher le score sur la vidéo
                cv2.putText(
                    frame,
                    f"Score: {episode_reward:6.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),  # Blanc
                    2,
                    cv2.LINE_AA
                )
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            if done or truncated:
                print(f"Épisode terminé | Score final: {episode_reward:.1f}")
                episode_reward = 0
                state, _ = env.reset()

    finally:
        env.close()

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CarRacing - DQN 1600ep (Modèle Final)</title>
        <style>
            body { background: #0a0a0a; color: #00ffcc; text-align: center; font-family: 'Courier New', monospace; padding: 20px; }
            h1 { margin-bottom: 15px; text-shadow: 0 0 8px #00ffcc; }
            .video-container { border: 2px solid #00ffcc; border-radius: 8px; display: inline-block; overflow: hidden; box-shadow: 0 0 12px rgba(0, 255, 204, 0.4); }
            img { display: block; }
            .info { margin-top: 12px; color: #aaa; font-size: 14px; }
        </style>
    </head>
    <body>
        <h1> Agent DQN - 1600 Épisodes </h1>
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Agent playing">
        </div>
        
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Vérifier que le modèle existe
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f" Modèle introuvable : {os.path.abspath(MODEL_PATH)}")
    
    print(" Démarrage de l'interface web...")
    print(f" Modèle utilisé : {MODEL_PATH}")
    print(" Ouvrez http://localhost:5000 dans votre navigateur")
    app.run(host='0.0.0.0', port=5000, threaded=True)
