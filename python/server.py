import os
import sys
import io
import time
import shutil
import threading
import subprocess
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Force UTF-8 stdout/stderr encoding on Windows to prevent UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

app = FastAPI(title="BaysianPatrol RL Training API Server")

# Enable CORS for Vite frontend (http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Training State Tracker
class TrainingState:
    is_training: bool = False
    progress_percent: float = 0.0
    current_timestep: int = 0
    total_timesteps: int = 50000
    mean_reward: float = 0.0
    status_message: str = "Prêt à démarrer l'entraînement."
    process: Optional[subprocess.Popen] = None
    tensorboard_process: Optional[subprocess.Popen] = None

state = TrainingState()

class StartTrainingRequest(BaseModel):
    total_timesteps: int = 50000
    model_name: str = "PPO_8"
    learning_rate: float = 0.0003
    reward_detection: float = 500.0
    reward_exploration: float = 10.0
    penalty_time: float = 0.1
    penalty_bingo: float = 1000.0

class SelectModelRequest(BaseModel):
    model_filename: str

@app.get("/api/status")
def get_status():
    return {
        "is_training": state.is_training,
        "progress_percent": state.progress_percent,
        "current_timestep": state.current_timestep,
        "total_timesteps": state.total_timesteps,
        "mean_reward": state.mean_reward,
        "status_message": state.status_message,
    }

def get_next_ppo_name() -> str:
    tb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "tensorboard_logs"))
    max_idx = 7
    if os.path.exists(tb_dir):
        for d in os.listdir(tb_dir):
            if d.startswith("PPO_"):
                try:
                    idx = int(d.replace("PPO_", ""))
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    pass
    return f"PPO_{max_idx + 1}"

def run_training_background(
    total_timesteps: int,
    model_name: str,
    learning_rate: float,
    reward_detection: float,
    reward_exploration: float,
    penalty_time: float,
    penalty_bingo: float,
):
    try:
        state.is_training = True
        state.total_timesteps = total_timesteps
        state.status_message = f"Entraînement {model_name} en cours ({total_timesteps} pas)..."
        state.progress_percent = 5.0

        train_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "train.py"))
        cmd = [
            sys.executable, train_script,
            "--timesteps", str(total_timesteps),
            "--name", model_name,
            "--lr", str(learning_rate),
            "--r_det", str(reward_detection),
            "--r_prob", str(reward_exploration),
            "--p_time", str(penalty_time),
            "--p_bingo", str(penalty_bingo),
        ]
        
        proc = subprocess.Popen(
            cmd,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        state.process = proc

        # Read stdout in real-time
        if proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                line_clean = line.strip()
                if "total_timesteps" in line_clean:
                    try:
                        parts = [p.strip() for p in line_clean.split("|") if p.strip()]
                        if len(parts) >= 2 and parts[0] == "total_timesteps":
                            ts = int(parts[1])
                            state.current_timestep = ts
                            state.progress_percent = min(95.0, (ts / total_timesteps) * 100.0)
                    except Exception:
                        pass
                elif "ep_rew_mean" in line_clean:
                    try:
                        parts = [p.strip() for p in line_clean.split("|") if p.strip()]
                        if len(parts) >= 2 and parts[0] == "ep_rew_mean":
                            state.mean_reward = float(parts[1])
                    except Exception:
                        pass

        proc.wait()
        state.progress_percent = 100.0
        state.status_message = f"🎉 Entraînement {model_name} terminé et activé avec succès !"
    except Exception as e:
        state.status_message = f"Erreur lors de l'entraînement : {str(e)}"
    finally:
        state.is_training = False
        state.process = None

@app.post("/api/train/start")
def start_training(req: StartTrainingRequest, background_tasks: BackgroundTasks):
    if state.is_training:
        raise HTTPException(status_code=400, detail="Un entraînement est déjà en cours.")
    
    custom_name = req.model_name.strip() if req.model_name else ""
    if not custom_name or custom_name == "baysian_patrol_policy":
        custom_name = get_next_ppo_name()

    background_tasks.add_task(
        run_training_background,
        req.total_timesteps,
        custom_name,
        req.learning_rate,
        req.reward_detection,
        req.reward_exploration,
        req.penalty_time,
        req.penalty_bingo,
    )
    return {"message": f"Entraînement {custom_name} démarré avec succès.", "model_name": custom_name}

def run_autoresearch_background(num_experiments: int, timesteps_per_exp: int):
    try:
        state.is_training = True
        state.total_timesteps = num_experiments * timesteps_per_exp
        state.status_message = f"Mode AutoResearch (Karpathy Loop) en cours ({num_experiments} experiences)..."
        state.progress_percent = 5.0

        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "autoresearch.py"))
        cmd = [
            sys.executable, script_path,
            "--experiments", str(num_experiments),
            "--timesteps", str(timesteps_per_exp)
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        state.process = proc

        if proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                line_clean = line.strip()
                if "Expérimentation" in line_clean:
                    state.status_message = line_clean
                elif "NOUVEAU CHAMPION" in line_clean:
                    state.status_message = line_clean
                elif "ÉVALUATION" in line_clean:
                    state.status_message = line_clean

        proc.wait()
        state.progress_percent = 100.0
        state.status_message = "Boucle AutoResearch terminee avec succes ! Modele ONNX Champion active."
    except Exception as e:
        state.status_message = f"Erreur AutoResearch: {str(e)}"
    finally:
        state.is_training = False
        state.process = None

@app.post("/api/train/autoresearch")
def start_autoresearch(background_tasks: BackgroundTasks, experiments: int = 5, timesteps: int = 25000):
    if state.is_training:
        raise HTTPException(status_code=400, detail="Un entraînement est déjà en cours.")
    background_tasks.add_task(run_autoresearch_background, experiments, timesteps)
    return {"message": "Boucle AutoResearch autonome démarrée avec succès."}

@app.get("/api/autoresearch/results")
def get_autoresearch_results():
    results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "autoresearch_results.json"))
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"best_score": -999999.0, "best_params": None, "history": []}

@app.post("/api/tensorboard/clear")
def clear_tensorboard_logs():
    tb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "tensorboard_logs"))
    if os.path.exists(tb_dir):
        for item in os.listdir(tb_dir):
            item_path = os.path.join(tb_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception:
                pass
    return {"message": "Logs TensorBoard purgés avec succès. Graphiques réinitialisés."}

@app.post("/api/train/stop")
def stop_training():
    if state.process and state.is_training:
        state.process.terminate()
        state.is_training = False
        state.status_message = "Entraînement interrompu par l'utilisateur."
        return {"message": "Entraînement interrompu."}
    return {"message": "Aucun entraînement actif."}

from export_onnx import export_onnx_model

@app.get("/api/models")
def list_models():
    public_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public", "models"))
    python_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    os.makedirs(public_models_dir, exist_ok=True)
    os.makedirs(python_models_dir, exist_ok=True)

    zip_files = [f for f in os.listdir(python_models_dir) if f.endswith(".zip")]
    models_info = []

    for f in zip_files:
        path = os.path.join(python_models_dir, f)
        size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d/%m/%Y à %H:%M")
        
        # Human-readable title mapping
        if "CHAMPION_AUTORESEARCH" in f:
            title = f"🏆 Modèle Champion AutoResearch (Invention Autonome du {mtime})"
        elif "14h30" in f or f == "baysian_patrol_policy.zip":
            title = f"🌟 Vrai PPO_5 (Votre Run Long du {mtime} - Score Élevé TensorBoard)"
        elif "ppo_baysian_patrol.zip" in f:
            title = f"🧠 Run PPO_3 (Entraînement du {mtime})"
        elif "PPO_5.zip" in f:
            title = f"🧪 Run PPO_7 (Test du {mtime})"
        else:
            title = f"🧠 Modèle {f.replace('.zip', '')} ({mtime})"

        models_info.append({
            "filename": f,
            "title": title,
            "type": "Modèle RL PPO",
            "size_mb": round(size_mb, 2),
            "date": mtime
        })

    # Sort so Champion AutoResearch is #1, Vrai PPO_5 is #2
    models_info.sort(key=lambda m: 0 if "CHAMPION" in m["filename"] else (1 if "Vrai PPO_5" in m["title"] else 2))

    active_default = "PPO_CHAMPION_AUTORESEARCH.zip" if any("CHAMPION" in m["filename"] for m in models_info) else "baysian_patrol_policy.zip"
    return {"models": models_info, "active": active_default}

@app.post("/api/models/select")
def select_model(req: SelectModelRequest):
    public_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public", "models"))
    python_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    dest_path = os.path.join(public_models_dir, "baysian_patrol_policy.onnx")

    if req.model_filename.endswith(".zip"):
        src_zip = os.path.join(python_models_dir, req.model_filename)
        if not os.path.exists(src_zip):
            raise HTTPException(status_code=404, detail=f"Fichier zip {req.model_filename} introuvable.")
        export_onnx_model(zip_model_path=src_zip, output_path=dest_path)
        return {"message": f"Modèle PyTorch {req.model_filename} exporté et activé avec succès !"}
    else:
        src_onnx = os.path.join(public_models_dir, req.model_filename)
        if not os.path.exists(src_onnx):
            raise HTTPException(status_code=404, detail=f"Fichier ONNX {req.model_filename} introuvable.")
        if src_onnx != dest_path:
            shutil.copyfile(src_onnx, dest_path)
        return {"message": f"Modèle ONNX {req.model_filename} activé avec succès !"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
