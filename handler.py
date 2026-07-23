import runpod, base64, os, subprocess, tempfile, glob, shutil

ESR = "/opt/realesrgan/realesrgan-ncnn-vulkan"
MODELS = "/opt/realesrgan/models"

def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)

def handler(job):
    inp = job.get("input", {}) or {}
    vb64 = inp.get("video") or inp.get("source_video")
    if not vb64:
        return {"error": "no video in input"}
    model = inp.get("model", "realesr-animevideov3-x2")
    scale = int(inp.get("scale", 2))
    work = tempfile.mkdtemp()
    try:
        vin = os.path.join(work, "in.mp4")
        with open(vin, "wb") as f:
            f.write(base64.b64decode(vb64))
        fps = subprocess.run(
            ["ffprobe","-v","0","-select_streams","v:0","-show_entries","stream=r_frame_rate","-of","csv=p=0",vin],
            capture_output=True, text=True).stdout.strip() or "16"
        frames = os.path.join(work, "frames"); out = os.path.join(work, "out")
        os.makedirs(frames); os.makedirs(out)
        run(["ffmpeg","-y","-i",vin, os.path.join(frames,"f%05d.png")])
        n = len(glob.glob(os.path.join(frames,"*.png")))
        if n == 0:
            return {"error": "no frames extracted"}
        run([ESR,"-i",frames,"-o",out,"-n",model,"-s",str(scale),"-f","png","-m",MODELS,"-g","0"])
        vout = os.path.join(work, "out.mp4")
        run(["ffmpeg","-y","-framerate",fps,"-i",os.path.join(out,"f%05d.png"),"-i",vin,
             "-map","0:v","-map","1:a?","-c:v","libx264","-crf","18","-preset","fast",
             "-pix_fmt","yuv420p","-c:a","aac","-shortest","-movflags","+faststart",vout])
        with open(vout, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return {"video": data, "frames": n}
    except subprocess.CalledProcessError as e:
        return {"error": "proc: " + (e.stderr.decode()[:600] if e.stderr else str(e))}
    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(work, ignore_errors=True)

runpod.serverless.start({"handler": handler})
