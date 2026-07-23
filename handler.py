import runpod, base64, os, subprocess, tempfile, glob, shutil, traceback
import cv2, torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

MODELS = "/models"
_half = torch.cuda.is_available()
_rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
_up = RealESRGANer(scale=4, model_path=f"{MODELS}/RealESRGAN_x4plus.pth", model=_rrdb,
                   tile=400, tile_pad=10, pre_pad=0, half=_half)
_gfp = GFPGANer(model_path=f"{MODELS}/GFPGANv1.4.pth", upscale=2, arch="clean",
                channel_multiplier=2, bg_upsampler=_up)

def run(c):
    subprocess.run(c, check=True, capture_output=True)

def handler(job):
    inp = job.get("input", {}) or {}
    vb64 = inp.get("video") or inp.get("source_video")
    if not vb64:
        return {"error": "no video in input"}
    face = bool(inp.get("face_enhance", True))
    outscale = float(inp.get("scale", 2))
    work = tempfile.mkdtemp()
    try:
        vin = os.path.join(work, "in.mp4")
        with open(vin, "wb") as f:
            f.write(base64.b64decode(vb64))
        fps = subprocess.run(["ffprobe","-v","0","-select_streams","v:0","-show_entries","stream=r_frame_rate","-of","csv=p=0",vin],
                             capture_output=True, text=True).stdout.strip() or "16"
        fr = os.path.join(work, "fr"); ou = os.path.join(work, "ou")
        os.makedirs(fr); os.makedirs(ou)
        run(["ffmpeg","-y","-i",vin, os.path.join(fr,"f%05d.png")])
        files = sorted(glob.glob(os.path.join(fr,"*.png")))
        if not files:
            return {"error": "no frames extracted"}
        for fp in files:
            img = cv2.imread(fp, cv2.IMREAD_COLOR)
            if face:
                _, _, res = _gfp.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5)
            else:
                res, _ = _up.enhance(img, outscale=outscale)
            cv2.imwrite(os.path.join(ou, os.path.basename(fp)), res)
        vout = os.path.join(work, "out.mp4")
        run(["ffmpeg","-y","-framerate",fps,"-i",os.path.join(ou,"f%05d.png"),"-i",vin,
             "-map","0:v","-map","1:a?","-c:v","libx264","-crf","18","-preset","fast",
             "-pix_fmt","yuv420p","-c:a","aac","-shortest","-movflags","+faststart",vout])
        with open(vout,"rb") as f:
            return {"video": base64.b64encode(f.read()).decode(), "frames": len(files)}
    except subprocess.CalledProcessError as e:
        return {"error": "proc: " + (e.stderr.decode()[:500] if e.stderr else str(e))}
    except Exception as e:
        return {"error": str(e)[:300], "tb": traceback.format_exc()[-400:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)

runpod.serverless.start({"handler": handler})
